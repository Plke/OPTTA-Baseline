from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA


class Tent_kmeans(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        alpha=[0.5],
        criterion="ent",
        n_cluster=10,
        nr=5,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.alpha = alpha
        self.criterion = criterion

        self.model0 = deepcopy(self.model)
        self.model0.fc = nn.Identity()
        self.n_cluster = n_cluster
        self.nr = nr
        for param in self.model0.parameters():
            param.detach()

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x,
                self.model0,
                self.model,
                self.optimizer,
                self.alpha,
                self.n_cluster,
                self.nr,
            )

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model0, model, optimizer, alpha, n_cluster, nr):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    features = model0(x)
    feature_map = features.detach().cpu().numpy()

    result = PCA(n_components=10).fit_transform(feature_map)
    # print(feature_map.shape)

    # 使用kmeans分为类别数个类，然后选择每个类别中距离中心点最近的nr个样本作为闭集样本进行训练
    # 问题: 1000个类，200个测试样本，全选上了
    kmeans = KMeans(n_clusters=n_cluster, random_state=9, n_init="auto")
    kmeans.fit(result)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    closest_sample_indices = []
    for i in range(10):
        # 找到属于第i个类别的样本的索引
        cluster_i_indices = np.where(labels == i)[0]

        # 计算这些样本到中心点的距离
        distances = np.linalg.norm(result[cluster_i_indices] - centers[i], axis=1)

        # 根据距离排序并选择最近的10个样本的索引
        closest_indices = np.argsort(distances)[:nr]
        closest_sample_indices.append(cluster_i_indices[closest_indices])

    outputs = model(x)
    close_set_index = np.concatenate(closest_sample_indices)
    close_set_data = outputs[close_set_index]

    other_data = outputs[~close_set_index]

    loss = softmax_entropy(close_set_data).mean(dim=0) - alpha[1] * softmax_entropy(
        other_data
    ).mean(dim=0)

    # 正则化项
    loss -= alpha[0] * softmax_mean_entropy(outputs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " "check which require grad"
    assert not has_all_params, (
        "tent should not update all params: " "check which require grad"
    )
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
