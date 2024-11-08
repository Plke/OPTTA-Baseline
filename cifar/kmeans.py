from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans

# fast
# from fast_pytorch_kmeans import KMeans
import numpy as np
from sklearn.decomposition import PCA


import numpy as np


class MyQueue:
    def __init__(self, max_len, dim):
        self.max_len = max_len
        self.dim = dim
        self.len = 0
        self.data = np.zeros([max_len, dim],dtype=np.float32)
        self.index = 0

    def add(self, x):
        if x.shape[0] > self.max_len:
            raise ValueError(
                f"输入数据长度 {x.shape[0]} 超过队列最大长度 {self.max_len}"
            )

        n = x.shape[0]
        if self.len + n <= self.max_len:
            # 队列未满，直接添加数据
            self.data[self.index : self.index + n] = x
            self.len += n
            self.index = (self.index + n) % self.max_len
        else:
            # 队列已满，处理循环队列
            remaining_space = self.max_len - self.len
            self.data[self.index : self.index + remaining_space] = x[:remaining_space]
            self.data[: n - remaining_space] = x[remaining_space:]
            self.index = (self.index + n) % self.max_len
            self.len = self.max_len

    def get(self):
        if self.len < self.max_len:
            return self.data[: self.len]
        else:
            return self.data


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
        # self.kmeans = KMeans(n_clusters=n_cluster, random_state=9, n_init="auto")
        # 长度为 n_cluster * 20
        self.queue = MyQueue(n_cluster * 20, 128)
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
                self.queue,
                # self.kmeans,
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


def compute_average_distances(feature_map, labels, centers, n_cluster):
    """
    计算每个类别所有样本到该类别中心的平均距离。

    :param feature_map: 特征向量，形状为 (n_samples, n_features)
    :param labels: 每个样本的类别标签，形状为 (n_samples,)
    :param centers: 每个类别的中心，形状为 (n_clusters, n_features)
    :param n_cluster: 类别数量
    :return: 每个类别的平均距离，形状为 (n_clusters,)
    """
    average_distances = np.zeros(n_cluster)

    for i in range(n_cluster):
        # 找到属于第i个类别的样本的索引
        cluster_i_indices = np.where(labels == i)[0]

        if len(cluster_i_indices) == 0:
            continue

        # 计算这些样本到中心点的距离
        distances = np.linalg.norm(feature_map[cluster_i_indices] - centers[i], axis=1)

        # 计算平均距离
        average_distance = np.mean(distances)
        average_distances[i] = average_distance

    return average_distances


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model0, model, optimizer, alpha, n_cluster, nr, queue):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """

    # forward
    features = model0(x)
    feature_map = np.array(features.detach().cpu(),dtype=np.float32)
    # queue.add(feature_map)

    # result = PCA(n_components=10).fit_transform(feature_map)
    # print(feature_map.shape)
    all_data = queue.get()
    if all_data.shape[0] == 0:
        all_data = feature_map
    # labels = kmeans.fit_predict(feature_map)
    kmeans = KMeans(n_clusters=n_cluster, random_state=9, n_init="auto")
    all_labels = kmeans.fit_predict(all_data)
    centers = kmeans.cluster_centers_
    average_distances = compute_average_distances(
        all_data, all_labels, centers, n_cluster
    )

    labels = kmeans.predict(feature_map)

    closest_sample_indices = []
    for i in range(n_cluster):
        # 找到属于第i个类别的样本的索引
        cluster_i_indices = np.where(labels == i)[0]
        # print("cluster_i_indices",cluster_i_indices)

        if len(cluster_i_indices) == 0:
            continue

        # 计算这些样本到中心点的距离

        distances = np.linalg.norm(feature_map[cluster_i_indices] - centers[i], axis=1)
        # print("distances",distances)
        closest_sample_indices.append(
            cluster_i_indices[distances < average_distances[i]]
        )
        # print(" cluster_i_indices[distances < average_distances[i]]", cluster_i_indices[distances < average_distances[i]])

        # distances = np.linalg.norm(feature_map[cluster_i_indices] - centers[i], axis=1)

        # closest_sample_indices.append()
        # # 根据距离排序并选择最近的 nr 个样本的索引
        # if len(cluster_i_indices) < nr:
        #     closest_indices = np.argsort(distances)[:]
        # else:
        #     closest_indices = np.argsort(distances)[:nr]

        # closest_sample_indices.append(cluster_i_indices[closest_indices])

    # print(111, (closest_sample_indices))
    close_set_index = np.concatenate(closest_sample_indices)
    queue.add(feature_map[close_set_index])
    # print(222, (close_set_index).shape)
    outputs = model(x)
    # print(333, outputs.shape)

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
