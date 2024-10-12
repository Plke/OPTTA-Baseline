from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np


class Tent_oslpp(nn.Module):
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
        orig=False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.alpha = alpha
        self.criterion = criterion
        self.orig = orig
        self.model0 = deepcopy(self.model)
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
                x, self.model0, self.model, self.optimizer, self.alpha, self.orig
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
def forward_and_adapt(x, model0, model, optimizer, alpha, orig):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    loss = 0
    outputs = model0(x)

    # 原域的计算类的均值,使用原模型实现
    classes = outputs.shape[1]
    output_len = outputs.shape[0]
    mean_dis = np.zeros((classes, classes))
    cpu_data = outputs.detach().cpu().numpy()
    for i in range(classes):
        class_outputs = cpu_data[np.argmax(cpu_data, axis=1) == i]
        if len(class_outputs) > 0:
            mean_dis[i] = class_outputs.mean(0)
    mean_dis = torch.tensor(mean_dis).to(x.device)

    # 使用适应后的模型进行适应
    outputs = model(x)

    # 获得每个样本的预测标签
    predicted_labels = torch.empty(output_len, 1).to(x.device)
    predicted_probs = torch.empty(output_len, classes).to(x.device)

    for i, output in enumerate(outputs):
        dis = torch.norm(mean_dis - output, dim=1, p=2)
        predicted_label = torch.argmin(dis)
        predicted_prob = torch.softmax(-dis, dim=0)

        predicted_labels[i] = predicted_label
        predicted_probs[i] = predicted_prob

    # 转换为张量

    prob, _ = torch.max(predicted_probs, dim=1)
    sorted_index = torch.argsort(prob)

    #  初始拒绝样本

    # for _ in range(5):
    # 选择接受样本，每个类别概率前几的样本作为接受样本，万一这一批次没有这类样本
    # 选择概率最高的几个样本作为接受样本
    selected_sample = outputs[sorted_index[output_len - int((output_len) / 5) :]]

    # 拒绝样本选择，概率最低的几个
    reject_sample = outputs[sorted_index[: int((output_len) / 5)]]
    # distances = torch.norm(reject_sample[0] - outputs, dim=1, p=2)
    # reject_sample.append(outputs[torch.argmin(distances)])

    loss = softmax_entropy(selected_sample).mean(0) - alpha[1] * softmax_entropy(
        reject_sample
    ).mean(0)

    # 正则化项
    loss -= alpha[0] * softmax_mean_entropy(outputs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if orig:
        return predicted_probs
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
