from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier





class MyQueue:
    def __init__(self, max_len, dim):
        self.max_len = max_len
        self.dim = dim
        self.len = 0
        self.data = np.zeros([max_len, dim], dtype=np.float32)
        self.labels = np.zeros([max_len], dtype=np.int32)
        self.index = 0

    def add(self, x, y):
        if x.shape[0] > self.max_len:
            raise ValueError(
                f"输入数据长度 {x.shape[0]} 超过队列最大长度 {self.max_len}"
            )

        n = x.shape[0]
        if self.len + n <= self.max_len:
            # 队列未满，直接添加数据
            self.data[self.index : self.index + n] = x
            self.labels[self.index : self.index + n] = y

            self.len += n
            self.index = (self.index + n) % self.max_len
        else:
            # 队列已满，处理循环队列
            remaining_space = self.max_len - self.len
            self.data[self.index : self.index + remaining_space] = x[:remaining_space]
            self.data[: n - remaining_space] = x[remaining_space:]

            self.labels[self.index : self.index + remaining_space] = y[:remaining_space]
            self.labels[: n - remaining_space] = y[remaining_space:]
            self.index = (self.index + n) % self.max_len
            self.len = self.max_len

    def get(self):
        if self.len < self.max_len:
            return self.data[: self.len], self.labels[: self.len]
        else:
            return self.data, self.labels


class OSTTA_NEIGH(nn.Module):
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
        gamma=0.99,
        nr=2,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = criterion
        
        # 原来时1000
        self.queue = MyQueue(2000,128)
        self.model0 = deepcopy(self.model)
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
                self.optimizer,
                self.alpha,
                self.criterion,
                self.model0,
                self.model,
                self.queue,
                self.nr,
            )

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, optimizer, alpha, criterion, model0, model, queue, nr):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs0 = model0(x)
    outputs0 = outputs0.softmax(1)
    values0, indices0 = outputs0.max(1)
    outputs = model(x)
    outputs_ = outputs.softmax(1)
    labels = torch.argmax(outputs, dim=1)

    # 获取该模型该样本在源模型预测的这个类别上的概率
    values = outputs_[torch.arange(outputs0.size(0)), indices0]

    # 局部
    model1 = deepcopy(model0)
    model1.fc = nn.Identity()
    features = model1(x)
    features = features.detach().cpu().numpy()
    # nbrs = NearestNeighbors(n_neighbors=2)
    all_data, all_labels = queue.get()
    if all_data.shape[0] == 0:
        all_data, all_labels = features, labels.detach().cpu().numpy()

    # 使用knn分类
    knn = KNeighborsClassifier(weights="distance", n_neighbors=nr).fit(
        all_data, all_labels
    )

    pred = knn.predict(features)

    close_ind = []
    open_ind = []

    for i in range(outputs.size(0)):
        # 得到距离样本i，最近的样本
        if values[i] >= values0[i] and labels[i] == pred[i]:
            close_ind.append(i)
        elif values[i] < values0[i] and labels[i] != pred[i]:
            open_ind.append(i)

    # 把闭集样本添加到队列中去
    queue.add(features[close_ind], labels[close_ind].detach().cpu().numpy())
    # adapt
    entropys = softmax_entropy(outputs)

    # loss = entropys[values >= values0].mean(0)
    # print("len", len(close_ind), len(open_ind))
    loss = 0
    if len(close_ind) != 0:
        loss = entropys[close_ind].mean(0)
    if len(open_ind) != 0:
        loss -= alpha[1] * entropys[open_ind].mean(0)

    loss -= alpha[0] * softmax_mean_entropy(outputs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # update ema-model

    return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = (
            alpha_teacher * ema_param[:].data[:]
            + (1 - alpha_teacher) * param[:].data[:]
        )
    return ema_model


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
