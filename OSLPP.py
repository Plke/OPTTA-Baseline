import math
from sklearn.decomposition import PCA
import scipy
import numpy as np
import scipy.io
import scipy.linalg
from dataclasses import dataclass

def _load_tensors(domain):
    mapping = {
        'art': 'Art',
        'clipart': 'Clipart',
        'product': 'Product',
        'real_world': 'RealWorld'
    }
    mat = scipy.io.loadmat(f'mats/OfficeHome-{mapping[domain]}-resnet50-noft.mat')
    features, labels = mat['resnet50_features'], mat['labels']
    features, labels = features[:,:,0,0], labels[0]
    assert len(features) == len(labels)
    # features, labels = torch.tensor(features), torch.tensor(labels)
    # features = torch.load(f'./data_handling/features/OH_{domain}_features.pt')
    # labels = torch.load(f'./data_handling/features/OH_{domain}_labels.pt')
    return features, labels

def create_datasets(source, target, num_src_classes, num_total_classes):
    """
    创建源域和目标域的数据集。

    该函数从给定的数据源中加载数据，筛选出属于指定类别范围内的样本，并对目标域的标签进行调整，以适应后续的训练需求。

    参数:
    - source: 源域数据的来源路径或标识。
    - target: 目标域数据的来源路径或标识。
    - num_src_classes: 源域中的类别数量。
    - num_total_classes: 总的类别数量，包括目标域中特有的类别。

    返回:
    - 源域的数据集，包括特征和标签。
    - 目标域的数据集，包括特征和调整后的标签。
    """

    # 从源域数据源加载特征和标签
    src_features, src_labels = _load_tensors(source)
    # 筛选源域中属于指定类别的样本
    idxs = src_labels < num_src_classes
    src_features, src_labels = src_features[idxs], src_labels[idxs]

    # 从目标域数据源加载特征和标签
    tgt_features, tgt_labels = _load_tensors(target)
    # 筛选目标域中属于指定类别的样本
    idxs = tgt_labels < num_total_classes
    tgt_features, tgt_labels = tgt_features[idxs], tgt_labels[idxs]
    # 调整目标域中不属于源域的标签，将其映射到一个单独的类别
    tgt_labels[tgt_labels >= num_src_classes] = num_src_classes

    # 确保源域标签覆盖了所有源类别，并且目标域标签覆盖了所有可能的类别
    assert (np.unique(src_labels) == np.arange(0, num_src_classes)).all()
    assert (np.unique(tgt_labels) == np.arange(0, num_src_classes+1)).all()
    # 确保特征和标签的数量匹配
    assert len(src_features) == len(src_labels)
    assert len(tgt_features) == len(tgt_labels)

    # 返回源域和目标域的数据集
    return (src_features, src_labels), (tgt_features, tgt_labels)

def get_l2_norm(features:np.ndarray): return np.sqrt(np.square(features).sum(axis=1)).reshape((-1,1))

def get_l2_normalized(features:np.ndarray): return features / get_l2_norm(features)

def get_PCA(features, dim):
    result = PCA(n_components=dim).fit_transform(features)
    assert len(features) == len(result)
    return result

def get_W(labels,):
    """
    根据给定的标签生成一个特定的权重矩阵W。
    
    该函数首先通过比较每个标签与所有其他标签是否相等来创建矩阵W，
    如果两个标签相等，则对应位置置为1，否则置为0。然后，对于所有标签为-1的位置，
    将其对应的行和列中的所有值都置为0，以表示这些标签不参与权重计算。
    
    参数:
    labels: 一维数组，包含每个样本的标签。
    
    返回:
    W: 二维数组，生成的权重矩阵。
    """
    # 根据labels生成初始的W矩阵，如果两个标签相等，则W[i][j]为1，否则为0
    W = (labels.reshape(-1,1) == labels).astype(np.int)
    
    # 找到所有标签为-1的索引
    negative_one_idxs = np.where(labels == -1)[0]
    
    # 将标签为-1的行和列在W矩阵中置为0
    W[:,negative_one_idxs] = 0
    W[negative_one_idxs,:] = 0
    
    return W

def get_D(W):
    """
    计算并返回矩阵W的度矩阵D。

    度矩阵D是一个对角矩阵，其对角线上的元素是矩阵W的各行元素之和。
    该函数通过将W矩阵的每一行求和，并将结果放置在对角线上来构建度矩阵。

    参数:
    W: numpy数组，表示输入的权重矩阵。

    返回值:
    D: numpy数组，表示计算得到的度矩阵。
    """
    # 使用numpy的eye函数创建一个单位矩阵，其大小与W矩阵相同，dtype参数指定矩阵元素类型为int
    # 然后将该单位矩阵的每一行元素与W矩阵对应行的元素求和结果相乘，得到度矩阵D
    return np.eye(len(W), dtype=np.int) * W.sum(axis=1)

def fix_numerical_assymetry(M):
    """
    修复数值非对称性。

    该函数旨在处理矩阵M，使其不对称的元素变得对称。通过将矩阵与其转置矩阵相加并除以2，
    可以确保结果矩阵关于主对角线对称。这一操作常用于需要对数值数据进行对称化处理的场景。

    参数:
    M: 数值矩阵，可以是二维numpy数组或其他可进行转置操作的矩阵对象。

    返回值:
    一个新的矩阵，该矩阵是对称的，即满足M[i][j] = M[j][i]对于所有i和j。
    """
    return (M + M.transpose()) * 0.5

def get_projection_matrix(features, labels, proj_dim):
    """
    获取投影矩阵。

    根据给定的特征和标签计算一个投影矩阵，该投影矩阵用于将数据投影到一个较低维度的空间中。
    这个函数实现了拉普拉斯特征映射（Laplacian Eigenmaps），它基于图论来保持数据的局部结构。

    参数:
    features: numpy数组，形状为(N, d)，表示N个d维特征向量。
    labels: numpy数组，形状为(N,)，表示每个特征向量的类别标签。
    proj_dim: int，投影后的维度。

    返回:
    v: numpy数组，形状为(d, proj_dim)，表示投影矩阵。

    """
    # 获取特征矩阵的行数和列数
    N, d = features.shape
    # 转置特征矩阵
    X = features.transpose()
    
    # 计算节点间权重矩阵W
    W = get_W(labels)
    # 计算度矩阵D
    D = get_D(W)
    # 计算拉普拉斯矩阵L
    L = D - W

    # 计算矩阵A，用于后续的特征值问题
    A = fix_numerical_assymetry(np.matmul(np.matmul(X, D), X.transpose()))
    # 计算矩阵B，用于后续的特征值问题
    B = fix_numerical_assymetry(np.matmul(np.matmul(X, L), X.transpose()) + np.eye(d))
    # 确保A和B都是对称矩阵
    assert (A.transpose() == A).all() and (B.transpose() == B).all()

    # 求解广义特征值问题，获取特征值w和特征向量v
    w, v = scipy.linalg.eigh(A, B)
    # 确保特征值按升序排列
    assert w[0] < w[-1]
    # 选择最大的proj_dim个特征值和对应的特征向量
    w, v = w[-proj_dim:], v[:, -proj_dim:]
    # 验证特征值和特征向量的正确性
    assert np.abs(np.matmul(A, v) - w * np.matmul(B, v)).max() < 1e-5

    # 反转特征值和特征向量的顺序，以便按降序排列
    w = np.flip(w)
    v = np.flip(v, axis=1)

    # 调整特征向量的符号，使得第一个元素非负
    for i in range(v.shape[1]):
        if v[0,i] < 0:
            v[:,i] *= -1
    # 返回投影矩阵
    return v

def project_features(P, features):
    """
    使用给定的投影矩阵P对特征矩阵进行投影变换。

    参数:
    P: 投影矩阵，形状为(pca_dim, proj_dim)，其中pca_dim是主成分分析的维度，proj_dim是投影后的维度。
    features: 待投影的特征矩阵，形状为(N, pca_dim)，其中N是特征的数量。

    返回值:
    result: 投影变换后的特征矩阵，形状为(N, proj_dim)。

    描述:
    该函数的主要作用是将一组特征从pca_dim维度空间投影到proj_dim维度空间。投影矩阵P定义了如何进行这种变换。
    """
    # P: pca_dim x proj_dim
    # features: N x pca_dim
    # result: N x proj_dim
    return np.matmul(P.transpose(), features.transpose()).transpose()

def get_centroids(features, labels): 
    centroids = np.stack([features[labels == c].mean(axis=0) for c in np.unique(labels)], axis=0)
    centroids = get_l2_normalized(centroids)
    return centroids

def get_dist(f, features):
    return get_l2_norm(f - features)

def get_closed_set_pseudo_labels(features_S, labels_S, features_T):
    """
    获取闭集的伪标签和伪概率。
    
    该函数首先计算源域特征的类中心，然后测量目标域特征与这些类中心的距离，
    最终基于这些距离分配伪标签和伪概率给目标域的每个样本。
    
    参数:
    - features_S: 源域的特征矩阵。
    - labels_S: 源域的标签，用于计算类中心。
    - features_T: 目标域的特征矩阵，其伪标签和伪概率将被计算。
    
    返回:
    - pseudo_labels: 目标域样本的伪标签。
    - pseudo_probs: 目标域样本的伪概率。
    """
    # 计算源域各类的中心
    centroids = get_centroids(features_S, labels_S)
    
    # 计算目标域每个特征与所有类中心的距离
    dists = np.stack([get_dist(f, centroids)[:,0] for f in features_T], axis=0)
    
    # 为每个目标域特征选择距离最近的类中心作为伪标签
    pseudo_labels = np.argmin(dists, axis=1)
    
    # 计算每个目标域特征对于其伪标签的伪概率
    pseudo_probs = np.exp(-dists[np.arange(len(dists)), pseudo_labels]) / np.exp(-dists).sum(axis=1)
    
    return pseudo_labels, pseudo_probs

def select_initial_rejected(pseudo_probs, n_r):
    """
    根据伪概率选择初始拒绝的项目。

    这个函数的作用是通过给定的伪概率值来确定哪些项目最初被拒绝。它会选择伪概率最低的n_r个项目，并标记为拒绝。

    参数:
    pseudo_probs: ndarray，包含所有项目的伪概率值。
    n_r: int，要拒绝的项目数量。

    返回:
    is_rejected: ndarray，一个与输入伪概率数组长度相同的整数数组，用于标记每个项目是否被拒绝（1表示拒绝，0表示接受）。
    """
    # 初始化一个与伪概率数组长度相同的零数组，用于标记项目是否被拒绝
    is_rejected = np.zeros((len(pseudo_probs),), dtype=np.int)

    # 找到伪概率最低的n_r个项目的索引，并将这些项目的拒绝标记设置为1
    is_rejected[np.argsort(pseudo_probs)[:n_r]] = 1

    # 返回标记数组，表示每个项目是否被拒绝
    return is_rejected

def select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, T):
    """
    选择闭合集合的伪标签。
    
    该函数根据当前时间步t和总时间步T，以及伪标签和伪概率，选择闭合集合中的伪标签。
    闭合集合的选取基于每个类别的伪概率，确保每个类别中只有足够“自信”的样本被选中。
    
    参数:
    pseudo_labels: ndarray, 伪标签数组。
    pseudo_probs: ndarray, 伪概率数组，与伪标签一一对应。
    t: int, 当前的时间步。
    T: int, 总的时间步数。
    
    返回:
    selected: ndarray, 与伪标签形状相同的数组，表示选中的闭合集合伪标签。
    """
    # 确保t不超过T-1，因为当t等于T时，没有类别可以被选中
    if t >= T: t = T - 1
    # 初始化selected数组，用于存放选中的伪标签
    selected = np.zeros_like(pseudo_labels)
    # 遍历每个独特的伪标签类别
    for c in np.unique(pseudo_labels):
        # 找到所有属于类别c的索引
        idxs = np.where(pseudo_labels == c)[0]
        Nc = len(idxs)
        # 当类别c有至少一个样本时，计算该类别的阈值并选择高于阈值的样本
        if Nc > 0:
            # 获取属于类别c的所有样本的伪概率
            class_probs = pseudo_probs[idxs]
            # 对类别的伪概率进行排序，以便确定阈值
            class_probs = np.sort(class_probs)
            # 根据当前时间步t和总时间步T，计算阈值
            threshold = class_probs[math.floor(Nc*(1-t/(T-1)))]
            # 找到高于阈值的样本的索引
            idxs2 = idxs[pseudo_probs[idxs] > threshold]
            # 确保这些样本在selected数组中尚未被选中
            assert (selected[idxs2] == 0).all()
            # 将这些样本标记为选中
            selected[idxs2] = 1
    # 返回选中的闭合集合伪标签
    return selected

def update_rejected(selected, rejected, features_T):
    """
    更新拒绝样本集合。

    该函数的目的是根据当前已选择和已拒绝的样本，以及所有未标记样本的特征，
    更新拒绝样本集合。具体策略是计算每个未标记样本与已选择和已拒绝样本集的
    最小距离，如果一个未标记样本更接近已拒绝样本集，那么将其加入拒绝样本集合。

    参数:
    - selected: 数组，标记了已选择的样本，1表示已选择，0表示未选择。
    - rejected: 数组，标记了已拒绝的样本，1表示已拒绝，0表示未拒绝。
    - features_T: 二维数组，存储了所有样本的特征，行对应样本，列对应特征。

    返回:
    - new_is_rejected: 数组，更新后的拒绝样本标记，1表示已拒绝，0表示未拒绝。
    """

    # 确定未标记样本的索引，即那些既未被选择也未被拒绝的样本
    unlabeled = (selected == 0) * (rejected == 0)

    # 创建拒绝样本标记的副本，用于在更新过程中保持原数据不变
    new_is_rejected = rejected.copy()

    # 遍历未标记样本，计算其与已选择和已拒绝样本集的最小距离，并据此更新拒绝样本集合
    for idx in np.where(unlabeled)[0]:
        # 计算当前未标记样本与已选择样本集的最小距离
        dist_to_selected = get_dist(features_T[idx], features_T[selected == 1]).min()
        # 计算当前未标记样本与已拒绝样本集的最小距离
        dist_to_rejected = get_dist(features_T[idx], features_T[rejected == 1]).min()
        # 如果当前未标记样本更接近已拒绝样本集，则将其标记为拒绝样本
        if dist_to_rejected < dist_to_selected:
            new_is_rejected[idx] = 1

    # 返回更新后的拒绝样本标记
    return new_is_rejected

def evaluate(predicted, labels, num_src_classes):
    acc_unk = (predicted[labels == num_src_classes] == labels[labels == num_src_classes]).mean()
    accs = [(predicted[labels == c] == labels[labels == c]).mean() for c in range(num_src_classes)]
    acc_common = np.array(accs).mean()
    hos = 2 * acc_unk * acc_common / (acc_unk + acc_common)
    _os = np.array(accs+[acc_unk]).mean()
    return f'OS={_os*100:.2f} OS*={acc_common*100:.2f} unk={acc_unk*100:.2f} HOS={hos*100:.2f}'

@dataclass
class Params:
    pca_dim: int # = 512
    proj_dim: int # = 128
    T: int # = 10
    n_r: int #  = 1200
    dataset: str # = 'OfficeHome'
    source: str # = 'art'
    target: str # = 'clipart'
    num_src_classes: int # = 25
    num_total_classes: int # = 65

def do_l2_normalization(feats_S, feats_T):
    """
    对两个特征集合进行L2归一化处理。

    确保每个特征向量的L2范数为1，这是在许多机器学习任务中常用的预处理步骤，
    可以提高模型的性能和训练速度。特别是，这对于距离度量和基于梯度的方法特别有效。

    参数:
    feats_S: 第一个特征集合，通常代表源域或学生模型的特征。
    feats_T: 第二个特征集合，通常代表目标域或教师模型的特征。

    返回:
    feats_S: L2归一化后的第一个特征集合。
    feats_T: L2归一化后的第二个特征集合。
    """
    # 对两个特征集合分别进行L2归一化处理。
    feats_S, feats_T = get_l2_normalized(feats_S), get_l2_normalized(feats_T)
    
    # 确保归一化后的特征集合的L2范数为1。
    assert np.abs(get_l2_norm(feats_S) - 1.).max() < 1e-5
    assert np.abs(get_l2_norm(feats_T) - 1.).max() < 1e-5

    # 返回归一化后的特征集合。
    return feats_S, feats_T

def do_pca(feats_S, feats_T, pca_dim):
    """
    对源特征和目标特征执行主成分分析（PCA）降维。
    
    该函数首先将源特征和目标特征合并，然后在同一数据集上计算PCA，
    最后将降维后的特征重新分割为与原始形状相对应的源特征和目标特征。
    
    参数:
    feats_S: 源特征矩阵，期望形状为 (n_samples_S, n_features)。
    feats_T: 目标特征矩阵，期望形状为 (n_samples_T, n_features)。
    pca_dim: PCA降维后的维度。
    
    返回:
    feats_S: 降维后的源特征矩阵，形状为 (n_samples_S, pca_dim)。
    feats_T: 降维后的目标特征矩阵，形状为 (n_samples_T, pca_dim)。
    """
    # 合并源特征和目标特征，以便统一进行PCA处理
    feats = np.concatenate([feats_S, feats_T], axis=0)
    # 执行PCA降维，将合并后的特征矩阵降至pca_dim维
    feats = get_PCA(feats, pca_dim)
    # 根据原始源特征的长度，将降维后的特征重新分割为源特征和目标特征
    feats_S, feats_T = feats[:len(feats_S)], feats[len(feats_S):]
    return feats_S, feats_T

def center_and_l2_normalize(zs_S, zs_T):
    """
    对两个输入的特征向量集合进行中心化和L2规范化。
    
    首先，将两个特征向量集合连接起来，计算沿特定轴的平均值，从而得到整体的平均值。
    然后，分别将每个特征向量集合减去这个整体平均值，实现中心化。
    最后，对中心化后的两个特征向量集合进行L2规范化，确保它们的范数为1。

    参数:
    zs_S: 源特征向量集合，期望是一个二维数组，每一行代表一个特征向量。
    zs_T: 目标特征向量集合，结构同源特征向量集合相同。

    返回:
    zs_S: 中心化和L2规范化后的源特征向量集合。
    zs_T: 中心化和L2规范化后的目标特征向量集合。
    """
    
    # 计算中心化所需的平均值
    zs_mean = np.concatenate((zs_S, zs_T), axis=0).mean(axis=0).reshape((1,-1))
    
    # 对源特征向量集合进行中心化
    zs_S = zs_S - zs_mean
    
    # 对目标特征向量集合进行中心化
    zs_T = zs_T - zs_mean
    
    # 对中心化后的特征向量集合进行L2规范化
    zs_S, zs_T = do_l2_normalization(zs_S, zs_T)
    
    return zs_S, zs_T

def main(params: Params):
    """
    主函数，执行整个伪标签生成和评估流程。

    参数:
    params: Params对象，包含各种配置参数，如数据源、目标、类别数量等。
    """

    # 创建数据集，分为源域和目标域
    (feats_S, lbls_S), (feats_T, lbls_T) = create_datasets(params.source, params.target, params.num_src_classes, params.num_total_classes)

    # 对特征进行l2归一化和PCA处理
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)

    # 初始化变量和数据结构
    feats_all = np.concatenate((feats_S, feats_T), axis=0)
    pseudo_labels = -np.ones_like(lbls_T)
    rejected = np.zeros_like(pseudo_labels)

    # 开始迭代过程
    for t in range(1, params.T+1):
        # 获取投影矩阵并投影特征
        P = get_projection_matrix(feats_all, np.concatenate((lbls_S, pseudo_labels), axis=0), params.proj_dim)
        proj_S, proj_T = project_features(P, feats_S), project_features(P, feats_T)
        proj_S, proj_T = center_and_l2_normalize(proj_S, proj_T)

        # 生成伪标签和概率
        pseudo_labels, pseudo_probs = get_closed_set_pseudo_labels(proj_S, lbls_S, proj_T)
        selected = select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, t, params.T)
        selected = selected * (1-rejected)

        # 处理初始拒绝的伪标签
        if t == 2:
            rejected = select_initial_rejected(pseudo_probs, params.n_r)
        if t >= 2:
            rejected = update_rejected(selected, rejected, proj_T)
        selected = selected * (1-rejected)

        # 调整伪标签状态
        pseudo_labels[selected == 0] = -1
        pseudo_labels[rejected == 1] = -2

    # 最终调整伪标签，准备评估
    pseudo_labels[pseudo_labels == -2] = params.num_src_classes
    assert (pseudo_labels != -1).all()

    # 评估伪标签性能
    return evaluate(pseudo_labels, lbls_T, params.num_src_classes)


if __name__ == '__main__':
    params = Params(pca_dim=512, proj_dim=128, T=10, n_r=1200, 
                  dataset='OfficeHome', source='clipart', target='art',
                  num_src_classes=25, num_total_classes=65)
    print(params.source, params.target, main(params))
