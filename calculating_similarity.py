import numpy as np  # 数值计算与数组/矩阵操作
import copy  # 用于创建对象副本（浅拷贝/深拷贝均可）


# ========= 预处理与核带宽 =========
def Preproces_Data(A, test_id):
    """将测试集中的阳性样本在关联矩阵 A 中置 0（不改动原始矩阵）"""
    copy_A = A / 1  # 创建 A 的副本，避免修改原数据
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A


def calculate_kernel_bandwidth(A):
    """计算高斯核带宽参数（基于每行谱的 L2 范数平方的平均值的倒数）"""
    # 向量化：避免逐行循环
    A = np.asarray(A, dtype=np.float32, order="C")
    # 每行 L2 范数平方的总和
    row_norm_sq = np.sum(A * A, axis=1, dtype=np.float64)
    IP_0 = np.sum(row_norm_sq, dtype=np.float64)
    # 防御性：空矩阵时给出合理带宽
    denom = (IP_0 / float(A.shape[0])) if A.shape[0] > 0 else 1.0
    lambd = 1.0 / denom
    return float(lambd)


def calculate_GaussianKernel_sim(A):
    """基于关联谱 A 计算高斯核相似度矩阵（向量化实现）"""
    A = np.asarray(A, dtype=np.float32, order="C")
    lam = calculate_kernel_bandwidth(A)
    # 计算所有点对的欧氏距离平方：||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    A64 = A.astype(np.float64, copy=False)
    row_norm_sq = np.sum(A64 * A64, axis=1, keepdims=True)  # (n,1)
    dist2 = row_norm_sq + row_norm_sq.T - 2.0 * (A64 @ A64.T)
    # 数值稳定：裁剪为非负
    dist2 = np.maximum(dist2, 0.0)
    K = np.exp(-lam * dist2)
    return K.astype(np.float32)


# ========= 功能相似度（PBPA） =========
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    """
    计算两个 RNA（i, j）的功能相似度：
    - 取各自关联疾病集合的子矩阵
    - 分别对两方向求最大相似度再求和，并按集合大小归一化
    """
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])


def getRNA_functional_sim(RNAlen, diSiNet, rna_di):
    """构建 RNA 功能相似度网络（对称矩阵，对角线为 1）"""
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    RNASiNet = RNASiNet + np.eye(RNAlen)  # 自相似度设为 1
    return RNASiNet


# ========= 标签二值化与相似度融合 =========
def label_preprocess(sim_matrix):
    """将相似度矩阵按阈值二值化：>=0.8 置为 1，否则为 0（向量化）"""
    sm = np.asarray(sim_matrix)
    return (sm >= 0.8).astype(np.float32)


def RNA_fusion_sim(G1, G2, F, threshold=0.1):
    """
    融合两种高斯相似度与功能相似度（向量化）：
    - 当 F[i][j] > threshold 时，优先采用 F[i][j]
    - 否则取 (G1+G2)/2
    - 最后二值化处理
    """
    G1 = np.asarray(G1)
    G2 = np.asarray(G2)
    F = np.asarray(F)
    G = (G1 + G2) / 2.0
    fusion_sim = np.where(F > threshold, F, G)
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


def dis_fusion_sim(G1, G2, SD):
    """融合两种疾病高斯相似度与语义相似度：先均值再二值化"""
    fusion_sim = (SD + (G1 + G2) / 2) / 2
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


# ========= 示例入口 =========
if __name__ == '__main__':
    # 使用 dataset1 的示例数据
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association.txt")
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    from log_output_manager import get_logger
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # 使用 dataset2 的示例数据（注意：原路径文本中使用了 dataset1，可能为笔误，保留原样）
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # 示例：使用全部样本进行计算（测试集置零流程保留在注释中）
    # lnc_dis_test_id = np.loadtxt("dataset1/lnc_dis_test_id1.txt")
    # mi_dis_test_id = np.loadtxt("dataset1/mi_dis_test_id1.txt")
    # mi_lnc_test_id = np.loadtxt("dataset1/mi_lnc_test_id1.txt")
    # lnc_dis = Preproces_Data(lnc_dis, lnc_dis_test_id)
    # mi_dis = Preproces_Data(mi_dis, mi_dis_test_id)
    # mi_lnc = Preproces_Data(lnc_mi.T, mi_lnc_test_id)

    # 计算 lncRNA 相似度
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)
    lnc_fun = getRNA_functional_sim(RNAlen=len(lnc_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(lnc_dis))
    lnc_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

    # 计算 miRNA 相似度
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)
    mi_fun = getRNA_functional_sim(RNAlen=len(mi_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(mi_dis))
    mi_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

    # 计算疾病相似度
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)
    dis_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)