import torch
import torch.nn.functional as F

def InfoNCE_loss(q, K):
    '''
    @Author: WangSuxiao
    @description: 计算InfoNCE损失
    @param {Any} q : (batch_size, embedding_size)           查询张量
    @param {Any} K : [(batch_size, embedding_size), ...]    样本集合
    @return {Any}
    '''
    # 计算normal特征分布距离
    pos_scores = torch.stack([torch.sum(q * k, dim=1) for k in K], dim=1)
    # 计算负样本得分
    neg_scores = torch.mm(q, q.t())
    neg_scores = neg_scores - torch.diag(torch.diag(neg_scores))
    # 计算正负样本的 logit
    logits = torch.cat([pos_scores, neg_scores], dim=1)

    # 计算 InfoNCE 损失
    labels = torch.zeros(q.size(0), dtype=torch.long).to(q.device)
    loss = F.cross_entropy(logits, labels)

    return loss
    # # 计算正样本得分
    # pos_scores = torch.stack([torch.sum(q * k, dim=1) for k in K], dim=1)
    # # 计算负样本得分
    # neg_scores = torch.mm(q, q.t())
    # neg_scores = neg_scores - torch.diag(torch.diag(neg_scores))
    # # 计算正负样本的 logit
    # logits = torch.cat([pos_scores, neg_scores], dim=1)

    # # 计算 InfoNCE 损失
    # labels = torch.zeros(q.size(0), dtype=torch.long).to(q.device)
    # loss = F.cross_entropy(logits, labels)

    # return loss

class FewShotLoss(torch.nn.Module):
    """使用小样本学习训练双图神经网络的损失函数"""
    def __init__(self, w1, w2, w3, labels):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} w1 : L_ins的权重
        @param {Any} w2 : L_dis的权重
        @param {Any} w3 : L_contrast的权重
        @param {Any} labels :
        @return {Any}
        '''
        super(FewShotLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.labels = labels

    def forward(self, Yins, Ydis, Y):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} Yins : (batch_size, embedding_size)                    双图网络的最终预测
        @param {Any} Ydis : (batch_size, sample_size, embedding_size)       分布图的各层预测 `\sum (dis_e_ij, y_j)`
        @param {Any} Y    : (batch_size, embedding_size)                    标签值
        @return {Any}
        '''
        # 控制双图网络的整体优化方向，模型预测与标签值越接近损失越小
        L_ins = F.cross_entropy(Yins, Y)
        # 通过控制分布图的优化方向，对分布图预测做惩罚(L_ins损失仅通过实例图权重)
        L_dis_tmp = [F.cross_entropy(item, Y) for item in Ydis]
        L_dis = torch.sum(torch.stack(L_dis_tmp))
        # 借鉴对比学习思想，拉大正常样本特征与异常样本特征在特征空间中的距离
        # 获取正负例的索引
        # nornal = (tensor == torch.tensor([1, 0])).all(dim=1).nonzero().squeeze()
        # abnormal = (tensor == torch.tensor([0, 1])).all(dim=1).nonzero().squeeze()
        normal_indexs = (Y == self.labels[0]).all(dim=1).nonzero().squeeze()
        abnormal_indexs = (Y == self.labels[1]).all(dim=1).nonzero().squeeze()
        normals = Yins[normal_indexs]
        abnormals = Yins[abnormal_indexs]
        L_contrast = [(normal, abnormals) for normal in normals]
        L1 = F.cross_entropy(Yins, Y)

        # Calculate loss for Ydis
        L2 = 0
        for dis_output in Ydis:
            L2 += F.cross_entropy(dis_output, Y)
        L2 /= len(Ydis)  # Average loss over layers

        # Combine losses
        L = self.a * L1 + self.b * L2
        return L
if __name__ == "__main__":
    # # Example usage:
    # a = 0.5  # Weight for L1
    # b = 0.5  # Weight for L2
    # loss_function = FewShotLoss(a, b)

    # # Example inputs
    # Yins = torch.randn(10, 3)  # Random predictions from ins network
    # Ydis = [torch.randn(10, 3) for _ in range(5)]  # Random predictions from dis network, 5 layers
    # Y = torch.randint(0, 3, (10,))  # Random labels

    # # Calculate loss
    # loss = loss_function(Yins, Ydis, Y)
    # print("Total Loss:", loss.item())


    import torch

    # # 测试获取正常样本
    # tensor = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # class_1_indices = (tensor == torch.tensor([0, 0, 1])).all(dim=1).nonzero().squeeze()
    # class_2_indices = (tensor == torch.tensor([0, 1, 0])).all(dim=1).nonzero().squeeze()
    # class_3_indices = (tensor == torch.tensor([1, 0, 0])).all(dim=1).nonzero().squeeze()
    # class_1 = tensor[class_1_indices]
    # class_2 = tensor[class_2_indices]
    # class_3 = tensor[class_3_indices]
    # print("Class 1:", class_1)
    # print("Class 2:", class_2)
    # print("Class 3:", class_3)


    import torch

    # 假设有三个类，标签分别为1，2，3
    labels = torch.tensor([1, 2, 3])

    # 将标签转换为 one-hot 编码
    one_hot = torch.nn.functional.one_hot(labels - 1, num_classes=3)

    print("One-hot 编码：", one_hot)

