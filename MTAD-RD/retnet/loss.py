import torch
import random
import torch.nn.functional as F
from util.base import mprint

def infoNCE(q, k, n, temperature=1.0):
    '''
    @Author: WangSuxiao
    @description:
    @param {Any} q : 锚点样本 (batch_size, 1, embedding_size)
    @param {Any} k : 正样本 (batch_size, k_number, embedding_size)
    @param {Any} n : 负样本 (batch_size, n_number, embedding_size)
    @param {Any} temperature : 温度参数，默认为1.0
    @return {Any}
    '''
    batch_size, k_number, embedding_size = k.size()
    positive_logits = torch.exp(torch.bmm(k, q.permute(0, 2, 1)) / temperature)  # (batch_size, k_number)
    negative_logits = torch.exp(torch.bmm(n, q.permute(0, 2, 1))/ temperature)  # (batch_size, n_number)
    p_logits = positive_logits.sum(dim=(1,2))
    n_logits = negative_logits.sum(dim=(1,2))

    return - torch.log(p_logits / n_logits).sum()


def infoNCE_loss(k, n, temperature=1.0):
    '''
    @Author: WangSuxiao
    @description:
    @param {Any} k : 正样本 (batch_size, k_number, embedding_size)
    @param {Any} n : 负样本 (batch_size, n_number, embedding_size)
    @param {Any} temperature : 温度参数，默认为1.0
    @return {Any}
    '''
    batch_size, k_number, embedding_size = k.size()
    positive_logits = torch.exp(torch.bmm(k, k.permute(0, 2, 1)) / temperature)  # (batch_size, k_number)
    negative_logits = torch.exp(torch.bmm(n, k.permute(0, 2, 1))/ temperature)  # (batch_size, n_number)
    p_logits = positive_logits.sum(dim=(1,2))
    n_logits = negative_logits.sum(dim=(1,2))
    return - torch.log(p_logits / n_logits).mean()


class FewShotLoss(torch.nn.Module):
    """使用小样本学习训练双图神经网络的损失函数"""
    def __init__(self, w1, w2, w3, labels, temperature = 1):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} w1 : L_ins的权重
        @param {Any} w2 : L_dis的权重
        @param {Any} w3 : L_contrast的权重
        @param {Any} temperature : 数据集中正负样本的比例，用与计算L_contrast时正负样本的采样
        @param {Any} labels :
        @return {Any}
        '''
        super(FewShotLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.labels = labels
        self.temperature = temperature

    def sample(self,  Ni, Y, size = (3,10)):
        '''
        @Author: WangSuxiao
        @description: batch中各个样本的abnormal和normal节点的数量，位置不一样；
                    此函数用于在单个batch中采样正常样本和异常样本；
        @param {Any} self :
        @param {Any} Ni : 单个图样本中各个节点的判别网络实例图节点输出(node_size, feature)
        @param {Any} Y : 单个图样本中各个节点的onehot标签(node_size, onthot_label)
        @param {Any} size : 异常类和正常类的数量

        @return {Any}
        '''
        # 正负样本采样, 获取正负例的索引
        prefix = "Sample"
        normal_indexs_:torch.Tensor = (Y == self.labels[0]).all(dim=1).nonzero().squeeze()
        abnormal_indexs_:torch.Tensor = (Y == self.labels[1]).all(dim=1).nonzero().squeeze()
        mprint(4, f"normal_indexs_.shape: {normal_indexs_.shape}", prefix=prefix)
        mprint(4, f"abnormal_indexs_.shape: {abnormal_indexs_.shape}", prefix=prefix)

        if abnormal_indexs_.numel() > size[0] and normal_indexs_.numel() > size[1]:
            # 数量不满足时，不再采样，返回None
            normal_indexs = random.sample(normal_indexs_.tolist(), size[1])
            abnormal_indexs = random.sample(abnormal_indexs_.tolist(), size[0])
            normal = Ni[normal_indexs]
            abnormal= Ni[abnormal_indexs]
            mprint(4, f"Ni.shape: {Ni.shape}", prefix=prefix)
            mprint(4, f"normal.shape: {normal.shape}", prefix=prefix)
            mprint(4, f"abnormal.shape: {abnormal.shape}", prefix=prefix)
            # print(Ni.shape)
            # print(normal.shape)
            # print(abnormal.shape)
            return abnormal, normal
        return None

    def forward(self, Yins, Ydis, Ni, Y):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} Yins : (batch_size, embedding_size)                    双图网络的最终预测
        @param {Any} Ydis : (batch_size, sample_size, embedding_size)       分布图的各层预测 `\sum (dis_e_ij, y_j)`
        @param {Any} Y    : (batch_size, embedding_size)                    标签值
        @return {Any}
        '''
        prefix = "Loss forward"
        # 控制双图网络的整体优化方向，模型预测与标签值越接近损失越小
        L_ins = F.cross_entropy(Yins, Y)
        # 通过控制分布图的优化方向，对分布图预测做惩罚(L_ins损失仅通过实例图权重)
        L_dis_tmp = [F.cross_entropy(item, Y) for item in Ydis]
        L_dis = torch.mean(torch.stack(L_dis_tmp))
        # 预训练的目标为拉大和缩小样本空间中的距离
        # 上下游任务的承接关系，在双图判别网络中加入对比损失
        # 拉大正常样本特征与异常样本特征在特征空间中的距离
        batch_size = Yins.shape[0]
        normal_res = []
        abnormal_res = []
        for i in range(batch_size):
            sample_res = self.sample(Ni[i],Y[i])
            if sample_res is None:
                mprint(1, f"采样失败", prefix=prefix)
            else:
                abnormal_res.append(sample_res[0])
                normal_res.append(sample_res[1])
        batch_normal = torch.stack(normal_res)
        batch_abnormal = torch.stack(abnormal_res)
        mprint(4, f"batch_normal.shape: {batch_normal.shape}", prefix=prefix)
        mprint(4, f"batch_abnormal.shape: {batch_abnormal.shape}", prefix=prefix)
        L_cont = infoNCE_loss(batch_abnormal, batch_normal, self.temperature)
        mprint(2, f"L_ins: {L_ins}", prefix=prefix)
        mprint(4, f"L_dis: {L_dis}", prefix=prefix)
        mprint(4, f"L_cont: {L_cont}", prefix=prefix)
        loss = self.w1 * L_ins + self.w2 * L_dis + self.w3 * L_cont
        mprint(4, f"loss All: {loss}", prefix=prefix + " return")
        mprint(2, f"loss : L_ins {L_ins}; L_dis {L_dis}; L_cont {L_cont}", prefix=prefix + " return")
        return loss
        return L_ins


if __name__ == "__main__":
    # Example usage:
    a = 0.5  # Weight for L1
    b = 0.5  # Weight for L2
    loss_function = FewShotLoss(a, b)

    # Example inputs
    Yins = torch.randn(10, 3)  # Random predictions from ins network
    Ydis = [torch.randn(10, 3) for _ in range(5)]  # Random predictions from dis network, 5 layers
    Y = torch.randint(0, 3, (10,))  # Random labels

    # Calculate loss
    loss = loss_function(Yins, Ydis, Y)
    print("Total Loss:", loss.item())


    # print("label.shape : ", label.shape)
    # index = torch.all(label == torch.tensor([0, 0, 1]), dim=-1)
    # print("index.shape : ", index.shape)
    # print(index)
    # data = label[index,None]
    # print(data)
    # print("---------------")
    # index = torch.all(label == torch.tensor([0, 0, 1]), dim=-1)
    # print("index.shape : ", index.shape)
    # print(index)
    # data = label[index,:]
    # print(data)
    # print("---------------")
    # index = torch.all(label == torch.tensor([0, 0, 1]), dim=-1)
    # index = index.unsqueeze(2).repeat(1,1,3)
    # print("index.shape : ", index.shape)
    # print(index)
    # data = label[index[:]]
    # print(data)



    # import torch
    # # 给出样本数据张量 X（假设形状与 label 相同）
    # X = torch.randn(2, 3, 3)  # 示例数据，随机生成，形状与 label 相同
    # label = torch.tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
    # # 找到标签为[0, 0, 1]的索引位置
    # index = torch.nonzero(torch.all(label == torch.tensor([0, 0, 1]), dim=-1))
    # # 根据索引取出对应的训练样本数据
    # selected_samples = label[index[:, 0]]
    # print(selected_samples)





    # # 测试获取正常样本
    # tensor = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    # class_1_indices = (tensor == torch.tensor([0, 0, 1])).all(dim=1).nonzero().squeeze()
    # class_2_indices = (tensor == torch.tensor([0, 1, 0])).all(dim=1).nonzero().squeeze()
    # class_3_indices = (tensor == torch.tensor([1, 0, 0])).all(dim=1).nonzero().squeeze()
    # class_1 = tensor[class_1_indices]
    # class_2 = tensor[class_2_indices]
    # class_3 = tensor[class_3_indices]
    # # print("tensor :", tensor.shape)
    # print(class_1_indices)
    # print(class_1)
    # tensor = [0, 3, 5]
    # random_elements = random.sample(tensor, 2)
    # print("随机选取的两个元素为:", random_elements)


    # 测试onehot编码
    # labels = torch.tensor([1, 2, 3])
    # print(labels)
    # # 将标签转换为 one-hot 编码
    # one_hot = torch.nn.functional.one_hot(labels - 1, num_classes=3)
    # print("One-hot 编码：", one_hot)
    # print(torch.arange(0,3))
    ...

