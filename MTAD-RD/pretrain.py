import numpy as np
import torch
import argparse
from torch import optim
from pretrain_util.retnet import RetNet
from data_V4.DateLoader import GraphDataset, AdjacencyMatrix
from torch.utils.data import DataLoader
from util.base import mprint


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='优化器')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')

    parser.add_argument('--retnet-params-file', type=str, default="./data/weight/retnet_params.pth", help='权重文件位置')
    parser.add_argument('--params-file', type=str, default="./data/weight/model_params.pth", help='权重文件位置')
    parser.add_argument('--data-file', type=str, default="./data/group/IBRL_data_anomaly.npz", help='数据集文件位置')
    parser.add_argument('--windows', type=int, default=50, help='窗口大小/序列长度')
    parser.add_argument('--step', type=int, default=5, help='间隔步长')
    parser.add_argument('--batch-size', type=int, default=5, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')


    parser.add_argument('--hidden-sizes', type=int, default=3, help='模态数')
    parser.add_argument('--label-sizes', type=int, default=2, help='标签长度')
    parser.add_argument('--node-sizes', type=int, default=51, help='节点数量')
    parser.add_argument('--r-layers', type=int, default=3, help='MultiScaleRetention的层数')
    parser.add_argument('--r-heads', type=int, default=3, help='MultiScaleRetention的头数')
    parser.add_argument('--r-ffn-size', type=int, default=3, help='RetNetBlock前向神经网络的隐藏层维度')
    parser.add_argument('--r-blocks', type=int, default=3, help='RetNetBlock的数量')
    parser.add_argument('--gat-heads', type=int, default=3, help='GAT的头数')

    parser.add_argument('--b-layers', type=int, default=4, help='双图判别网络的层数')

    parser.add_argument('--double-v-dim', action='store_true', help='Retnet网络前向计算过程中，V的维度是否加倍')
    parser.add_argument('--retnet-output-dim', type=int, default=25, help='Retnet网络前向运算中，节点经过网络输出的维度')
    parser.add_argument('--loss-w1', type=int, default=0.7, help='损失函数权重')
    parser.add_argument('--loss-w2', type=int, default=0.1, help='损失函数权重')
    parser.add_argument('--loss-w3', type=int, default=0.2, help='损失函数权重')
    parser.add_argument('--loss3-sample-size', type=tuple, default=(3,10), help='判别器损失函数中的对比损失的正常异常样本采样数')
    return parser.parse_args()

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
    # print("======== INFONCE Input ==========")
    # print("q: ",q)
    # # print(q.shape)
    # # print(k.shape)
    # # print(n.shape)
    # print("=================================")
    positive_logits = torch.exp(torch.bmm(k, q.unsqueeze(-1)) / temperature)  # (batch_size, k_number)
    negative_logits = torch.exp(torch.bmm(n, q.unsqueeze(-1))/ temperature)  # (batch_size, n_number)
    p_logits = positive_logits.sum(dim=(1,2))
    n_logits = negative_logits.sum(dim=(1,2))

    return -torch.log(p_logits / (p_logits + n_logits)).sum()






def sample_nodes(X, mask):
    batch_size, num_nodes, feature_dim = X.shape
    num_a_nodes = 11
    num_b_nodes = 20
    A = torch.zeros((batch_size, num_a_nodes, feature_dim))
    B = torch.zeros((batch_size, num_b_nodes, feature_dim))
    for i in range(batch_size):
        # 获取当前图的掩码和特征
        mask_i = mask[i]
        X_i = X[i]
        a_indices = torch.where(mask_i == 0)[0]
        b_indices = torch.where(mask_i == 1)[0]
        if len(a_indices) < num_a_nodes:
            extra_a_indices = np.random.choice(a_indices, num_a_nodes - len(a_indices), replace=True)
            a_indices = np.concatenate([a_indices, extra_a_indices])
        else:
            a_indices = np.random.choice(a_indices, num_a_nodes, replace=False)

        if len(b_indices) < num_b_nodes:
            extra_b_indices = np.random.choice(b_indices, num_b_nodes - len(b_indices), replace=True)
            b_indices = np.concatenate([b_indices, extra_b_indices])
        else:
            b_indices = np.random.choice(b_indices, num_b_nodes, replace=False)
        A[i] = X_i[a_indices]
        B[i] = X_i[b_indices]
    return A, B


def pretrain(opt:argparse.Namespace):
    '''
    @Author: WangSuxiao
    @description: 使用对比学习预训练特征提取网络
    @return {Any}
    '''
    # 加载数据集
    ibrl = np.load(opt.data_file)
    data_tmp = torch.from_numpy(ibrl["data_ab"])
    label_tmp = torch.from_numpy(ibrl["data_la"])
    dataset = GraphDataset(data=data_tmp, label=label_tmp, W=opt.windows, step=opt.step)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,drop_last=True)   # 训练数据
    adjacencyMatrixHelper = AdjacencyMatrix()
    A = adjacencyMatrixHelper.get_A_COO(opt.batch_size,node_size = 51)          # 邻接矩阵
    mprint(3, f"A: {A.shape}", prefix="Train Dataset")
    # 预训练模型
    retnet = RetNet(hidden_size=opt.hidden_sizes,sequence_len=opt.windows, double_v_dim=opt.double_v_dim)

    loss = infoNCE
    # Adam优化器
    optimizer = optim.Adam(retnet.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lt = []
    for epoch in range(800):
        ltt = []
        for data in dataloader:
            X, labels = data
            L_onehot = torch.nn.functional.one_hot(labels, num_classes=4).double()
            L_sample = (labels != 0).double()
            optimizer.zero_grad()
            Y = retnet(X,A)
            NX, ABX = sample_nodes(Y, L_sample)
            l = loss(NX[:,0,:], NX[:,0:,:], ABX)
            # print("\tLoss: ",l)
            l.backward()
            ltt.append(l.item())
            torch.nn.utils.clip_grad_norm_(retnet.parameters(), 1)
            optimizer.step()
        print("Loss Mean: ",np.mean(ltt))
        torch.save(retnet.state_dict(), f"weights/{epoch}.pth")


def set_nonzero_elements_to_one(X):
        return (X != 0).float()

if __name__ == "__main__":

    opt = parse_opt()
    pretrain(opt)

