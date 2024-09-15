import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import argparse
from torch import nn, optim
from retnet.retention import SimpleRetention, MultiScaleRetention
from retnet.retnet import RetNetPositive,RetNet
from retnet.discriminatorNet import DiscriminatorNet
from retnet.module import MTAD_RD
from data.self.ibrlDateLoader import GraphDataset, AdjacencyMatrix
from torch.utils.data import DataLoader
from tmp.utilfunc import compute_f1_score
from retnet.loss import FewShotLoss, infoNCE
from util.base import mprint
from util.utilfunc import SingleSample, PNSample

# DATA_FILE = './data/IBRL_data_anomaly.npz'
# WINDOW = 200    # 时序窗口大小
# STEP = 5        # 间隔
# BATCH_SIZE = 16



def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='优化器')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')

    parser.add_argument('--retnet-params-file', type=str, default="./data/weight/retnet_params.pth", help='权重文件位置')
    parser.add_argument('--params-file', type=str, default="./data/weight/model_params.pth", help='权重文件位置')
    parser.add_argument('--data-file', type=str, default="./data/group/IBRL_data_anomaly.npz", help='数据集文件位置')
    parser.add_argument('--windows', type=int, default=50, help='窗口大小/序列长度')
    parser.add_argument('--step', type=int, default=5, help='间隔步长')
    parser.add_argument('--batch-size', type=int, default=3, help='批量大小')
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
    subgraphMask = adjacencyMatrixHelper.get_subgraphMask()                     # 子图掩码
    mprint(3, f"A: {subgraphMask}", prefix="Train Dataset")
    mprint(3, f"A: {A.shape}", prefix="Train Dataset")
    # 预训练模型
    retnet = RetNet(hidden_size=opt.hidden_sizes,sequence_len=opt.windows, double_v_dim=opt.double_v_dim)
    # 正例对模型
    retnetPositive = RetNetPositive(hidden_size=opt.hidden_sizes,sequence_len=opt.windows, double_v_dim=opt.double_v_dim)
    label_onehot = torch.nn.functional.one_hot(torch.arange(0, opt.label_sizes), num_classes=opt.label_sizes)
    # 对比学习的损失函数
    loss = infoNCE
    # Adam优化器
    optimizer = optim.Adam(retnet.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lt = []
    for epoch in range(800):
        ltt = []
        for i, data in enumerate(dataloader, 0):
            X, labels = data
            labels = torch.nn.functional.one_hot(labels, num_classes=2).double()
            optimizer.zero_grad()
            Y_retnet = retnet(X,A)
            a, p, n = PNSample(Y_retnet, subgraphMask)
            # Y_retnetPositive = retnetPositive(X,A)
            # _, k, _ = SingleSample(Y_retnet,Y_retnetPositive, 10)
            # l = loss(a, k, n)
            l = loss(a, p, n)
            l.backward()
            ltt = [].append(l.item())
            optimizer.step()
        lt.append(np.mean(ltt))
    ndarr = np.array(lt)
    np.save('pretrain.npy', ndarr)
    torch.save(retnet.state_dict(), opt.retnet_params_file)


def train(opt:argparse.Namespace):
    '''
    @Author: WangSuxiao
    @description: 前向训练
    @return {Any}
    '''

    # 加载数据集
    ibrl = np.load(opt.data_file)
    data_tmp = torch.from_numpy(ibrl["data_ab"])
    label_tmp = torch.from_numpy(ibrl["data_la"])
    dataset = GraphDataset(data=data_tmp, label=label_tmp, W=opt.windows, step=opt.step)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    adjacencyMatrixHelper = AdjacencyMatrix()
    A = adjacencyMatrixHelper.get_A_COO(opt.batch_size,node_size = 51)
    mprint(1, f"A: {A.shape}", prefix="Train Dataset")
    # 创建网络模型
    # retnet = RetNet(hidden_size=opt.hidden_sizes,sequence_len=opt.windows, double_v_dim=opt.double_v_dim)
    # discriminatorNet = DiscriminatorNet(layers=opt.b_layers, ins_node_feature=opt.retnet_output_dim + 2, ins_node_size=opt.node_sizes)
    net = MTAD_RD(opt=opt)  # 整个的网络
    torch.save(net.feature_extraction().state_dict(), opt.retnet_params_file)
    net.feature_extraction().load_state_dict(torch.load(opt.retnet_params_file))

    label_onehot = torch.nn.functional.one_hot(torch.arange(0, opt.label_sizes), num_classes=opt.label_sizes)
    loss = FewShotLoss(opt.loss_w1, opt.loss_w2, opt.loss_w3, label_onehot, temperature = 1)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lt =[]
    for epoch in range(opt.epochs):
        ltt = []
        mprint(2, f"epoch : {epoch}", prefix="Train Dataset")
        for i, data in enumerate(dataloader, 0):
            X, labels = data
            labels = torch.nn.functional.one_hot(labels, num_classes=2).double()
            optimizer.zero_grad()
            Y_ins, Y_dis, Ni= net.forward(X, A, labels)
            l = loss(Y_ins, Y_dis, Ni, labels)
            l.backward()
            optimizer.step()
            ltt.append(l.item())
        lt.append(np.mean(ltt))
    ndarr = np.array(lt)
    np.save('train.npy', ndarr)
    torch.save(net.state_dict(), opt.params_file)

def inference():
    '''
    @Author: WangSuxiao
    @description: 在线推理
    @return {Any}
    '''
    ibrl = np.load(opt.data_file)
    data_tmp = torch.from_numpy(ibrl["data_ab"])
    label_tmp = torch.from_numpy(ibrl["data_la"])
    dataset = GraphDataset(data=data_tmp, label=label_tmp, W=opt.windows, step=opt.step, train=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    adjacencyMatrixHelper = AdjacencyMatrix()
    A = adjacencyMatrixHelper.get_A_COO(opt.batch_size,node_size = 51)
    net = MTAD_RD(opt=opt)  # 整个的网络
    net.feature_extraction().load_state_dict(torch.load(opt.retnet_params_file))
    tmp_f = []
    for i in range(20): # 测试的次数，加载数据集使用了随机起点
        tmp_g = []
        tmp_p = []
        for data in dataloader:
            X, labels = data
            labels = torch.nn.functional.one_hot(labels, num_classes=2).double()
            mprint(3, f"X.shape: {X.shape}", prefix="inference")
            Y_ins, _, _= net.forward(X, A, labels)
            predict_labels =torch.nn.functional.one_hot(torch.argmax(Y_ins, dim=-1) , num_classes=2)# Y_ins是实例图的最终预测
            tmp_g.append(labels)
            tmp_p.append(predict_labels)
        g = torch.cat(tmp_g, dim=0)
        p = torch.cat(tmp_p, dim=0)
        precision, recall, f1_score = compute_f1_score(g.reshape(-1,2), p.reshape(-1,2))
        tmp_f.append(f1_score)
        print(f"第{i}次测试： ",precision, recall, f1_score)
    print("F1均值： ",np.mean(tmp_f))


def test_retnet():
    '''
    @Author: WangSuxiao
    @description: 测试函数
    @return {Any}
    '''
    batch_size = 2
    node_size = 40
    sequence_length = 100
    hidden_size = 9
    heads = 3
    layers = 4
    ffn_size = 128

    X = torch.rand(batch_size, node_size, sequence_length, hidden_size)
    adjacencyMatrix = AdjacencyMatrix()
    A = adjacencyMatrix.get_A()
    retnet = RetNet(hidden_size, sequence_length, double_v_dim=True)
    print("Double v_dim:",sum(p.numel() for p in retnet.parameters() if p.requires_grad))

    Y_parallel = retnet(X)      # (batch_size,sequence_length,hidden_size)



if __name__ == "__main__":
    opt = parse_opt()
    inference()

