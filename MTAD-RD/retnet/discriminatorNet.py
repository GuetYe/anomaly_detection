# !/usr/bin/env python
# -*- coding: utf-8 -*-

# *****************************************************

# Author       : WangSuxiao
# Date         : 2024-03-12 15:03:58
# LastEditTime : 2024-03-17 15:12:30
# Description  : 双图判别网络
# Tips         :

# *****************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from util.base import mprint


import torch
import torch.nn as nn

# class MLP(nn.Module):
#     """ 一个简单的MLP 供Dis2Ins、Ins2Dis以及node2edge调用 """
#     def __init__(self, input_dim, output_dim = None, hidden_dim = None):
#         super(MLP, self).__init__()

#         self.output_dim = output_dim if output_dim else input_dim * 2
#         self.hidden_dim = hidden_dim if hidden_dim else input_dim * 2
#         self.net = nn.Sequential(
#             nn.Linear(input_dim,self.hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.net(x)
import torch.nn.init as init

class MLP(nn.Module):
    """ 一个简单的MLP 供Dis2Ins、Ins2Dis以及node2edge调用 """
    def __init__(self, input_dim, output_dim=None, hidden_dim=None):
        super(MLP, self).__init__()

        self.output_dim = output_dim if output_dim else input_dim * 2
        self.hidden_dim = hidden_dim if hidden_dim else input_dim * 2
        self.linear1 = nn.Linear(input_dim, self.hidden_dim, dtype=torch.float64)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim, dtype=torch.float64)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # 对线性层的权重进行Xavier初始化
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        mprint(4,f"x.dtype: {x.shape}", prefix="MLP forward")
        mprint(4,f"self.linear1(x).dtype: {self.linear1(x).shape}", prefix="MLP forward")
        x = self.activation(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x


class Dis2Ins(nn.Module):
    """
    借助分布图更新实例图的节点
    节点i的更新：   分布图边(i, j) * 上层节点j || 上层节点i
    """
    def __init__(self, ins_node_feature):
        super(Dis2Ins, self).__init__()
        self.mlp = MLP(ins_node_feature*2, ins_node_feature,ins_node_feature + ins_node_feature//2)

    def forward(self, N_ins, E_dis):
        '''
        @Author: WangSuxiao
        @description: 改为batch训练方式
        @param {Any} self :
        @param {Any} N_ins :  (batch, NK, m)
        @param {Any} E_dis :  (batch, NK, NK)  // (NK*(NK-1), 1)
        @return {Any}
        '''
        batch_size, NK = E_dis.shape[0], E_dis.shape[1]
        # for i in range(NK):
        #     # 不计算 e_ii * vi
        #     E_dis[i, i] = 0
        E_dis_ = E_dis.clone()
        tmp = torch.diagonal(E_dis_, dim1=1, dim2=2)
        tmp.fill_(0)
        X = torch.cat((N_ins, E_dis_ @ N_ins),dim=-1)
        return self.mlp(X)

    def forward_s(self, N_ins, E_dis):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} N_ins :  (NK, m)
        @param {Any} E_dis :  (NK, NK)  // (NK*(NK-1), 1)
        @return {Any}
        '''
        NK = E_dis.shape[0]
        E_dis_ = E_dis.clone()
        for i in range(NK):
            # 不计算 e_ii * vi
            E_dis_[i, i] = 0
        #  N_ins : (NK, m)
        #       v_i, $i \in [1,NK]$
        #  E_dis @ N_ins:
        #       (NK, NK): (节点i, 与j的边的权重)
        X = torch.cat((N_ins, E_dis_ @ N_ins),dim=-1)
        return self.mlp(X)
# ============  用以测试 Dis2Ins 网络  =================
# node_ins = torch.arange(1,11,dtype=torch.float).reshape(2,5)
# edge_dis = torch.arange(1,5,dtype=torch.float).reshape(2, 2)
# print(node_ins.dtype, node_ins)
# print(edge_dis.dtype, edge_dis)
# dis2ins = Dis2Ins(5)
# y = dis2ins(node_ins, edge_dis)
# print(y.dtype, y)
# =====================================================

class Ins2Dis(nn.Module):
    """
    借助实例图更新分布图的节点
    节点i的更新：   分布图边(i, j) * 上层节点j || 上层节点i
    """
    def __init__(self, node_number):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} node_number : 实例图的节点数量 (小样本学习中的 `N * K`)
        @return {Any}
        '''
        super(Ins2Dis, self).__init__()
        self.mlp = MLP(node_number*2, node_number,node_number + node_number//2)

    def forward(self, N_dis, E_ins):
        '''
        @Author: WangSuxiao
        @description: E_ins的维度已经将i节点到其他节点的变得权重拼接了，不需要进一步处理
        @param {Any} self :
        @param {Any} N_dis :  (NK, NK)      v_i的特征维度为NK
        @param {Any} E_ins :  (NK, NK)      dim_1的特征为节点i到其他节点的边的权重
        @return {Any}
        '''
        # print(N_dis)
        # print(E_ins)
        # print(torch.cat((N_dis,E_ins),dim=-1))
        return self.mlp(torch.cat((N_dis,E_ins),dim=-1))
# ============  用以测试 Ins2Dis 网络  =================
# node_dis = torch.arange(1,26,dtype=torch.float).reshape(5,5)
# edge_ins = torch.arange(1,26,dtype=torch.float).reshape(5, 5)
# print(node_dis.dtype, node_dis)
# print(edge_ins.dtype, edge_ins)
# ins2Dis = Ins2Dis(5)
# y = ins2Dis(node_dis, edge_ins)
# print(y.dtype, y)
# =====================================================



class DiscriminatorNet(nn.Module):
    """
    双图判别网络
        实例图： 节点维度：ins_node_size;   边维度：1
        关系图： 节点维度：n*k;             边维度：1
        N_ins   (NK, m)
        E_ins   (NK*(NK-1), 1)  有向图
        N_dis   (NK, NK)
        E_dis   (NK*(NK-1), 1)  有向图
    """
    # def __init__(self,layer, ins_node_feature, way, shot):
    def __init__(self, layers, ins_node_feature,ins_node_size):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} layers : 判别网络的层数
        @param {Any} ins_node_feature : 实例图的节点特征长度
        @param {Any} way : 类别数
        @param {Any} shot : 单类样本数
        @return {Any}
        '''

        super(DiscriminatorNet, self).__init__()
        self.layer = layers
        self.ins_node_feature = ins_node_feature
        # self.ins_node_size = way * shot
        self.ins_node_size =ins_node_size

        self.insEdgeEncoder = [
            MLP(input_dim = self.ins_node_feature, output_dim = 1, hidden_dim = self.ins_node_feature//2)
            for _ in range(self.layer)
        ]

        self.disEdgeEncoder = [
            MLP(input_dim = self.ins_node_size, output_dim = 1, hidden_dim = self.ins_node_size//2)
            for _ in range(self.layer)
        ]

        # 用来初始化边
        self.initInsEdge = MLP(input_dim = self.ins_node_feature, output_dim = 1, hidden_dim = self.ins_node_feature//2)
        self.initDisEdge = MLP(input_dim = self.ins_node_size, output_dim = 1, hidden_dim = self.ins_node_size//2)

        self.dis2Ins = [
            Dis2Ins(self.ins_node_feature)
            for _ in range(self.layer)
        ]

        self.ins2Dis = [
            Ins2Dis(self.ins_node_size)
            for _ in range(self.layer)
        ]
        # 可学习参数的初始化

    def forward_batch(self, X, L:torch.Tensor):
        '''
        @Author: WangSuxiao
        @description: 前向计算,加入了批量训练的版本
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, WSN_feature)
        @param {Any} L : (batch_size, node_size, label)
        @return {Any} :
            G_ins: 双图结构最终预测概率，优化双图网络最终结果
            G_dis: 分布图各层预测结果，优化各层训练中的偏差
        '''
        bathc_size, node_size = X.shape[0], X.shape[1]
        # 初始化实例图节点
        Node_ins_0 = torch.cat((X, L), dim=-1)
        # 初始化实例图边
        mprint(4, f"Node_ins_0.shape: {Node_ins_0.shape}", prefix="discriminatorNet forward_batch")
        Edge_ins_0 = self.initInsEdge(torch.square(Node_ins_0.unsqueeze(2) - Node_ins_0.unsqueeze(1))).squeeze(-1)
        # print(Edge_ins_0)
        # 初始化分布图节点
        # 有标签的计算
        # tmp = L.unsqueeze(-1) == L.unsqueeze(-2)
        tmp = L.unsqueeze(2) == L.unsqueeze(1)
        Node_ins_0_labeled = torch.all(tmp,dim=-1)
        flag_labeled = torch.any(L, dim=-1).unsqueeze(-1)
        # print(Node_ins_0_labeled)
        # print(flag_labeled.float())
        # print(Node_ins_0_labeled * flag_labeled)
        # 无标签的计算
        Node_ins_0_unlabeled = torch.full((bathc_size, node_size, node_size), 1/node_size, dtype=torch.float64)
        flag_unlabeled= (flag_labeled ^ 1)
        # print(Node_ins_0_unlabeled)
        # print(flag_unlabeled)
        # print(Node_ins_0_unlabeled * flag_unlabeled)
        # 融合得到Node_ins
        Node_dis_0 = Node_ins_0_labeled * flag_labeled + Node_ins_0_unlabeled * flag_unlabeled
        # 初始化分布图边


        Edge_dis_0 = self.initDisEdge(torch.square(Node_dis_0.unsqueeze(2) - Node_dis_0.unsqueeze(1))).squeeze(-1)
        # 更新: Ei Nd Ed Ni -> Ei Nd Ed Ni -> ....
        Ei = Edge_ins_0
        Nd = Node_dis_0
        Ed = Edge_dis_0
        Ni = Node_ins_0
        Ydis = []
        for i in range(self.layer):
            Ei = self.insEdgeEncoder[i](torch.square(Ni.unsqueeze(2) - Ni.unsqueeze(1))).squeeze(-1) * Ei
            Nd = self.ins2Dis[i](Nd, Ei)
            Ed = self.disEdgeEncoder[i](torch.square(Nd.unsqueeze(2) - Nd.unsqueeze(1))).squeeze(-1)
            Ni = self.dis2Ins[i](Ni,Ed)
            mprint(4, f"Ed.dtype: {Ed.dtype}", prefix="DiscriminatorNet forward_batch")
            mprint(4, f"L.dtype: {L.dtype}", prefix="DiscriminatorNet forward_batch")
            Ydis.append(F.softmax(Ed@L, dim=-1))
        # print("========forward_batch=======")
        # print(Ni.shape)
        # print(Ei.shape)
        # print(L.shape)
        # print("============================")
        # 预测 ： 得到各类的得分
        Yins = F.softmax(Ei@L, dim=-1)
        return Yins, Ydis, Ni

    def forward(self, X, L:torch.Tensor):
        '''
        @Author: WangSuxiao
        @description: 前向计算
        @param {Any} self :
        @param {Any} X : (node_size, WSN_feature)
                    当前版本无批量训练，特征提取网络的一个batch作为node_size
        @param {Any} L : (node_size, label)
        @return {Any} :
            G_ins: 双图结构最终预测概率，优化双图网络最终结果
            G_dis: 分布图各层预测结果，优化各层训练中的偏差
        '''
        node_size = X.shape[0]
        # 初始化实例图节点
        Node_ins_0 = torch.cat((X, L), dim=-1)
        # 初始化实例图边
        Edge_ins_0 = self.initInsEdge(torch.square(Node_ins_0.unsqueeze(1) - Node_ins_0.unsqueeze(0))).squeeze(-1)
        # 初始化分布图节点
        # 有标签的计算
        tmp = L.unsqueeze(1) == L.unsqueeze(0)
        Node_ins_0_labeled = torch.all(tmp,dim=-1).float()
        flag_labeled = torch.any(L, dim=-1).unsqueeze(1)
        # 无标签的计算
        Node_ins_0_unlabeled = torch.full((node_size, node_size), 1/node_size)
        flag_unlabeled= (flag_labeled ^ 1)
        # 融合得到Node_ins
        Node_dis_0 = Node_ins_0_labeled * flag_labeled + Node_ins_0_unlabeled * flag_unlabeled
        # 初始化分布图边
        Edge_dis_0 = self.initDisEdge(torch.square(Node_dis_0.unsqueeze(1) - Node_dis_0.unsqueeze(0))).squeeze(-1)
        # 更新: Ei Nd Ed Ni -> Ei Nd Ed Ni -> ....
        Ei = Edge_ins_0
        Nd = Node_dis_0
        Ed = Edge_dis_0
        Ni = Node_ins_0
        Ydis = []
        for i in range(self.layer):
            Ei = self.insEdgeEncoder[i](torch.square(Ni.unsqueeze(1) - Ni.unsqueeze(0))).squeeze(-1) * Ei
            Nd = self.ins2Dis[i](Nd, Ei)
            Ed = self.disEdgeEncoder[i](torch.square(Nd.unsqueeze(1) - Nd.unsqueeze(0))).squeeze(-1)
            Ni = self.dis2Ins[i].forward_s(Ni,Ed)
            Ydis.append(F.softmax(Ed@L,dim=-1))

        # 预测 ： 得到各类的得分
        Yins = F.softmax(Ei@L,dim=-1)
        return Yins, Ydis

if __name__ =="__main__":

    # 判别网络的测试
    X = torch.tensor([[1, 2, 4],
                    [1, 2, 4],
                    [4, 2, 3],
                    [4, 2, 3]],dtype=torch.float32)
    L = torch.tensor([[0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0]],dtype=torch.float32)

    # # 使用广播机制代替for循环，判断各节点标签是否相同
    # xs = X.unsqueeze(0)
    # xt = X.unsqueeze(1)
    # print(xs.shape , xs)
    # print(xt.shape , xt)
    # comparison = X.unsqueeze(1) == X.unsqueeze(0)
    # print(comparison)
    # print(torch.all(comparison,dim=-1))
    # print(torch.all(comparison,dim=-1).float())

    # # 使用广播机制代替for循环，计算初始化边时的输入
    # print(X.unsqueeze(1) - X.unsqueeze(0))
    # initInsEdge = MLP(input_dim = 3, output_dim = 1, hidden_dim = 3//2)
    # print(initInsEdge(X.unsqueeze(1).float() - X.unsqueeze(0).float()))

    # # 前向计算的验证
    discriminatorNet = DiscriminatorNet(4,3 + 4,2*2)
    Yins, Ydis = discriminatorNet(X,L)
    X = X.unsqueeze(0).repeat(2, 1, 1)
    L = L.unsqueeze(0).repeat(2,1,1)
    print("==========================")
    print(X.shape, X.dtype)
    print(L.shape, L.dtype)
    Yins_b, Ydis_b = discriminatorNet.forward_batch(X,L)

    # print(Yins.shape)
    # print(Yins)
    # print(Yins_b.shape)
    # print(Yins_b)

    # print(type(Ydis))
    # print(len(Ydis))
    # print(Ydis[0])
    # print(type(Ydis_b))
    # print(len(Ydis_b))
    # print(Ydis_b[0])
    # print("size =",len(Ydis),Ydis_b[0])

