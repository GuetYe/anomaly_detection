# !/usr/bin/env python
# -*- coding: utf-8 -*-

# *****************************************************

# Author       : WangSuxiao
# Date         : 2024-02-24 15:11:32
# LastEditTime : 2024-03-09 14:00:33
# Description  : RetNet网络的搭建
# Tips         :

# *****************************************************

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from retnet.retention import MultiScaleRetention
from util.base import mprint

class RetNetBlock(nn.Module):
    """
    RetNet块的实现，一个块中包含多层RetNet Layer和前向网络
    """
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} layers : MultiScaleRetention的层数
        @param {Any} hidden_dim : token的维度
        @param {Any} ffn_size : 单层FFN中隐藏层大小
        @param {Any} heads : MultiScaleRetention的头数
        @param {Any} double_v_dim :
        @return {Any}
        '''
        super(RetNetBlock, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        mprint(4, f"layers: {layers}",prefix="RetNetBlock Config")

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size,dtype=torch.float64),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim, dtype=torch.float64)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim ,dtype=torch.float64)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim, dtype=torch.float64)
            for _ in range(layers)
        ])

    def forward(self, X) ->torch.Tensor:
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any}
        '''
        for i in range(self.layers):
            # 含有残差结构一层编码器
            mprint(4, X.shape, prefix="RetNetBlock")
            mprint(4, self.layer_norms_1[i](X).shape, prefix="RetNetBlock")
            mprint(4, self.retentions[i](self.layer_norms_1[i](X)).shape, prefix="RetNetBlock")
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            Y = F.relu(Y)
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
            X = F.relu(X)
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} x_n :
        @param {Any} s_n_1s :
        @param {Any} n :
        @return {Any}
        '''

        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is


class RetNet(nn.Module):
    def __init__(self, hidden_size, sequence_len, double_v_dim=False,
                    layer_size = 5, layer_heads = 1,ffn_size = 6, blocks = 5,
                    gat_head = 3):
        '''
        @Author: WangSuxiao
        @description: 特征提取网络的实现
        @param {Any} self :
        @param {Any} hidden_size : 词向量维度
        @param {Any} sequence_len : 序列长度
        @param {Any} double_v_dim : RetionLayer中是否使用double_v_dim
        @param {Any} layer_size : MultiScaleRetention的层数
        @param {Any} layer_heads : MultiScaleRetention的头数
        @param {Any} ffn_size : RetNetBlock前向神经网络的隐藏层维度
        @param {Any} blocks : RetNetBlock的数量
        @param {Any} gat_head : GAT网络的头数
        @return {Any}
        '''

        super(RetNet, self).__init__()
        self.input_dim = hidden_size
        # ================引入配置文件===============
        # ==========================================
        block_ffns_input = sequence_len * hidden_size       # RetNetBlock转图特征向量时的隐藏层
        block_ffns_hidden = block_ffns_input // 4           # RetNetBlock转图特征向量时的隐藏层
        block_ffns_output = sequence_len   // 2                 # RetNetBlock转图特征向量时的隐藏层
        self.gat_head = gat_head                # GAT注意力头数
        graph_gat_hidden = sequence_len   // 4                 # RetNetBlock转图特征向量时的隐藏层
        # ==========================================

        mprint(4, f"hidden_size: {hidden_size}",prefix="RetNet Config")
        mprint(4, f"sequence_len: {sequence_len}",prefix="RetNet Config")
        mprint(4, f"blocks: {blocks}",prefix="RetNet Config")
        # mprint(4, f"hidden_size: {hidden_size}",prefix="RetNet Config")
        # mprint(4, f"hidden_size: {hidden_size}",prefix="RetNet Config")

        self.block_size = blocks
        # RetNet网络
        # 提取时序间&模态间相关性
        self.retentblock = nn.ModuleList([
            RetNetBlock(layer_size, hidden_size, ffn_size, layer_heads, double_v_dim=double_v_dim)
            for _ in range(self.block_size)
        ])

        # block to facture
        # 获得各block的节点级特征表示
        # (batch,node,sequence,hidden) ==> Liner(batch,node,sequence*hidden)
        self.block_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(block_ffns_input, block_ffns_hidden, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(block_ffns_hidden, block_ffns_output, dtype=torch.float64)
            )
            for _ in range(self.block_size)
        ])

        # 汇集各block的节点级特征表示
        self.mlp = nn.Linear(self.block_size, 1, dtype=torch.float64)

        # GAT获得图集特征表示
        self.gatconv1 = gnn.GATConv(block_ffns_output, graph_gat_hidden, heads=self.gat_head,
                            bias=True).double()
        self.gatconv2 = gnn.GATConv(graph_gat_hidden * self.gat_head, block_ffns_output,
                            bias=True).double()


    def forward(self, X:torch.Tensor, A = None):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any}
        '''
        batch_size,node_size = X.shape[0], X.shape[1]

        block_res = []
        for i in range(self.block_size):
            # 含有残差结构一层编码器
            mprint(3, f"X.shape: {X.shape}", prefix="Retnet forward")
            # mprint(1, f"X.dtype: {X.dtype}", prefix="Retnet forward")
            X = self.retentblock[i](X)  # torch.Size([2, 51, 100, 3]) (batch_size, node_size, sequence_length, hidden_size)
            # print(X.shape)
            # 获得各block的节点级特征表示
            block_res.append(self.block_ffns[i](X.view(batch_size,node_size,-1)))
        # print(len(block_res),block_res[0].shape)  # block_size    (batch_size, node_size, feature)
        X = torch.stack(block_res,dim=-1)           # (batch_size, node_size, feature, block_size)
        # print(X.shape)
        X = self.mlp(X).squeeze(-1)     # (batch_size, node_size, feature)
        X = F.relu(X)
        # print(X.shape)
        Xshape = X.shape
        # X = self.gatconv1(X.view(Xshape[0]*Xshape[1],Xshape[2]), A)
        # X = self.gatconv2(X, A)
        X = self.gatconv1(X.reshape(batch_size*node_size, -1),A)
        X = F.relu(X)
        X = self.gatconv2(X,A).reshape(batch_size,node_size,-1)
        X = F.relu(X)
        return X.view(*Xshape)
        # return X.view(3, 51, 25)


class RetNetPositive(nn.Module):
    """ 以不同的网络结构作为对比学习中的正样本 """
    def __init__(self, hidden_size, sequence_len, double_v_dim=False, net = 3,
                    layer_size = 5, layer_heads = 1,ffn_size = 6, blocks = 5,
                    gat_head = 3):
        '''
        @Author: WangSuxiao
        @description: 特征提取网络的实现
        @param {Any} self :
        @param {Any} hidden_size : token的维度
        @param {Any} sequence_len : 序列长度
        @param {Any} double_v_dim : RetionLayer中是否使用double_v_dim
        @param {Any} layer_size : MultiScaleRetention的层数
        @param {Any} layer_heads : MultiScaleRetention的头数
        @param {Any} ffn_size : RetNetBlock前向神经网络的隐藏层维度
        @param {Any} blocks : RetNetBlock的数量
        @param {Any} gat_head : GAT网络的头数
        @return {Any}
        '''
        super(RetNetPositive, self).__init__()
        self.net = net
        # ================引入配置文件===============
        # ==========================================
        block_ffns_input = sequence_len * hidden_size       # RetNetBlock转图特征向量时的隐藏层
        block_ffns_hidden = block_ffns_input // 4           # RetNetBlock转图特征向量时的隐藏层
        block_ffns_output = sequence_len   // 2                 # RetNetBlock转图特征向量时的隐藏层
        self.gat_head = gat_head                # GAT注意力头数
        graph_gat_hidden = sequence_len   // 4                 # RetNetBlock转图特征向量时的隐藏层
        # ==========================================

        mprint(1, f"hidden_size: {hidden_size}",prefix="RetNetPositive Config")
        mprint(1, f"sequence_len: {sequence_len}",prefix="RetNetPositive Config")
        mprint(1, f"blocks: {blocks}",prefix="RRetNetPositive Config")
        self.retent = nn.ModuleList([
            RetNet(hidden_size, sequence_len, double_v_dim, layer_size, layer_heads, ffn_size, blocks, gat_head)
            for _ in range(net)
        ])

        # 汇集各block的节点级特征表示
        self.mlp = nn.Linear(sequence_len//2 * net, sequence_len//2, dtype=torch.float64)

        # GAT获得图集特征表示
        self.gatconv1 = gnn.GATConv(block_ffns_output, graph_gat_hidden, heads=self.gat_head,
                            bias=True).double()
        self.gatconv2 = gnn.GATConv(graph_gat_hidden * self.gat_head, block_ffns_output,
                            bias=True).double()


    def forward(self, X:torch.Tensor, A = None):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any}
        '''
        batch_size,node_size = X.shape[0], X.shape[1]
        retent_res = []
        for i in range(self.net):
            mprint(3, f"X: {X.view(batch_size,node_size,-1).shape}",prefix="RetNetPositive Config")
            mprint(3, f"self.retent[i](X): {self.retent[i](X).shape}",prefix="RetNetPositive Config")
            retent_res.append(self.retent[i](X))
        # print(len(block_res),block_res[0].shape)  # block_size    (batch_size, node_size, feature)
        X = torch.cat(retent_res, dim=-1)           # (batch_size, node_size, feature, block_size)
        # print(X.shape)
        X = self.mlp(X).squeeze(-1)     # (batch_size, node_size, feature)
        print("X.shape ====>> ")
        X = self.gatconv1(X.reshape(batch_size*node_size, -1),A)
        X = self.gatconv2(X,A).reshape(batch_size,node_size,-1)
        return X




