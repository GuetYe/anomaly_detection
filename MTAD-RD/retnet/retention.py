# !/usr/bin/env python
# -*- coding: utf-8 -*-

# *****************************************************

# Author       : WangSuxiao
# Date         : 2024-02-24 15:11:32
# LastEditTime : 2024-03-15 16:05:28
# Description  : RetNet Layer 的实现
# Tips         :

# *****************************************************

import math

import torch
import torch.nn as nn
import torch.nn.init as init

# from xpos_relative_position import XPOS
from .xpos_relative_position import XPOS
from util.base import mprint

class SimpleRetention(nn.Module):

    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        '''
        @Author: WangSuxiao
        @description: retention mechanism based on the paper
        @param {Any} self :
        @param {Any} hidden_size : 输入token的维度
        @param {Any} gamma :
        @param {Any} head_size : 输出token的维度
        @param {Any} double_v_dim : 输入token维度是否翻倍
        @return {Any}
        '''

        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size      # 输入维度 / 模型维度       # (6.)
        if head_size is None:               # 没有指定本头的`输出维度`，则与`输入维度`相同，即使用hidden_size；
            head_size = hidden_size
        self.head_size = head_size
        # Vi的维度
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma
        # 注册为WQ WK WV为模型的参数
        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size, dtype=torch.float64) / hidden_size)      # (6, 6)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size, dtype=torch.float64) / hidden_size)      # (6, 6)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim, dtype=torch.float64) / hidden_size)     # (6, 12)
        init.kaiming_uniform_(self.W_Q, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_K, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_V, a=math.sqrt(5))
        self.xpos = XPOS(head_size)

    def forward(self, X):
        '''
        @Author: WangSuxiao
        @description: 并行训练的前向计算过程
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any} V : (batch_size, sequence_length, v_dim)
                v_dim可能等于 2 * head_size
        '''
        # sequence_length = X.shape[1]        # 时序长度
        sequence_length = X.shape[2]        # 时序长度
        D = self._get_D(sequence_length).to(self.W_Q.device)
        # ======================
        mprint(4, f"self.W_Q.dtype: {self.W_Q.dtype}", prefix="SimpleRetention forward")
        mprint(4, f"X.dtype: {X.dtype}", prefix="SimpleRetention forward")
        Q = (X @ self.W_Q)
        K = (X @ self.W_K)
        mprint(4, f"Q.shape: {Q.shape}", prefix="SimpleRetention forward")
        s = Q.shape
        sp = s[0] * s[1],s[2],s[3]
        mprint(4, f"Q.reshape(sp): {Q.reshape(sp).shape}", prefix="SimpleRetention forward")
        Q = self.xpos(Q.reshape(sp)).reshape(s)
        K = self.xpos(K.reshape(sp), downscale=True).reshape(s)

        V = X @ self.W_V
        # ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        ret = (Q @ K.permute(0,1, 3, 2)) * D.unsqueeze(0).unsqueeze(0)
        # permute对1、2维转置；unsqueeze升维
        # @是矩阵乘法运算，文中可以忽略这个运算符号：(batch_size, m, n) @ (batch_size, n, p) = (batch_size, m, p)
        # *是按元素相乘，文章中使用表示⊙
        # Q (batch_size, sequence_length, hidden_size) @ K^T (batch_size, hidden_size, sequence_length)
        # 是各个时刻的特征向量之间计算相关系数 ： A
        # D是一个掩码矩阵，屏蔽了当前时刻和当前时刻之后的信息，代表了前后因果关系
        # A*D得到的还是相关系数矩阵

        # A (batch_size, sequence_length, sequence_length) V (batch_size, sequence_length, hidden_size)
        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        '''
        @Author: WangSuxiao
        @description: 循环推理的前向计算过程
        @param {Any} self :
        @param {Any} x_n : 第n个token                   (batch_size, 1, hidden_size)
        @param {Any} s_n_1 : 前n-1个token的累计状态      (batch_size, hidden_size, v_dim)  v_dim可能等于double_hidden_dim
        @param {Any} n : 位置/下标/时刻
        @return {Any}
        '''
        # (batch_size, 1, hidden_size)  (batch_size, hidden_size, head_size)
        Q = (x_n @ self.W_Q)

        # (batch_size, 1, hidden_size)  (batch_size, hidden_size, head_size)
        K = (x_n @ self.W_K)
        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V
        # (batch_size, 1, hidden_size)  (batch_size, hidden_size, double_hidden_size)

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V
        # 按照论文，将那个A矩阵简化为一个标量，多个层时，各个layer不同
        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)        # gamma * sn + K^T @ V
        # O = Q @ sn
        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} x_i : (batch_size, chunk_size, hidden_size)
        @param {Any} r_i_1 : (batch_size, hidden_size, v_dim)
        @param {Any} i :
        @return {Any}
        '''
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)
        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)
        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)
        V = x_i @ self.W_V
        # ==================================================================================
        # tmp1 = K
        # tmp2 = K.transpose(-1, -2)              # 转置
        # tmp1 = D[-1]                            # (4, 4) 取出最后一个 (4,)
        # tmp2 = D[-1].view(1, chunk_size, 1)     # 维度变换  (1, 4, 1)
        # tmp1 = self.gamma
        # tmp1 = chunk_size
        # tmp1 = self.gamma ** chunk_size         # 权重因子的chunk_size(sequence_size)倍
        # tmp2 = r_i_1
        # tmp2 = (self.gamma ** chunk_size) * r_i_1   # 对之前的状态A/S衰退
        # ==================================================================================
        # 计算状态矩阵/A矩阵/S矩阵 :
        # (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1)))
        # 计算当前块的状态S，块内的各个token的需要衰减[self.gamma^(chunk_size-1), self.gamma^(chunk_size-2), ..., self.gamma^0]
        # 计算之前块对与当前最终S的贡献，之前的各个token需要衰减self.gamma^chunk_size
        # 需要衰减的原因: 并行计算时，也进行了衰减`ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)`
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        # 计算输出特征
        # 块间计算，使用之前各个块累计的A/S进行注意力计算
        # 就token_0而言，之前的A/S需要衰减self.gamma^1；token_0而言，之前的A/S需要衰减self.gamma^2
        # 这里为什么引入衰减？
        # 因为token_0距离r_i_1相差一个时刻，这里与前面计算r_i时的衰减类似，只是这里利用的是相对输入块而言未衰减的累计状态r_i_1计算输出张量，
        e = torch.zeros(batch, chunk_size, 1)
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        cross_chunk = (Q @ r_i_1) * e

        # 块内的并行计算。块内的各个token的需要衰减[self.gamma^(chunk_size-1), self.gamma^(chunk_size-2), ..., self.gamma^0]
        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        '''
        @Author: WangSuxiao
        @description: 返回一个`sequence_length * sequence_length`的掩码矩阵
        @param {Any} self :
        @param {Any} sequence_length : token的长度
        @return {Any}
            a_{ij} = self.gamma ** (i - j) if i>=j else 0
            self.gamma < 0 , 如果i-j较大时，可能导致a_{ij}值过小
            通过 : D[D != D] = 0 , 将极小值设置为0
            Example :
            self.gamma = 0.3; sequence_length = 3
            [
                [1,     0,      0,      0],
                [0.3,   1,      0,      0],
                [0.09,  0.3,    1,      0],
                [0.027, 0.09,  0.3,    1],
            ]
        '''
        n = torch.arange(sequence_length).unsqueeze(1)      # (12,1)
        m = torch.arange(sequence_length).unsqueeze(0)      # (1,12)

        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        D[D != D] = 0
        # D != D: 这部分创建了一个与 D 大小相同的布尔型张量，其中元素的值为 True 或 False，表示对应位置上的元素是否是 NaN。
        # 在大多数情况下，一个值不等于自己通常是 NaN 的标志。
        return D



class MultiScaleRetention(nn.Module):
    """ 多头注意力的实现方式 """

    def __init__(self, hidden_size, heads, double_v_dim=False):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} hidden_size : token的输入维度
        @param {Any} heads : 头数
        @param {Any} double_v_dim : token是否翻倍输出
        @return {Any}
        '''
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size          # 模型的维度 / 各个layer的输入维度 / 各个layer的输出维度
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads                      # 头数
        #多头retent 和单头retent整体上相同。即：头数 * 一头的输出 = 单头的输出
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads   # 各个头的维度
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        # ======================================================================================
        # 各层的gamma值（论文里的那个、`\gamma` r），是个衰退值，不是模型的可学习参数，需要剥离计算图
        # 按照论文，将那个A矩阵简化为一个标量，多个层时，各个layer不同

        # 指数和对数是逆运算
        # 对定义域为[1/512, 1/32]的log函数的值域均匀划分，
        # 即为对值域为[math.log(1/32), math.log(1/512)]的e^x函数的定义域均匀划分
        # 则，torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))的效果为：
        # 对值域的均匀划分，定义域划分间距越来越小
        # ======================================================================================
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads, dtype=torch.float64))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)     # Swish激活函数
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim, dtype=torch.float64) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size, dtype=torch.float64) / hidden_size)     # V_out的维度加倍后，这里也会还原回去
        init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))
        self.group_norm = nn.GroupNorm(heads, self.v_dim,dtype=torch.float64)
        # 分组归一化，组数为heads组，通道数为self.v_dim
        # 最终效果为：各个head分别归一化

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X):
        '''
        @Author: WangSuxiao
        @description: 并行训练的前向计算过程
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any} V : (batch_size, sequence_length, v_dim)  v_dim可能是翻倍的
        '''
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        # (batch_size, sequence_size, V_out) 将token的不同头的输出拼接起来
        Y = torch.cat(Y, dim=2)     # batch_size,sequence,v_dim
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)     # 分别归一化各个头
        # 一个前向传播的过程
        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} x_n : (batch_size, 1, hidden_size)
        @param {Any} s_n_1s : (batch_size, hidden_size/heads, v_out/ heads) * heads
        @param {Any} n :
        @return {Any}
        '''

        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(x_n[:, :, :], s_n_1s[i], n)
            #  (batch_size, 1, hidden_size/heads)     (batch_size, hidden_size/heads, v_out/ heads)
            Y.append(y)
            s_ns.append(s_n)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        # s_ns = (batch_size, hidden_size/heads, v_out/ heads) * heads
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)


        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is
