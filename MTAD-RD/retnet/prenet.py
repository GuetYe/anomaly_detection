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

from retnet.retnet import RetNet
from util.base import mprint


class PreTrainNet(nn.Module):
    """ 预训练网络 """
    def __init__(self, hidden_size, sequence_len, double_v_dim=False,
                    layer_size = 5, layer_heads = 1,ffn_size = 6, blocks = 5,
                    gat_head = 3):
        '''
        @Author: WangSuxiao
        @description: 创建特征提取网络的预训练模型
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

        super(PreTrainNet, self).__init__()
        self.retnet = RetNet(hidden_size=hidden_size,
                                sequence_len=sequence_len,
                                double_v_dim=double_v_dim,
                                layer_size=layer_size,
                                layer_heads=layer_heads,
                                ffn_size=ffn_size,
                                blocks=blocks,
                                gat_head=gat_head)


    def forward(self, X:torch.Tensor, A = None):
        '''
        @Author: WangSuxiao
        @description:
        @param {Any} self :
        @param {Any} X : (batch_size, node_size, sequence_length, hidden_size)
        @return {Any}
        '''
        # 特征提取的前向计算
        X = self.retnet(X, A)

        # 平均池化获取子图特征表示

        # 采样正负样本对


        return X







