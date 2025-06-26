#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/3 21:32
@File:Timeblock.py
@Desc:*******************
"""
"""
时序特征提取
时序间：时序升维+交叉融合
时序内：多尺度膨胀因果卷积+mamba
时序特征=原始特征+时序内特征+时序间特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
from thop import profile
"""
交叉模块
"""


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, device):
        super(CrossAttention, self).__init__()

        self.device = device
        # 定义用于查询、键和值的线性变换
        self.W_Q = nn.Linear(d_model, d_k).to(self.device)  # 将输入的 d_model 映射到 d_k 维度（查询）
        self.W_K = nn.Linear(d_model, d_k).to(self.device)  # 将输入的 d_model 映射到 d_k 维度（键）
        self.W_V = nn.Linear(d_model, d_k).to(self.device)  # 将输入的 d_model 映射到 d_k 维度（值）

    def forward(self, queries, keys, values):
        """
        :param queries: 输入查询张量，形状 (batch_size, 51, seq_len, d_model)
        :param keys: 输入键张量，形状 (batch_size, 51, seq_len, d_model)
        :param values: 输入值张量，形状 (batch_size, 51, seq_len, d_model)
        :return: 融合后的输出张量，形状 (batch_size, 51, seq_len, d_model)
        """
        # 进行线性变换
        Q = self.W_Q(queries)  # 变换后的查询，形状 (batch_size, 51, seq_len, d_k)
        K = self.W_K(keys)  # 变换后的键，形状 (batch_size, 51, seq_len, d_k)
        V = self.W_V(values)  # 变换后的值，形状 (batch_size, 51, seq_len, d_k)

        # 计算注意力分数，形状 (batch_size, 51, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))

        # 计算注意力权重，进行 softmax 操作
        attn_weights = F.softmax(scores, dim=-1)  # 形状 (batch_size, 51, seq_len, seq_len)

        # 将注意力权重应用于值张量，得到最终输出，形状 (batch_size, 51, seq_len, d_k)
        output = torch.matmul(attn_weights, V)  # 形状 (batch_size, 51, seq_len, d_k)

        return output


"""
多时序信息融合模块
"""


class MultiModalFusion(nn.Module):
    def __init__(self, d_model, d_k, device):
        """

        :param d_model: 模态特征维度
        :param d_k: 键值维度
        :param device: 设备
        :param is_cross: yes CA  no Pierre's coefficient 皮埃尔系数
        """
        super(MultiModalFusion, self).__init__()
        self.d_model = d_model  # 模态特征维度
        self.d_k = d_k  # 注意力机制中键和值的维度

        self.device = device

        self.cross_attn = CrossAttention(self.d_model, self.d_k, device=self.device)  # 交叉注意力机制

    def forward(self, x):
        """
        :param x: 输入张量，形状 (batch_size, 51, 4, seq_len)
        :return: 融合后的特征张量，形状 (batch_size, 51, 3, seq_len)
        """
        # 获取输入张量的形状
        batch_size, nodes, modals, seq_len = x.shape
        x = x.to(self.device)

        modalities = x

        # 初始化用于存储每个模态的交叉注意力计算结果
        fused_features = []

        # 对每个模态与其他模态进行交叉注意力计算
        for i in range(modals):  # 3个模态
            modal_i = modalities[:, :, i, :].to(self.device)  # 取第 i 个模态，形状 (batch_size, 51, seq_len)

            # 将模态的形状扩展到 (batch_size, 51, seq_len, d_model)
            modal_i = modal_i.unsqueeze(-1).expand(-1, -1, -1, self.d_model)  # (batch_size, 51, seq_len, d_model)

            # 初始化一个空张量，存储每个模态的交叉注意力结果
            fused_i = []

            # 对该模态与其他模态进行交叉注意力计算
            for j in range(modals):  # 3个模态
                if i != j:  # 确保不对同一模态计算交叉注意力
                    modal_j = modalities[:, :, j, :].to(self.device)  # 取第 j 个模态，形状 (batch_size, 51, seq_len)

                    # 将模态 j 的形状扩展到 (batch_size, 51, seq_len, d_model)
                    modal_j = modal_j.unsqueeze(-1).expand(-1, -1, -1,
                                                           self.d_model)  # (batch_size, 51, seq_len, d_model)

                    # 进行交叉注意力计算
                    fused_i_j = self.cross_attn(modal_i, modal_j,
                                                modal_j)  # 融合后，形状 (batch_size, 51, seq_len, d_model)

                    fused_i.append(fused_i_j)

            # 对该模态的交叉注意力结果进行拼接，形状仍然保持 (batch_size, 51, seq_len, d_model)
            fused_i = torch.cat(fused_i, dim=-1)  # (batch_size, 51, seq_len, d_model * 2)

            # 将每个模态的融合结果加入到最终的输出中
            # 对模态进行拼接，最终的输出形状为 (batch_size, 51, 3, seq_len)
            fused_features.append(fused_i)

        # 将所有模态的交叉注意力结果堆叠起来，最终形状为 (batch_size, 51, 3, seq_len)
        fused_features = torch.stack(fused_features, dim=2).to(self.device)  # (batch_size, 51, 3, seq_len)

        # 对最后一个维度（64）进行平均池化
        fused_features = fused_features.mean(dim=-1)  # 平均池化，移除最后一个维度（64）
        return fused_features


"""
mamba块
"""


class Mamba_Encoder_Layer(nn.Module):
    def __init__(self, d_model=3, d_ff=128, dropout=0.05, act='relu', d_state=16, d_conv=4, device=None):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv).to(device)
        self.lin1 = nn.Linear(d_model, d_ff).to(device)
        self.lin2 = nn.Linear(d_ff, d_ff).to(device)
        self.ln = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.act = F.relu if act == "relu" else F.gelu

    def forward(self, x):
        # print(x.shape)
        x = self.mamba(x)
        x = self.lin2(self.act(self.lin1(x)))

        return x


"""
结合多尺度膨胀卷积的mamba解码器
"""


class Mamba_Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim, m_layers, d_model, d_ff, dropout=0.05, act='relu', d_state=16, d_conv=4,
                 is_ms=False,
                 device=None):
        super().__init__()
        self.device = device
        self.m_layers = m_layers
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.is_ms = is_ms
        self.dilations = [1, 2, 4, 8]

        self.convs = nn.ModuleList([
            nn.Conv1d(self.in_dim, self.latent_dim, kernel_size=3, dilation=d, padding=d) for d in self.dilations
        ]).to(self.device)
        self.fuse = nn.Conv1d(len(self.dilations) * self.latent_dim, self.d_model, kernel_size=1).to(self.device)
        self.relu = nn.ReLU()

        if self.is_ms:
            self.mamba_layers = nn.ModuleList(
                [Mamba_Encoder_Layer(self.d_model if i == 0 else self.d_ff, self.d_ff, dropout, act, d_state, d_conv,
                                     device=self.device)
                 for i in range(self.m_layers)])
            self.lin1 = nn.Linear(self.d_ff, self.d_model).to(device)
            self.lin2 = nn.Linear(self.d_model, self.in_dim).to(device)
        else:
            self.mamba_layers = nn.ModuleList(
                [Mamba_Encoder_Layer(self.in_dim if i == 0 else self.d_model, self.d_model, dropout, act, d_state, d_conv,
                                     device=self.device)
                 for i in range(self.m_layers)])
            self.lin1 = nn.Linear(self.d_model, self.d_model).to(device)
            self.lin2 = nn.Linear(self.d_model, self.in_dim).to(device)

    def forward(self, x):
        """

        :param x: [B, N, M, T]
        :return: [B,N,T,D]
        """
        x = x.to(self.device)
        B, N, M, T = x.shape
        # print("原始数据：", x.shape)
        x = x.view(B * N, M, T)  # 类似 B C T
        # print("数据转化：", x.shape)
        if self.is_ms:
            # 多尺度膨胀因果卷积
            outs = [conv(x.to(conv.weight.dtype)) for conv in self.convs]
            x = torch.cat(outs, dim=1)
            # print("尺度拼接：", x.shape)  # (B*N, 3*hidden, T)
            x = self.relu(self.fuse(x))  # (B*N, out_dim, T)
            # print("信息膨胀：", x.shape)
        x = x.transpose(1, 2)  # (B*N, T, C')
        for i in range(self.m_layers):
            x = self.mamba_layers[i](x)
        # print("mamba_layer:", x.shape)
        x = self.lin2(self.relu(self.lin1(x)))
        x = x.view(B, N, M, T)

        return x


if __name__ == "__main__":
    B, N, M, T = 16, 51, 3, 300
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Mamba_Encoder(in_dim=3, latent_dim=16, m_layers=2, d_model=64, d_ff=128,is_ms=False, device=device)
    x = torch.randn(B, N, M, T)
    # output = model(x)
    # print(output.shape)

    macs, params = profile(model, inputs=(x), verbose=False)
    flops = 2 * macs  # 1 MAC（乘加） = 2 FLOPs
    print(f"FLOPs: {flops:.2e}, Params: {params:.2e}")
