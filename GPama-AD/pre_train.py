#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/1 18:53
@File:pre_train.py
@Desc:*******************
"""
import torch
import torch.nn as nn
import os
import copy
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from MYAD12.data.MyDataset1 import MyDataSet
from MYAD12.data.dis_A import edge_index
from MYAD12.models.Decoder import Decoder
from MYAD12.models.Encoder import Encoder


# ----------------- Predictor for BYOL -----------------
class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, z):
        B, N, D = z.shape
        return self.net(z.view(B * N, D)).view(B, N, D)


class PreModel(nn.Module):
    def __init__(self, embed_dim, d_k, in_dim, latent_dim, m_layers, d_model, d_ff, time_in, time_out, is_ms=True,
                 is_ca=True, is_gnn=True, recon_len=300, pre_len=1, device=None):
        super(PreModel, self).__init__()
        # 时序间编码器参数
        self.embed_dim = embed_dim
        self.d_k = d_k

        # 时序内编码器参数
        ## 多尺度
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        ## Mamba
        self.m_layers = m_layers
        self.d_model = d_model
        self.d_ff = d_ff

        # 空间编码器参数
        self.time_in = time_in
        self.time_out = time_out

        self.device = device
        self.recon_len = recon_len  # 重构长度
        self.pre_len = pre_len  # 预测长度

        self.is_ms = is_ms
        self.is_ca = is_ca
        self.is_gnn = is_gnn

        self.encoder_online = Encoder(embed_dim=self.embed_dim, d_k=self.d_k, in_dim=self.in_dim,
                                      latent_dim=self.latent_dim, m_layers=self.m_layers, d_model=self.d_model,
                                      d_ff=self.d_ff,
                                      time_in=self.time_in,
                                      time_out=self.time_out, is_ms=self.is_ms, is_ca=self.is_ca, is_gnn=self.is_gnn,
                                      device=device)
        self.encoder_target = copy.deepcopy(self.encoder_online)  # 目标编码器
        for p in self.encoder_target.parameters():
            p.requires_grad = False  # 冻结参数

        # self.predictor = Predictor(dim=self.out_dim).to(self.device)

        # 作为对比学习中的额外投影层，将节点嵌入变换到一个新的对比学习空间，增强对比学习
        # 作用：使得模型学习更丰富的特征，使相似样本接近，不相似样本远离，增强表征能力
        # 作用于 主编码器
        self.out_dim = self.time_out * self.in_dim
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.out_dim, self.out_dim),
            nn.BatchNorm1d(self.out_dim),  # 对第三维度正则化
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_dim, self.out_dim)).to(self.device)

        self.decoder = Decoder(z_dim=self.out_dim, out_dim=self.in_dim, recon_len=self.recon_len,
                               pred_len=self.pre_len).to(self.device)

    def update_target_encoder(self, momentum: float):
        """
        通过指数移动平均EMA的方式更新EMA
        :param momentum:
        :return:
        """
        for p, new_p in zip(self.encoder_target.parameters(), self.encoder_online.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def temporal_augmentation(self, x, method='noise', scale=0.05):
        if method == 'noise':
            return x + torch.randn_like(x) * scale
        elif method == 'mask':
            # mask一段连续时间片
            B, N, M, T = x.shape
            x_masked = x.clone()
            num_mask = int(T * scale)
            for b in range(B):
                for n in range(N):
                    idx = torch.randperm(T)[:num_mask]
                    x_masked[b, n, :, idx] = 0  # 时间维度在最后一维
            return x_masked
        else:
            raise NotImplementedError

    def perturb_edges(self, edge_index, num_nodes, drop_rate=0.1, add_rate=0.1):
        """
        对边进行扰动：随机删除一些原有边 + 随机添加一些新的边。

        参数：
            edge_index: [2, E] 原始图的边列表
            num_nodes: 节点数量
            drop_rate: 删除边的比例
            add_rate: 添加边的比例

        返回：
            edge_index_perturbed: [2, E'] 扰动后的边列表
        """
        device = edge_index.device
        E = edge_index.shape[1]

        # ==== 1. 删除部分边 ====
        num_drop = int(E * drop_rate)
        if num_drop > 0:
            keep_idx = torch.randperm(E, device=device)[num_drop:]
        else:
            keep_idx = torch.arange(E, device=device)
        edge_index = edge_index[:, keep_idx]

        # ==== 2. 添加随机边 ====
        num_add = int(E * add_rate)
        row = torch.randint(0, num_nodes, (num_add,), device=device)
        col = torch.randint(0, num_nodes, (num_add,), device=device)

        new_edges = torch.stack([row, col], dim=0)

        # ==== 3. 合并并去重 ====
        edge_index_aug = torch.cat([edge_index, new_edges], dim=1)

        # 去重（可选）
        edge_index_aug = torch.unique(edge_index_aug, dim=1)

        return edge_index_aug

    def generate_views(self, x, edge_index):
        # x: [B, N, M, T]
        # edge_index: 图结构
        # 增强视图
        _, N, _, _ = x.shape
        x_aug = self.temporal_augmentation(x, method='mask', scale=0.1)
        edge_index_aug = self.perturb_edges(edge_index, num_nodes=N, drop_rate=0.1, add_rate=0.1)

        return x_aug, edge_index_aug

    def forward(self, x, edge_index):
        # print(x.shape)
        x1 = x.clone()
        x2, edge_index2 = self.generate_views(x, edge_index)
        if self.is_gnn:
            z1, mu, logvar = self.encoder_online(x1, edge_index)
            z2, _, _ = self.encoder_target(x2, edge_index2)
            B, N, D = z1.shape
            # 用于对比学习
            proj1 = self.projection_head(z1.view(B * N, D)).view(B, N, D)
            proj2 = z2.clone()
            # 用于重构、预测
            x_recon, x_pre = self.decoder(z1)
            return proj1, proj2, x_recon, x_pre, mu, logvar
        else:
            z1 = self.encoder_online(x1, edge_index)
            z2 = self.encoder_target(x2, edge_index2)
            B, N, D = z1.shape
            # 用于对比学习
            proj1 = self.projection_head(z1.view(B * N, D)).view(B, N, D)
            proj2 = z2.clone()
            # 用于重构、预测
            x_recon, x_pre = self.decoder(z1)
            return proj1, proj2, x_recon, x_pre


# ----------------- BYOL Loss -----------------
def byol_loss(p_online, z_target):
    p_online = F.normalize(p_online, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return 2 - 2 * (p_online * z_target).sum(dim=-1).mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / 51




