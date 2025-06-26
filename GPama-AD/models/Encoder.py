#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/3 21:32
@File:Encoder.py
@Desc:*******************
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MYAD12.models.Timeblock import MultiModalFusion, Mamba_Encoder
from MYAD12.models.Spaceblock import VGAEGraphEncoder
from copy import deepcopy
from tqdm import tqdm
from MYAD12.data.dis_A import edge_index
from thop import profile


class TimeEncoder(nn.Module):
    def __init__(self, embed_dim, k_dim, in_dim, latent_dim, m_layers, d_model, d_ff, slide_win, dropout=0.05,
                 act='relu',
                 d_state=16, d_conv=4, is_ms=False, is_ca=False,
                 device=None):
        super(TimeEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.in_dim = in_dim
        self.hidden_dim = latent_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.m_layers = m_layers
        self.device = device
        self.win = slide_win
        self.is_ms = is_ms
        self.is_ca = is_ca
        if self.is_ca:
            d = 3
        else:
            d = 2

        # 变量内时序信息提取
        self.intra_modal = Mamba_Encoder(in_dim=self.in_dim, latent_dim=self.hidden_dim, d_model=self.d_model,
                                         d_ff=self.d_ff, m_layers=self.m_layers, is_ms=self.is_ms, device=self.device)
        # 变量间时序信息提取
        # self.inter_modal = VariableWiseAttention(dim=self.embed_dim, heads=self.heads, device=self.device)
        self.inter_modal = MultiModalFusion(d_model=self.embed_dim, d_k=self.k_dim, device=self.device)
        self.fc = nn.Linear(d * self.in_dim, self.in_dim).to(self.device)

    def forward(self, x):
        B, N, M, T = x.shape
        x = x.to(self.device)
        if self.is_ca:
            x1 = self.inter_modal(x)
            # print("x1:",x1.device)
            x2 = self.intra_modal(x)
            # print("x2:",x2.device)
            x = torch.cat((x1, x2, x), dim=-2).to(self.device)
            # print("信息融合")
        else:
            x1 = self.intra_modal(x)
            # print("x1:",x1.device)
            x = torch.cat((x1, x), dim=-2).to(self.device)

        x = x.view(B * N * T, -1)
        x = self.fc(x)
        x = x.view(B, N, M, T)
        return x


def temporal_augmentation(x, method='noise', scale=0.05):
    if method == 'noise':
        return x + torch.randn_like(x) * scale
    elif method == 'cutout':
        # mask一段连续时间片
        B, N, M, T = x.shape
        start = np.random.randint(0, T // 2)
        length = T // 4
        x[:, :, :, start:start + length] = 0
        return x
    else:
        raise NotImplementedError


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


class Encoder(nn.Module):
    def __init__(self, embed_dim, d_k, in_dim, latent_dim, m_layers, d_model, d_ff, time_in, time_out, is_ms=False,
                 is_ca=False,
                 is_gnn=True, device=None):
        """

        :param embed_dim:
        :param heads:
        :param in_dim:
        :param latent_dim:
        :param m_layers:
        :param d_model:
        :param d_ff:
        :param time_in:
        :param time_out:
        :param device:
        """
        super(Encoder, self).__init__()
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
        self.is_ms = is_ms  # 是否多尺度
        self.is_ca = is_ca  # 是否模态融合

        # 空间编码器参数
        self.time_in = time_in
        self.time_out = time_out
        self.is_gnn = is_gnn  # 是否使用图神经网络

        self.device = device

        self.temporal_encoder = TimeEncoder(embed_dim=self.embed_dim, k_dim=self.d_k, in_dim=self.in_dim,
                                            latent_dim=self.latent_dim, m_layers=self.m_layers, d_model=self.d_model,
                                            d_ff=self.d_ff,
                                            slide_win=self.time_in, is_ms=self.is_ms, is_ca=self.is_ca,
                                            device=self.device)

        self.graph_encoder = VGAEGraphEncoder(in_dim=self.time_in, hidden_dim=time_out).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * time_in, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim * time_out)
        ).to(self.device)

    def forward(self, x, edge_index):
        # print("start time encoder")
        B, N, M, T = x.shape
        x = self.temporal_encoder(x)
        # print("time:", x.shape)
        if self.is_gnn:
            x = x.view(B * M, N, T)
            z, mu, logvar = self.graph_encoder(x, edge_index)
            z = z.view(B, N, -1)
            return z, mu, logvar
        else:
            x = x.view(B,N,-1)
            # print(x.shape)
            z = self.mlp(x)
            return z


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    B, N, M, T = 16, 51, 3, 300
    data = torch.randn(B, N, M, T)
    print(data.shape)

    model = Encoder(embed_dim=16, d_k=32, in_dim=3, latent_dim=16, m_layers=2, d_model=64, d_ff=128, time_in=300,
                    time_out=64, is_ms=False, is_ca=False,is_gnn=True, device=device)
    # out,_,_= model(data,edge_index)
    # print(out.shape)

    # B, N, M, T = 16, 51, 3, 300
    # data = torch.randn(B, N, M, T)
    macs, params = profile(model, inputs=(data, edge_index), verbose=False)
    flops = 2 * macs  # 1 MAC（乘加） = 2 FLOPs
    print(f"FLOPs: {flops:.2e}, Params: {params:.2e}")
