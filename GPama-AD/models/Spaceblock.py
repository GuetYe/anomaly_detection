#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/3 21:32
@File:Spaceblock.py
@Desc:*******************
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


# ----------------- Graph Encoder -----------------
class VGAEGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # self.gcn1 = GCNConv(in_dim, 2 * hidden_dim)
        # self.gcn2 = GCNConv(2 * hidden_dim, hidden_dim)
        self.num_nodes = 51
        self.gcn = GCNConv(in_dim, hidden_dim)
        self.act = nn.ELU()

        self.gcn_mu = GCNConv(hidden_dim, hidden_dim)
        self.gcn_logvar = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.act(self.gcn(x, edge_index))
        # print(x.device)
        mu = self.gcn_mu(x, edge_index)
        # print(mu.device)
        logvar = self.gcn_logvar(x, edge_index)
        # print(logvar.device)
        z = self.reparameterize(mu, logvar)
        # adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.num_nodes
