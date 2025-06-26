#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/12 20:40
@File:gp_finetune.py
@Desc:*******************
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from MYAD12.data.MyDataset import MyDataSet
from MYAD12.models.prompt2 import GraphPrompt
from MYAD12.pre_train import PreModel
from MYAD12.data.dis_A import edge_index

class PFTModel(nn.Module):
    def __init__(self, pretrain_model, graph_prompt, device=None):
        super(PFTModel, self).__init__()
        self.PTM = pretrain_model
        self.encoder_online = pretrain_model.encoder_online
        self.encoder_target = pretrain_model.encoder_target
        self.decoder = pretrain_model.decoder
        self.projection_head = pretrain_model.projection_head
        self.prompt = graph_prompt
        self.device = device
        self.update_target_encoder = pretrain_model.update_target_encoder
        self.generate_views = pretrain_model.generate_views

        for p in self.PTM.parameters():
            p.requires_grad = False


    def forward(self, x, edge_index):
        # print(x.device)
        # print(edge_index.device)
        x1 = x.clone()
        x2, edge_index2 = self.generate_views(x, edge_index)
        z1, mu, logvar = self.encoder_online(x1, edge_index)
        z2, _, _ = self.encoder_target(x2, edge_index2)
        B, N, D = z1.shape
        z1_prompt = self.prompt(z1)
        z1 = z1 + z1_prompt
        # 用于对比学习
        proj1 = self.projection_head(z1.view(B * N, D)).view(B, N, D)
        proj2 = z2.clone()
        # 用于重构、预测
        x_recon, x_pre = self.decoder(z1)
        return proj1, proj2, x_recon, x_pre, mu, logvar


# ----------------- BYOL Loss -----------------
def byol_loss(p_online, z_target):
    p_online = F.normalize(p_online, dim=-1)
    z_target = F.normalize(z_target.detach(), dim=-1)
    return 2 - 2 * (p_online * z_target).sum(dim=-1).mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / 51



