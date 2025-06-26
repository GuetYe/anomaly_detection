#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/3 21:32
@File:Decoder.py
@Desc:*******************
"""
import torch
import torch.nn as nn



# ----------------- Future Predictor -----------------
class FuturePredictor(nn.Module):
    def __init__(self, latent_dim, out_dim, pred_len):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim * pred_len)
        )
        self.pred_len = pred_len
        self.out_dim = out_dim

    def forward(self, z):
        B, N, D = z.shape
        pred = self.fc(z)
        return pred.view(B, N, self.pred_len * self.out_dim)



class Decoder(nn.Module):
    def __init__(self, z_dim, out_dim, recon_len, pred_len):
        """
        z_dim: 编码器输出维度（例如 64）
        out_dim: 原始特征维度 D（例如 3）
        recon_len: 重构历史的时间步数 T
        pred_len: 预测未来的时间步数 T'
        """
        super().__init__()
        self.recon_len = recon_len
        self.pred_len = pred_len

        # 重构分支（输出 [B, N, T, D]）
        self.reconstruct = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, recon_len * out_dim)
        )

        # 预测分支（输出 [B, N, T', D]）
        self.predict = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, pred_len * out_dim)
        )

    def forward(self, z):
        """
        z: [B, N, z_dim]
        return:
            x_recon: [B, N, recon_len, out_dim]
            x_pred:  [B, N, pred_len, out_dim]

            x_recon: [B, N, out_dim, recon_len]
            x_pred:  [B, N, out_dim]
        """
        B, N, _ = z.shape

        # 重构
        x_recon = self.reconstruct(z)  # (B, N, T*D)
        x_recon = x_recon.view(B, N, self.recon_len, -1)

        # 预测
        x_pred = self.predict(z)       # (B, N, T'*D)
        x_pred = x_pred.view(B, N, self.pred_len, -1)

        x_recon = x_recon.transpose(2,3)
        x_pred = x_pred.squeeze(2)

        return x_recon, x_pred