#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/4/14 14:44
@File:scorer.py
@Desc:*******************
"""
# models/scorer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class AnomalyScorer(nn.Module):
    def __init__(self, method='sum', alpha=0.5, use_prompt=False):
        """
        method: 融合方式 ('sum' or 'weighted')
        alpha: 融合权重（当 method='weighted'）
        use_prompt: 是否加入提示引导调节异常分数
        """
        super().__init__()
        self.method = method
        self.alpha = alpha
        self.use_prompt = use_prompt

        if use_prompt:
            self.prompt_weight = nn.Parameter(torch.ones(1))  # 可学习权重调节提示引导

    def forward(self, recon_error, pred_error, prompt=None):
        """
        recon_error: [B, N] 重构误差
        pred_error:  [B, N] 预测误差
        prompt:      [B, N] 可选，提示引导分数

        return: anomaly_score: [B, N]
        """
        if self.method == 'sum':
            score = recon_error + pred_error
        elif self.method == 'weighted':
            score = self.alpha * recon_error + (1 - self.alpha) * pred_error
        else:
            raise ValueError("Unsupported method")

        if self.use_prompt and prompt is not None:
            score = score * (1 + self.prompt_weight * prompt)

        return score


def compute_errors(x_recon, x_pred, x_true_hist, x_true_future):
    """
    x_recon: [B, N, T, D]
    x_pred:  [B, N, D]
    x_true_hist: [B, N, T, D]
    x_true_future: [B, N, D]
    return: [B, N] 的误差
    """
    recon_error = F.mse_loss(x_recon, x_true_hist, reduction='none')  # [B, N, M, T]
    pred_error = F.mse_loss(x_pred, x_true_future, reduction='none')  # [B, N, M]

    recon_error = recon_error.mean(dim=[2, 3])  # → [B, N]
    pred_error = pred_error.mean(dim=2)         # → [B, N]

    return recon_error, pred_error




def evaluate_auc(anomaly_score, labels):
    """
    anomaly_score: [B, N] 或展平 [B*N]
    labels: 同 shape，0=正常，1=异常
    """
    return roc_auc_score(labels.reshape(-1), anomaly_score.detach().cpu().numpy().reshape(-1))


def evaluate_all_metrics(scores, labels, threshold=None, topk_ratio=0.1):
    """
    scores: numpy array [B, N] 或 [total_samples]
    labels: numpy array 同 shape，0/1 异常标签
    threshold: 手动设置的分数阈值
    topk_ratio: 如果 threshold=None，就自动用 top-k 作为阈值

    return: auc, acc, precision, recall, f1
    """
    scores = scores.flatten()
    labels = labels.flatten()

    auc = roc_auc_score(labels, scores)

    # 自动选择阈值
    if threshold is None:
        k = int(len(scores) * topk_ratio)
        threshold = np.sort(scores)[-k]  # top-k 中最小的作为阈值

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        'auc': auc,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }