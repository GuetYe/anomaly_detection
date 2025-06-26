#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/4/14 14:44
@File:prompt.py
@Desc:*******************
"""
import torch
import torch.nn as nn


class GraphPrompt(nn.Module):
    def __init__(self, num_nodes, prompt_dim, mode='add', shared=False):
        """
        num_nodes: 节点数
        prompt_dim: prompt 向量维度
        mode: 'add' 或 'concat'
        shared: 是否共享提示向量（否则每个节点独立）
        """
        super().__init__()
        self.mode = mode
        self.shared = shared

        if shared:
            self.prompts = nn.Parameter(torch.randn(1, 1, prompt_dim))
        else:
            self.prompts = nn.Parameter(torch.randn(1, num_nodes, prompt_dim))

    def forward(self, x):
        """
        x: 编码器输出 [B, N, D]
        return: 添加提示后的表示 [B, N, D (+ prompt_dim)]
        """
        if self.shared:
            prompt = self.prompts.expand(x.size(0), x.size(1), -1)  # [B, N, prompt_dim]
        else:
            prompt = self.prompts.expand(x.size(0), -1, -1)  # [B, N, prompt_dim]

        if self.mode == 'add':
            # return x + prompt, prompt
            return prompt
        elif self.mode == 'concat':
            # return torch.cat([x, prompt], dim=-1), prompt
            return prompt
        else:
            raise ValueError("Unsupported mode: choose 'add' or 'concat'")
