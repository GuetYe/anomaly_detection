#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/6/19 15:20
@File:dis_A2.py.py
@Desc:*******************
"""
import torch

# 原始邻接表（1-based 节点编号）
A = {
    1: [2, 14],
    2: [1, 3],
    3: [2, 4],
    4: [5, 3],
    5: [4, 6],
    6: [5, 7],
    7: [6, 8],
    8: [7, 9],
    9: [8, 10],
    10: [9, 11],
    11: [10, 12, 13],
    12: [11, 13],
    13: [11, 12, 14],
    14: [13, 1],
}

# 去重后的无向边集合（转为 0-based 索引）
edge_set = set()
for src in A:
    for dst in A[src]:
        i, j = src - 1, dst - 1
        if i <= j:
            edge_set.add((i, j))
        else:
            edge_set.add((j, i))

# 展开为两个方向（无向图双向边）
src_nodes = []
dst_nodes = []
for i, j in edge_set:
    src_nodes.extend([i, j])
    dst_nodes.extend([j, i])

# 构建 edge_index
# edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
# torch.save(edge_index,"edge_index.pth")
# # 打印结果
# print("edge_index shape:", edge_index.shape)
# print(edge_index)

# edge_index = torch.load("edge_index.pth")
# print(edge_index.shape)
# print(edge_index)