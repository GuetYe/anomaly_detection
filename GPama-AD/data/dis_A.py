#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2024/12/14 21:49
@File:dis_A.py
@Desc:*******************
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2024/12/10 9:56
@File:dis_A.py
@Desc:*******************
"""
"""
计算邻接矩阵A
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import os
current_dir = os.path.dirname(__file__)
# print(current_dir)
# 获取上两级目录，即 time_series/dataset
target_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'dataset'))
# 拼接目标文件的路径
file_path = os.path.join(target_dir, 'IBRL/node.txt')

# 1. 从 'node.txt' 中读取节点数据
def read_node_file(file_path):
    node_positions = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            node_id = int(parts[0])  # 节点编号
            x = float(parts[1])  # x坐标
            y = float(parts[2])  # y坐标
            node_positions.append([x, y])  # 保存为坐标对
    return np.array(node_positions)


# 2. 使用 Top-k 方法计算邻接矩阵
def create_adjacency_matrix(node_positions, k=3):
    N = node_positions.shape[0]  # 节点数量
    # 使用 K 最近邻算法计算最近的 k 个节点
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(node_positions)  # k+1 因为会包括节点自己
    distances, indices = nbrs.kneighbors(node_positions)

    # 创建邻接矩阵
    adj_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in indices[i, 1:]:  # 忽略第一个邻居（即节点自己）
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # 因为邻接矩阵是对称的
    return adj_matrix

# 3. 以距离关系设置邻接矩阵的权重，得出A
def create_adjacency_matrix2(nodes):
    # 计算节点间的欧几里得距离
    def calculate_distance(node1, node2):
        return np.linalg.norm(node1 - node2)

    # 创建邻接矩阵
    N = nodes.shape[0]
    adj_matrix = np.zeros((N, N))

    # 填充邻接矩阵
    for i in range(N):
        for j in range(N):
            if i != j:
                distance = calculate_distance(nodes[i], nodes[j])
                adj_matrix[i][j] = 1 / (distance + 1e-6)  # 使用距离的倒数作为权重，防止除以零的错误
    return adj_matrix


node_positions = read_node_file(file_path)
node_positions = node_positions[:51]
adj_matrix = create_adjacency_matrix2(node_positions)
# print(adj_matrix)
# print(adj_matrix.shape)


# 示例的邻接矩阵
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

# 找到所有非零元素的索引（即边）
row, col = torch.nonzero(adj_matrix>0.2, as_tuple=True)

# 构建edge_index (包括反向边)
edge_index = torch.stack([row, col], dim=0)

# 对于无向图，我们将反向边添加到edge_index中
# 无向图边 (i, j) 和 (j, i) 是相同的，因此只需一次性添加反向边并删除重复边
edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)

# 去除重复的边 (i, j) 和 (j, i) 相同的情况
edge_index = torch.unique(edge_index, dim=1)
# 输出结果
# print(edge_index)
# print(edge_index_final.shape)  # torch.Size([2, 96])
# 主程序
if __name__ == "__main__":
    # 读取节点信息
    # node_positions = read_node_file("../dataset/IBRL/node.txt")
    #
    # node_positions = node_positions[:51]
    # print(node_positions)
    #
    # # 设置 k 值
    # k = 3  # 每个节点与2个最近邻连接
    #
    # # 创建邻接矩阵
    # #adj_matrix = create_adjacency_matrix(node_positions, k)
    #
    # adj_matrix = create_adjacency_matrix2(node_positions)
    #
    # # 打印邻接矩阵
    # # print("Adjacency Matrix (Top-k based):")
    # print(adj_matrix)
    # print(adj_matrix.shape)
    pass