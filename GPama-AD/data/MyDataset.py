#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/3/1 14:46
@File:MyDataset1.py
@Desc:*******************
"""
"""
之前的数据集没有加入异常

现在针对添加异常的数据进行数据集加载自定义

准备完成两版
1 训练（正常）测试（异常）  
2 预训练（正常）训练（训练）测试（异常）

本文件先完成第一版 2025/3/1

读取数据集 数据归一化 滑动窗口 （训练数据 目标数据 标签）
"""

import numpy as np
import torch
from torch import nn
# from MYAD4.dataset.dis_A import edge_index
from torch.utils.data import Dataset, DataLoader


class MyDataSet(Dataset):
    def __init__(self, data_path, mode='train', config=None):
        """

        :param data_path:
        :param mode:
        :param config:
        """
        super(MyDataSet, self).__init__()

        datastr = np.load(data_path)
        self.data = datastr["data_ab"]
        self.labels = datastr["data_la"]

        # 正常数据：含有异常数据 5：1：2:2
        self.seg1 = int(self.data.shape[-1] * 0.5)
        self.seg2 = int(self.data.shape[-1] * 0.6)
        self.seg3 = int(self.data.shape[-1] * 0.8)

        # print(self.seg)
        # self.edge_index = edge_index
        self.mode = mode
        self.config = config

        self.labels = self.labels_merge()

        # 将numpy格式转化为tensor格式
        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)

        if self.mode == "pretrain":
            self.data_t, _, _, _ = self.data_Preprocess(self.data, self.labels)
        elif self.mode == "train":
            _, self.data_t, _, _ = self.data_Preprocess(self.data, self.labels)
        elif self.mode == "val":
            _, _, self.data_t, _ = self.data_Preprocess(self.data, self.labels)
        elif self.mode == "test":
            _, _, _, self.data_t = self.data_Preprocess(self.data, self.labels)

        # print("data_t:", type(data_t))
        # print(data_t.shape)

        self.x, self.y, self.label_x, self.label_y = self.process(self.data_t)

    def labels_merge(self):
        """
        原有的标签都是每个模态
        将每个时间步上 不同模态的标签 转化为 当前时间步所在节点上的标签
        同一节点的 不同模态 只要出现异常 就是该节点异常

        比如
        某个节点某个时间步 不同模态标签 0 1 0 ————> 该节点该时间步 标签 1
        :return:
        """
        new_label = (self.labels != 0).any(axis=1).astype(int)
        return new_label

    def normalize(self, data, mode=False):
        """
        False 标准差 True 最大最小值
        :param mode: Z_score标准差归一化  Min_Max缩放归一化
        :return:
        """
        # 计算 min 和 max，保持数据形状
        if isinstance(data, torch.Tensor):
            data = data.numpy()  # 转换为 NumPy 数组

        if mode:

            data_min = np.min(data, axis=2, keepdims=True)  # (51, 3, 1)
            data_max = np.max(data, axis=2, keepdims=True)  # (51, 3, 1)

            # 避免除零错误
            data_normalized = (data - data_min) / (data_max - data_min + 1e-8)

        else:
            # 计算均值和标准差
            data_mean = np.mean(data, axis=2, keepdims=True)  # (51, 3, 1)
            data_std = np.std(data, axis=2, keepdims=True)  # (51, 3, 1)

            # data_mean = data.mean(dim=2, keepdim=True)  # 按节点和时间求均值
            # data_std = data.std(dim=2, keepdim=True)  # 按节点和时间求标准差

            # 避免除零错误
            data_normalized = (data - data_mean) / (data_std + 1e-8)

        data_normalized = torch.from_numpy(data_normalized)

        return data_normalized

    def data_Preprocess(self, data, labels):
        """
        第一版 把整个数据集一起进行归一化处理了 不同数据集应该单独处理 大失误
        对训练集和测试集单独处理
        测试集加上标签

        先分割数据集，单独归一化，再划分标签并拼接
        训练集划分出 训练集和验证集
        :return:
        """
        # print("before normal :",type(data))

        pretrain_data = data[:, :, :self.seg1]  # (51, 3, 12900)  5分 12900*0.5 10320
        pretrain_label = labels[:, :self.seg1]  # (51, 2580)
        pretrain_data = self.normalize(pretrain_data, mode=False)

        train_data = data[:, :, self.seg1:self.seg2]  # (51, 3, 12900)  3分
        train_label = labels[:, self.seg1:self.seg2]  # (51, 2580)
        train_data = self.normalize(train_data, mode=False)

        val_data = data[:, :, self.seg2:self.seg3]  # (51, 3, 2580)  两份 12900*0.2  2580
        val_label = labels[:, self.seg2:self.seg3]  # (51, 2580)
        val_data = self.normalize(val_data, mode=False)

        test_data = data[:, :, self.seg3:]  # (51, 3, 1290)   12900*0.1  1290
        test_label = labels[:, self.seg3:]  # (51, 1290)
        test_data = self.normalize(test_data, mode=False)

        return (pretrain_data, pretrain_label), (train_data, train_label), (val_data, val_label), (
            test_data, test_label)

    def process(self, data):
        # print("data:", type(data))
        # print(data.shape)

        data, labels = data
        # print("labels:", type(labels))
        # print(labels.shape)
        # print("data:", type(data))
        # print(data.shape)

        x_arr, y_arr = [], []  # 初始化特征和目标列表
        labels_x_arr = []  # 初始化特征标签列表
        labels_y_arr = []  # 初始化目标标签列表

        # 从配置中获取滑动窗口的大小和步幅
        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_test = self.mode

        node_num, modal_num, total_time_len = data.shape  # 获取数据的形状，node_num是节点数，total_time_len是时间序列的长度
        # print("原始数据的节点数和时序长度 node_num,total_time_len:", node_num, total_time_len)
        # 设置滑动窗口的范围，训练模式下按步幅划分，测试模式下仅按窗口大小划分
        if is_test == 'test':
            rang = range(slide_win, total_time_len)
        else:
            rang = range(slide_win, total_time_len, slide_stride)
        # 对每个滑动窗口位置进行遍历，生成特征和目标数据
        for i in rang:
            # 获取当前时间步前一个窗口大小的特征数据（滑动窗口）
            ft = data[:, :, i - slide_win:i]
            # 获取当前时间步的目标值
            tar = data[:, :, i]
            # 将特征和目标数据加入到列表中
            x_arr.append(ft)
            y_arr.append(tar)
            # 标签数据（目标标签）也加入到列表中
            labels_x_arr.append(labels[:, i - slide_win:i])
            # 标签数据（目标标签）也加入到列表中
            labels_y_arr.append(labels[:, i])
        # 将特征和目标列表转换为 PyTorch 张量
        x = torch.stack(x_arr).contiguous()
        # print("x:", x.shape)
        y = torch.stack(y_arr).contiguous()
        # print("y:", y.shape)
        # 将标签数据转换为 PyTorch 张量
        # labels = torch.Tensor(labels_arr).contiguous()
        labels_x = torch.stack(labels_x_arr).contiguous()
        labels_y = torch.stack(labels_y_arr).contiguous()
        # print("label:", labels.shape)
        # 返回处理后的特征数据、目标数据和标签
        return x, y, labels_x, labels_y

    def __getitem__(self, item):
        # # 获取特定索引的特征数据
        # feature = self.x[item].double()
        # # 获取特定索引的目标数据
        # target = self.y[item].double()
        # # 获取特定索引的标签数据
        # label = self.label[item].double()

        # 获取特定索引的特征数据
        feature = self.x[item]
        # 获取特定索引的目标数据
        target = self.y[item]
        # 获取特定索引的标签数据
        label_x = self.label_x[item]
        # 获取特定索引的标签数据
        label_y = self.label_y[item]

        return feature, target, label_x, label_y

    def __len__(self):

        return len(self.x)


# data_path = "./data/IBRL_data_anomaly2.npz"
# config = {
#     'slide_win': 300,  # 窗口大小为3
#     'slide_stride': 100  # 步幅为1
# }
#
# train_dataset = MyDataSet(data_path,  mode='train', config=config)
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# test_dataset = MyDataSet(data_path, mode='test', config=config)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# pretrain_dataset = MyDataSet(data_path, mode='pretrain', config=config)
# pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True)


if __name__ == "__main__":
    config = {
        'slide_win': 300,  # 窗口大小为3
        'slide_stride': 100  # 步幅为1
    }
    data_path = "./IBRL_data_anomaly.npz"

    dataset = MyDataSet(data_path, mode='train', config=config)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    print(dataset.__len__())
    print(dataset.x.shape)  # torch.Size([75, 51, 4, 300])  标签位为[:,:,-1,:]
    print(dataset.y.shape)  # torch.Size([75, 51, 4]) 标签位为[:,:,-1]
    # dataloader = train_dataloader
    print(len(dataloader))
    for batch_idx, (feature, target, label_x, label_y) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Feature shape:", feature.shape)  # [batch_size, slide_win, node_num]
        print("Target shape:", target.shape)  # [batch_size, node_num]
        # print(target)
        print("Label shape:", label_x.shape)  # [batch_size, node_num]
        print(label_x.any(dim=-1).to(torch.int))
        print("Label shape:", label_y.shape)  # [batch_size, node_num]
        print(label_y)
        # print(label)
        print()
        break
