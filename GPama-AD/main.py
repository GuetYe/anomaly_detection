#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Cjing_63
@Contact:cj037419@163.com
@Time:2025/5/1 18:54
@File:main.py
@Desc:*******************
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset


from train import train
from test import test

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

from torchsummary import summary
from torchstat import stat



# edge_index = edge_index.to(device)

class Main():
    def __init__(self, model_config, env_config):
        """

        :param model_config: 模型参数
        :param env_config: 环境参数
        """
        self.model_config = model_config
        self.env_config = env_config



        self.edge_index = edge_index
        # self.mode = mode

        self.cfg = {
            'slide_win': self.env_config['slide_win'],  # 滑动窗口大小
            'slide_stride': self.env_config['slide_stride'],  # 滑动步长
        }
        self.datapath = self.env_config['datapath']  # 数据集地址
        self.batch_size = self.env_config['batch_size']  # 批次大小

        self.node_num = self.model_config['node_num']  # 节点数
        self.modal_num = self.model_config['modal_num']  # 模态数
        self.time_step = self.model_config['time_step']  # 滑动窗口大小
        self.out_dim = self.model_config['out_dim']  # 时间模块最后输出维度
        self.topk = self.model_config['topk']  # 最近邻选择数目
        self.gnn_layer_num = self.model_config['gnn_layer_num']  # GNN层数
        self.time_layer_num = self.model_config['time_layer_num']  # 时间模块层数
        self.device = self.env_config['device']  # 设备

        self.is_cross = self.model_config['is_cross']  # 时间模块是否加入交叉块
        self.is_fusion = self.model_config['is_fusion']  # 并行融合 还是 串行提取

        self.lr = self.env_config['lr']  # 学习率
        self.decay = self.env_config['decay']  #
        self.epoch = self.env_config['epoch']  # 训练轮次
        self.save_path = self.env_config['save_path']  # 模型保存路径

        self.load_path = self.env_config['load_path']  # 模型加载路径
        self.val_threshold = self.env_config['val_threshold']  # 验证阈值
        self.test_threshold = self.env_config['test_threshold']  # 测试阈值

        self.model = MSTModel(self.edge_index, self.node_num, self.modal_num, self.time_step, self.out_dim,
                              device=self.device,
                              time_layer=self.time_layer_num,
                              space_layer=self.gnn_layer_num, topk=self.topk, is_cross=self.is_cross,
                              is_fusion=self.is_fusion)

    def run(self, mode):
        """
        训练--验证--测试
        :param mode:
        :return:
        """

        if mode == "train":
            train_dataset = MyDataSet(self.datapath, mode='train', config=self.cfg)  # 训练集（正常）
            train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=False)
            print(len(train_dataloader))
            train(self.model, self.save_path, train_dataloader, self.lr, self.decay, self.epoch, self.device)

        elif mode == "val":
            model = torch.load(self.load_path).to(self.device)

            val_dataset = MyDataSet(self.datapath, mode='val', config=self.cfg)  # 验证集（正常）
            val_dataloader = DataLoader(val_dataset, self.batch_size, shuffle=False)

            # 准确度，精确率，召回率，f1分数
            acc, pre, rec, f1 = test(model, val_dataloader, threshold=self.val_threshold, device=self.device)
        else:
            model = torch.load(self.load_path).to(self.device)

            test_dataset = MyDataSet(self.datapath, mode='test', config=self.cfg)  # 测试集（异常）
            test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle=False)
            acc, pre, rec, f1 = test(model, test_dataloader, threshold=self.test_threshold, device=self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-node_num', help='node num', type=int, default=51)
    parser.add_argument('-modal_num', help='modal num', type=int, default=3)
    parser.add_argument('-time_step', help='slide win', type=int, default=200)
    parser.add_argument('-out_dim', help='dimension', type=int, default=64)

    parser.add_argument('-topk', help='topk num', type=int, default=6)
    parser.add_argument('-gnn_layer_num', help='gnn_layer num', type=int, default=2)
    parser.add_argument('-time_layer_num', help='time_layer_num', type=int, default=1)

    parser.add_argument('-datapath', help='dataset path', type=str, default='../MYAD12/data/IBRL_data_anomaly.npz')
    parser.add_argument('-batch_size', help='batch size', type=int, default=6)
    parser.add_argument('-epoch', help='train epoch', type=int, default=200)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=200)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=100)
    parser.add_argument('-decay', help='decay', type=float, default=1e-4)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.005)
    parser.add_argument('-val_threshold', help='val_threshold', type=float, default=0.1)
    parser.add_argument('-test_threshold', help='test_threshold', type=float, default=0.4)

    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')

    parser.add_argument('-save_path', help='training model path', type=str, default='..//MYAD12/pretrained/mymodel.pth')
    parser.add_argument('-load_path', help='trained model path', type=str, default='..//MYAD12/pretrained/mymodel.pth')

    args = parser.parse_args()

    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    env_config = {
        'datapath': args.datapath,
        'batch_size': args.batch_size,
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'lr': args.lr,
        'decay': args.decay,
        'epoch': args.epoch,
        'save_path': args.save_path,
        'load_path': args.load_path,
        'device': args.device,
        'val_threshold': args.val_threshold,
        'test_threshold': args.test_threshold
    }

    model_config = {
        'node_num': args.node_num,
        'modal_num': args.modal_num,
        'time_step': args.time_step,
        'out_dim': args.out_dim,
        'topk': args.topk,
        'gnn_layer_num': args.gnn_layer_num,
        'time_layer_num': args.time_layer_num,
        'is_cross': True,
        'is_fusion': False,
    }

    main = Main(model_config, env_config)
    # main.run(mode='train')  # 训练模型
    # main.run(mode='val')  # 训练出阈值
    main.run(mode='test')  # 测试模型