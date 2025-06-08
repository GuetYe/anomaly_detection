import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data.dataloader import Dataset, DataLoader
from datetime import datetime
# from data.dataset.RevIN import RevIN
import random
import torch
import copy
import os
import math
from scipy.sparse import diags, csr_matrix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# revin_layer = RevIN(3, affine=True)

A = {
    1: [2, 3, 35, 33],
    2: [1, 35, 4],
    3: [1, 4, 6, 33],
    4: [2, 3, 5, 6],
    5: [4, 6, 7],
    6: [3, 4, 7, 10],
    7: [5, 6, 8, 10],
    8: [7, 9, ],
    9: [8, 7, 10],
    10: [6, 7, 9],
    11: [12, 13],
    12: [11, 13],
    13: [11, 12, 14],
    14: [13, 14, 15, 18],
    15: [14, 18, 17, 16],
    16: [15, 17],
    17: [16, 18, 19],
    18: [14, 15, 19],
    19: [18, 17, 20, 21],
    20: [19, 21, 22, 17],
    21: [19, 20, 22, 23],
    22: [20, 21, 23, 24],
    23: [27, 22, 25, 21],
    24: [25, 22],
    25: [24, 23, 26, 27],
    26: [25, 27, 28],
    27: [23, 25, 26, 28, 29],
    28: [26, 27, 29, 30],
    29: [31, 30, 28, 27],
    30: [28, 29, 31],
    31: [29, 30, 32, 33],
    32: [31, 33, 34],
    33: [1, 34, 32, 31],
    34: [32, 33, 35, 36],
    35: [1, 37, 34, 36],
    36: [34, 35, 37, 38],
    37: [35, 36, 39],
    38: [36, 37, 39, 40],
    39: [40, 38, 37, 43],
    40: [39, 38, 43, 41],
    41: [42, 40, 43],
    42: [41, 44],
    43: [41, 40, 39, 45],
    44: [45],
    45: [44, 43, 47, 46],
    46: [45, 48, ],
    47: [45, 48, 49],
    48: [46, 47, 51],
    49: [50, 51, 47, 48],
    50: [51, 49],
    51: [50, 48, 52],
    52: [51, 53],
    53: [52, 54],
    54: [53]
}

def timestampTodatetime(timestamp):
    '''
    @Author: WangSuxiao
    @description: 时间戳转字符串
    @param {Any} timestamp :
    @return {Any}
    '''
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_date


def strToTimestamp(date_str, time_str):
    '''
    @Author: WangSuxiao
    @description: 将日期和时间转化为时间戳
    @param {Any} date_str : 日期字符串
    @param {Any} time_str : 时间字符串
    @return {Any}
    '''
    # date_str = "2004-03-15"
    # time_str = "07:32:43.879505"

    datetime_str = f"{date_str} {time_str}"
    try:
        dt_object = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            dt_object = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(time_str)
            raise ValueError("Invalid time format")
    timestamp = datetime.timestamp(dt_object)
    # print(f"拼接后的时间戳：{timestamp}")
    return timestamp


def search(data, target):
    '''
    @Author: WangSuxiao
    @description: 递增序列的二分查找，找到最大的小于等于目标值的item的index
    @param {Any} data : list of item
    @param {Any} target : target item
    @return {Any} : max_index{item <= target}
    '''
    l = len(data) - 1
    f = 0
    while f <= l:
        m = (l + f) // 2
        if data[m] > target:
            l = m - 1
        else:
            f = m + 1
    return l


def getLocationFromTxt(file_path="./data/raw/locs.txt") -> list:
    '''
    @Author: WangSuxiao
    @description: 从文件中读取节点的坐标位置
    @return {Any} list of list
    '''

    locs = []  # 为各个节点创建一个空列表，存储各节点的记录
    with open(file_path, 'r') as file:
        for line_no, line in enumerate(file):
            # 将各行数据按空格分隔
            item_data = line.split()
            try:
                item_data[0] = int(item_data[0])
            except Exception:
                print(line_no, "转换失败")
            for i in range(1, len(item_data)):
                try:
                    item_data[i] = float(item_data[i])
                except Exception:
                    print(line_no, "转换失败")
            locs.append(item_data)
    return locs


def getNeighbour(data, threshold) -> dict:
    '''
    @Author: WangSuxiao
    @description: 获取各个节点的邻居
    @return {Any}
    '''
    node_num = 54
    domain = dict()
    distance = np.array([[0] * node_num for _ in range(node_num)], dtype=np.float64)
    for i in range(0, node_num - 1):
        for j in range(i + 1, node_num):
            d = np.sqrt((data[i][1] - data[j][1]) ** 2 + (data[i][2] - data[j][2]) ** 2)
            # print(data[i][1], data[j][1]," = ",(data[i][1] - data[j][1]) ** 2,"    ", data[i][2], data[j][2], (data[i][2] - data[j][2]) ** 2 ,"    ",d)
            distance[i, j] = d
            distance[j, i] = d
            # print(distance[i,j], distance[j,i])
    for i, item in enumerate(distance):
        domain[i] = []
        for j, n in enumerate(item):
            if n < threshold and 0 < n:
                domain[i].append(j)
    return domain


def pearsonCorrelationCoefficient(vector1, vector2):
    '''
    @Author: WangSuxiao
    @description: 计算向量之间相关性
    @param {Any} vector1 : 向量1
    @param {Any} vector2 : 向量2
    @return {Any}
    '''
    assert len(vector1) == len(vector2), "输入向量的长度不一致"
    # 计算均值
    mean_vector1 = np.mean(vector1)
    mean_vector2 = np.mean(vector2)
    # 计算协方差
    covariance = np.sum((vector1 - mean_vector1) * (vector2 - mean_vector2))
    # 计算标准差
    std_dev_vector1 = np.sqrt(np.sum((vector1 - mean_vector1) ** 2))
    std_dev_vector2 = np.sqrt(np.sum((vector2 - mean_vector2) ** 2))
    # 计算皮尔逊相关系数
    pearson_coefficient = covariance / (std_dev_vector1 * std_dev_vector2)

    return pearson_coefficient


def reflect_points(x, y, x1, y1, x2, y2) -> tuple:
    '''
    @Author: WangSuxiao
    @description: (x1, y1)与(x2, y2)构成直线L，计算点P(x, y)关于L的对称点
    @param {Any} x : P_x
    @param {Any} y : P_y
    @param {Any} x1 : L2 x1
    @param {Any} y1 : L2 y1
    @param {Any} x2 : L2 x2
    @param {Any} y2 : L2 y2
    @return {Any} tuple (x', y')
    '''
    assert y1 != y2, "斜率为0"
    # 两点之间直线L2的斜率和截距
    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1
    # 垂直于L2且过P点的直线L2'
    m2 = -1 / m1
    b2 = y - m2 * x

    # return find_intersection(m1, b1, m2, b2)
    # 计算L2和L2'的交点
    mid_x = (b2 - b1) / (m1 - m2)
    mid_y = m1 * mid_x + b1

    # return mid_x, mid_y
    # 计算P关于L2的对称点
    traget_x = 2 * mid_x - x
    traget_y = 2 * mid_y - y
    if False:
        print("m1 : ", m1, "m2 : ", m2)
        print("mid_x : ", mid_x, "mid_y : ", mid_y)
        print(traget_x, traget_y)
    return traget_x, traget_y


def node51254(s, remove_list=[5, 11, 15]) -> int:
    '''
    @Author: WangSuxiao
    @description: 51个节点映射为50个节点
    @param {Any} s : 代转化的数据
    @param {Any} remove_list : 被删除的节点
    @return {Any} 转化后的节点编号
    '''
    sorted(remove_list)
    t = s
    for i in remove_list:
        if t >= i:
            t = t + 1
        else:
            break
    return t


def node54251(s, remove_list=[5, 11, 15]) -> int:
    '''
    @Author: WangSuxiao
    @description: 51个节点映射为50个节点
    @param {Any} s : 代转化的数据
    @param {Any} remove_list : 被删除的节点
    @return {Any} 转化后的节点编号
    '''
    sorted(remove_list)
    t = s
    for i in remove_list:
        if s >= i:
            t = t - 1
    return t


# 测试51和54之间的转化
# for i in range(55):
#         if i not in [5,11,15]:
#             assert i == node51254(node54251(i)), i

def injectCorrelationAnomalyBetweenNeighbour(data, remove_list=[5, 11, 15], windows_scale=(0.2, 0.6)) -> list:
    '''
    @Author: WangSuxiao
    @description: 注入节点间相关性异常
    @param {Any} data : 某段`原始`数据集
    @param {Any} A : 原始邻接矩阵(元素为节点编号，1-54)
    @param {Any} start : 本段数据相对整体数据的位置
    @param {Any} file_name : 记录操作日志
    @param {Any} remove_list : 删除的节点编号
    @param {Any} windows_scale : 窗口缩放的尺度
    @return {Any}
        转化后的数据，转化窗口的起始位置，转化窗口的结束位置，邻居节点
        窗口位置是相对于所传入标准窗口（例如：300）而言
    '''
    node_num, mode_num, sequences_len = data.shape
    finish_nodes = set()  # 记录处理完成的节点
    labels = np.zeros(data.shape)
    neighbours = dict()
    while (len(finish_nodes) != node_num):
        # 0. 选择节点
        node_index = random.randint(0, node_num - 1)
        # for node_index in range(0,node_num):
        node_54 = node51254(node_index + 1)
        neighbour_indexs = [node54251(index) - 1 for index in A[node_54] if index not in remove_list]
        while len(neighbour_indexs) == 0 or node_index in finish_nodes:
            # print(f"节点: {node_54} 邻居为空")
            finish_nodes.add(node_index)  # 没用邻居，则该节点不再处理
            node_index = random.randint(0, node_num - 1)
            node_54 = node51254(node_index + 1)
            # print("node_index : ",node_index)
            # print("node_54 : ",node_54)
            neighbour_indexs = [node54251(index) - 1 for index in A[node_54] if index not in remove_list]

        finish_nodes.add(node_index)
        for neighbour in neighbour_indexs:
            # 当前注入节点的邻居节点，不再注入
            finish_nodes.add(neighbour)

        # 1. 选择模态
        m = np.random.randint(0, mode_num - 1)
        # 2. 选择注入窗口
        scale = random.uniform(*windows_scale)
        wl = int(scale * sequences_len)  # 窗口大小
        ws = np.random.randint(0, sequences_len - wl + 1)  # 起始位置

        # 4. 映射
        # points = [reflect_points(xi, yi, x[0], v_i[0], x[-1], v_i[-1]) for xi, yi in zip(x, v_i)]
        middle = (max(data[node_index, m, ws:ws + wl]) + min(data[node_index, m, ws:ws + wl])) / 2
        data[node_index, m, ws:ws + wl] = 2 * middle - data[node_index, m, ws:ws + wl]

        # 打标签
        labels[node_index, m, ws:ws + wl] = 5

    return data, labels


def injectTemporalCorrelationAnomaly(data, num=30, windows_scale=(0.2, 0.6)):
    '''
    @description: 用于注入时序相关性异常
    @param {Any} data : 要注入异常的数据片段，shape:[node,model,time]
    @param {Any} start : 本段数据相对整体数据的位置
    @param {Any} file_name : 记录注入异常的文件
    @param {Any} num : 注入异常的节点个数
    @param {Any} windows_scale : 异常时序窗口大小比例（最小，最大）
    @return {Any} data_win, label_win : 异常数据，标签
    '''
    nod_num, mode_num, sequences_len = data.shape
    # 异常最小、最大窗长
    len_min = int(sequences_len * windows_scale[0])
    len_max = int(sequences_len * windows_scale[1])

    # 窗口长度数据
    data_win = data
    # 数据标签
    label_win = np.zeros(data.shape)
    # 注入异常的节点
    node_num = random.sample(range(nod_num), num)
    # 注入异常的模态
    model_num = random.choices(range(mode_num), k=num)
    # 注入异常的长度
    time_len = random.choices(range(len_min, len_max), k=num)
    # 注入异常的开始时刻
    time_point = []
    for tlen in time_len:
        time_point.append(random.randint(0, sequences_len - 1 - tlen))

    for n, m, t, l in zip(node_num, model_num, time_point, time_len):
        # 异常标签
        label_win[n, m, t:t + l] = 4
        node_54 = node51254(n + 1)
        # print("注入节点编号:%d\n模态:%d\n窗口位置:%d\n窗口大小:%d" % (node_54, m, start + t, l))
        # 计算注入异常前的相关系数
        r = np.corrcoef(data_win[n, :, t:t + l], rowvar=True)
        # print("注入前相关系数：%0.3f %0.3f %0.3f" % (r[0, 1], r[0, 2], r[1, 2]))
        # 注入异常
        flip = data_win[n, m, t:t + l]
        data_win[n, m, t:t + l] = flip - (flip - data_win[n, m, t]) * 2
        # 注入异常后的相关系数
        r = np.corrcoef(data_win[n, :, t:t + l], rowvar=True)
        print(r)
        # print("注入后相关系数：%0.3f %0.3f %0.3f" % (r[0, 1], r[0, 2], r[1, 2]))
        # # 文件记录注入异常
        # if file_name:
        #     import os
        #     if not os.path.exists(file_name):
        #         # 如果文件不存在，则创建文件
        #         with open(file_name, 'w') as file:
        #             # 可以选择写入一些初始内容
        #             file.write('文件用于记录注入异常\n')
        #     with open(file_name, 'a') as f:
        #         f.write("注入节点编号:%d\n模态:%d\n窗口位置:%d\n窗口大小:%d\n" % (node_54, m, start + t, l))

    return data_win, label_win


def injectTraditionAnomaly(data, num=30,
                           windows_scale=(0.1, 0.2),
                           scale_choice=0.3, contextual_choice=0.06, w=10):
    '''
    @description: 用于注入点异常、上下文异常、集体异常,每种异常占1/3
    @param {Any} data : 要注入异常的数据片段，shape:[node,model,time]
    @param {Any} start : 本段数据相对整体数据的位置
    @param {Any} file_name : 记录注入异常的文件
    @param {Any} num : 注入异常的节点个数
    @param {Any} windows_scale : 异常时序窗口大小比例（最小，最大）
    @param {Any} scale_choice : 点异常放大比例
    @param {Any} contextual_choice : 上下文异常偏移的比例
    @param {Any} collective_choice : 集体异常偏移值
    @param {Any} w : 集体异常中
    @param {Any} a :
    @return {Any} data_win, label_win : 异常数据，标签
    '''
    nod_num, mode_num, sequences_len = data.shape
    # 异常最小、最大窗长
    len_min = int(sequences_len * windows_scale[0])
    len_max = int(sequences_len * windows_scale[1])

    # 窗口长度数据
    data_win = data
    # 数据标签
    label_win = np.zeros(data.shape)
    # 注入异常的节点
    node_num = random.sample(range(nod_num), num)
    # 注入异常的模态
    model_num = random.choices(range(mode_num), k=num)
    # 注入异常的长度
    time_len = random.choices(range(len_min, len_max), k=num)
    # 注入异常的开始时刻
    time_point = []
    for tlen in time_len:
        time_point.append(random.randint(0, sequences_len - 1 - tlen))

    # 记录第几个异常
    index = 0
    for n, m, t, l in zip(node_num, model_num, time_point, time_len):
        # 注入点异常
        if index < num * (1 / 3):
            # 随机选择l个点
            inject_point = random.sample(range(sequences_len), l)
            # 注入异常公式
            if index < num * (1 / 5):
                data_win[n, m, inject_point] = data_win[n, m, inject_point] * (1 - scale_choice)
            else:
                data_win[n, m, inject_point] = data_win[n, m, inject_point] * (1 + scale_choice)
            label_win[n, m, inject_point] = 1
        # 注入上下文异常
        elif num * (1 / 3) < index and index < num * (2 / 3):
        # else:
            max_num = max(data[n, m, :])
            min_num = min(data[n, m, :])
            # 注入异常公式
            if index < num * (1 / 2):
                data_win[n, m, t:t + l] = data_win[n, m, t:t + l] * (1 - scale_choice)#- ((max_num - min_num) * contextual_choice)
            else:
                data_win[n, m, t:t + l] = data_win[n, m, t:t + l] * (1 + scale_choice)#+ ((max_num - min_num) * contextual_choice)
            # data_win[n, m, t:t + l] = data_win[n, m, t:t + l]* (1 - scale_choice)# + ((max_num - min_num) * contextual_choice)
            label_win[n, m, t:t + l] = 2
        # 注入集体异常
        else:
            # # 注入异常公式
            # max_num = max(data[n, m, :])
            # min_num = min(data[n, m, :])
            avge=np.mean(data_win[n, m, t:t + l],axis=-1)

            # collective_choice = 0.5 * (max_num + min_num)
            # a=max_num/100
            x = np.linspace(0, w * np.pi, l)  # 在0到2*pi之间生成n个点
            if index < num * (4 / 5):
                data_win[n, m, t:t + l] = (avge/10) * np.sin(x) + avge * (1 - scale_choice)
            else:
                data_win[n, m, t:t + l] = (avge/10) * np.sin(x) + avge * (1 + scale_choice)
            label_win[n, m, t:t + l] = 3

        index += 1

    return data_win, label_win


class sensor_data(Dataset):
    def __init__(self, file_path, time_win=300, stride=300, mode='train'):
        self.mode=mode
        IBRL_dataset = np.load(file_path+"/IBRL_data_anomaly.npz")
        data=np.load(file_path+"/IBRL.npy")
        data_abnorm=IBRL_dataset["data_ab"]
        # if mode=='train':
        #     data=np.load(file_path+"/IBRL.npy")
        # else:
        #     data=IBRL_dataset["data_ab"]
        label=IBRL_dataset["data_la"]
        self.time_win=time_win
        IBRL_len=(data.shape[-1]-time_win)/stride
        assert IBRL_len % 1 == 0, 'Cannot divide data exactly'
        # print(dataset_len)
        self.data_win=[]
        self.label_win=[]
        self.abnorm_win=[]
        for start in range(0,data.shape[-1]-time_win+stride,stride):
            self.abnorm_win.append(data_abnorm[:,:,start:start+time_win])
            self.data_win.append(data[:,:,start:start+time_win])
            self.label_win.append(label[:,:,start:start+time_win])
        if mode=='train':
            self.IBRL_abnorm=self.abnorm_win[:int(0.8*len(self.data_win))]
            self.IBRL_data=self.data_win[:int(0.8*len(self.data_win))]
            self.IBRL_label=self.label_win[:int(0.8*len(self.label_win))]
            print("train")
        else:
            self.IBRL_abnorm=self.abnorm_win[int(0.8*len(self.data_win)):]
            self.IBRL_data=self.data_win[int(0.8*len(self.data_win)):]
            self.IBRL_label=self.label_win[int(0.8*len(self.label_win)):]
            print("test")
        self.data_len=len(self.IBRL_data)
        print(self.data_len)
        print(IBRL_len)

    def __getitem__(self, index):
        win_data =self.IBRL_data[index]
        win_abnorm = self.IBRL_abnorm[index]
        # win_abnorm = copy.deepcopy(win_data)
        win_label =self.IBRL_label[index]
        data_out = win_abnorm
        label_out=win_label

        return win_data, data_out, np.where(label_out>=1,1,0)#np.where(np.sum(label_out,axis=1)>=1,1,0)

    def __len__(self):
        return self.data_len


class adj_data():
    def __init__(self, file_path):
        super(adj_data, self).__init__()
        self.path=file_path

    def adj_distance(self):
        # 读取节点坐标文件
        node_coords = {}
        with open(self.path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                if node_id in [5, 11, 15]:
                    continue
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)

        # 获取所有节点ID并按顺序排列
        nodes = sorted(node_coords.keys())
        n = len(nodes)

        # 初始化邻接矩阵
        adj_matrix = np.array([[0.0 for _ in range(n)] for _ in range(n)])

        # 计算欧氏距离填充矩阵
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj_matrix[i][j] = 0.0
                else:
                    x1, y1 = node_coords[nodes[i]]
                    x2, y2 = node_coords[nodes[j]]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    adj_matrix[i][j] = distance
        return node_coords, adj_matrix
    
    def adj_matrix(self,threshold):
        _, adj_matrix=self.adj_distance()
        adj_matrix = np.where(adj_matrix<=threshold,1,0)
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix
    
    def adj_show(self,threshold):
        node_coords, adj_matrix = self.adj_distance()
        adj_matrix = np.where(adj_matrix<=threshold,1,0)
        np.fill_diagonal(adj_matrix, 0)  # 对角线置零
        G = nx.Graph()

        # 添加节点 (使用实际坐标)
        for node, (x, y) in node_coords.items():
            G.add_node(node-1, pos=(x, y))  # 保存坐标信息

        # 添加边 (根据邻接矩阵)
        edges = np.argwhere(adj_matrix == 1)
        G.add_edges_from(edges)  # 自动忽略重复边

        # plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()

        # 获取节点坐标布局
        pos = nx.get_node_attributes(G, 'pos')

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500,
            node_color='skyblue',
            edgecolors='black',
            linewidths=1)

        # 绘制边
        nx.draw_networkx_edges(G, pos, width=1.5,
            edge_color='gray',
            alpha=0.7)

        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=12,
            font_family='sans-serif',
            font_color='black')

        plt.title("Graph Visualization with Coordinates", fontsize=14)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()

        # 旋转180度：同时翻转x轴和y轴
        ax.invert_xaxis()  # 翻转x轴方向（从右到左）
        ax.invert_yaxis()  # 翻转y轴方向（从上到下）
        ax.legend()
        # plt.xticks(rotation=45)  # x轴标签旋转45度
        # plt.yticks(rotation=90)  # y轴标签旋转90度
        plt.show()


    def normalize_adjacency(self, threshold, normalization_type='symmetric', add_self_loop=True):
        """
        邻接矩阵归一化
        :param adj_matrix: 邻接矩阵（np.array 或 scipy.sparse矩阵）
        :param normalization_type: 'symmetric' 或 'random_walk'
        :param add_self_loop: 是否添加自环
        :return: 归一化后的邻接矩阵
        """
        adj_matrix=self.adj_matrix(threshold)
        if add_self_loop:
            adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])  # 添加自环
        
        # 计算度数矩阵
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # 处理零度数（避免除以零）
        degrees[degrees == 0] = 1e-12  # 添加微小值
        
        if normalization_type == 'symmetric':
            # 对称归一化 D^{-1/2} A D^{-1/2}
            degree_matrix_inv_sqrt = diags(1.0 / np.sqrt(degrees))
            normalized_adj = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
        elif normalization_type == 'random_walk':
            # 随机游走归一化 D^{-1} A
            degree_matrix_inv = diags(1.0 / degrees)
            normalized_adj = degree_matrix_inv @ adj_matrix
        else:
            raise ValueError("Unsupported normalization type. Choose 'symmetric' or 'random_walk'.")
        
        return normalized_adj


# # 输出邻接矩阵（示例输出前5行）
# print("邻接矩阵示例（前5行）：")
# for row in adj_matrix[:5]:
#     print(' '.join(f"{val:.2f}" for val in row[:5]))

# # 保存完整矩阵到文件
# with open('adjacency_matrix.txt', 'w') as f:
#     for row in adj_matrix:
#         f.write(' '.join(f"{val:.4f}" for val in row) + '\n')

if __name__=='__main__':
    file_path="./data/dataset"

    # data=np.load(file_path+"/IBRL.npy")
    # print(data.shape)
    # plt.figure()
    # # plt.ylim(10,35)
    # plt.subplots_adjust(left=0.04, right=0.99, bottom=0.04, top=0.99)
    # plt.axvline(x=0, color='r', linestyle='--')
    # plt.axvline(x=1000, color='r', linestyle='--')
    # plt.text(200, 2.64, 'zoom in', fontsize=13, color='r')
    # plt.plot(data[0,2,:].T)
    
    # plt.show()
    # plt.close()


    adj=adj_data("./data/dataset/IBRL/node.txt")
    # print(adj.adj_distance())
    # print(adj.adj_matrix(10))
    # adj.adj_show(8)
    print(adj.normalize_adjacency(8).shape)



