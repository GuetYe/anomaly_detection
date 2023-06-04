import numpy as np
import random
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from processed.IBRL_load import get_IBRL_data
from matplotlib import pyplot as  plt

def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, max_iter=10000):
    centers = {}  # 初始聚类中心
    # 初始化，随机选k个样本作为初始聚类中心。 random.sample(): 随机不重复抽取k个值
    n_data = data.shape[0]  # 样本个数
    np.random.sample()
    for idx, i in enumerate(np.random.choice(range(n_data), k, replace=False)):
        # idx取值范围[0, k-1]，代表第几个聚类中心;  data[i]为随机选取的样本作为聚类中心
        centers[idx] = data[i]

        # 开始迭代
    for i in range(max_iter):  # 迭代次数
        # print("开始第{}次迭代".format(i + 1))
        clusters = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
        for j in range(k):  # 初始化为空列表
            clusters[j] = []

        for sample in data:  # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
            for c in centers:  # 遍历每个聚类中心
                # 添加该样本点到聚类中心的距离
                distances.append(distance(sample, centers[c]))
            idx = np.argmin(distances)  # 最小距离的索引
            clusters[idx].append(sample)  # 将该样本添加到第idx个聚类中心

        pre_centers = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)

        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:
            # 如果新旧聚类中心不变，则迭代停止
            break
    return centers, clusters


def predict(p_data, centers):  # 预测新样本点所在的类
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)


def kmeans_paa(orindata,positionFile,cluster_num,paa_out_dimension,rate):
    node_num = orindata.shape[1]

    filepath = positionFile


    file = open(filepath,"r",encoding="utf8")
    txt = file.read().split("\n")
    data_res = []
    for linetxt in txt:
        linedata = linetxt.split(" ")
        data_res.append(list(map(float,linedata[1:3])))
    data_res = data_res[0:node_num]

    np.random.seed(5)
    x = np.array(data_res)

    # colors = ['r','b','y','m','c','g']
    centers, clusters = k_means(x, cluster_num)
    # for c in clusters:
    #     for point in clusters[c]:
    #         plt.scatter(point[0],point[1],c=colors[c])
    # plt.show()

    clusters_nums =[]
    for c in clusters:
        # print("#####")
        nums_res = []
        for point in clusters[c]:
            # print(data_res.index([point[0],point[1]]),[point[0],point[1]])
            nums_res.append(data_res.index([point[0],point[1]]))
        clusters_nums.append(nums_res)
    # print(clusters_nums)

    paa = PiecewiseAggregateApproximation(window_size=2)
    split_result = []
    for num_list in clusters_nums:
        # print(orindata[:,num_list,:].shape)
        clusters_data = np.zeros((orindata.shape[0],len(num_list),
                                  int(paa_out_dimension * rate)))
        count = 0
        for modal_data in orindata[:,num_list,:]:
            # print(modal_data.shape)
            node_count = 0
            for node_data in modal_data:

                node_data = np.expand_dims(node_data,0)
                # print(node_data.shape)
                # from matplotlib import pyplot as plt
                # ddd = node_data[0]
                # plt.plot([i for i in range(ddd.shape[0])], ddd)
                # plt.show()
                paa_data = paa.fit_transform(node_data)
                # from matplotlib import pyplot as plt
                # ddd = paa_data[0]
                # plt.plot([i for i in range(ddd.shape[0])], ddd)
                # plt.show()
                #
                # print(paa_data.shape)
                clusters_data[count][node_count] = paa_data
                node_count = node_count + 1
            count = count + 1
        print(clusters_data.shape)
        # from matplotlib import pyplot as plt
        # ddd = clusters_data[0][0]
        # plt.plot([i for i in range(ddd.shape[0])], ddd)
        # plt.show()
        split_result.append(clusters_data)

    return split_result

def get_splitblock_node(orindata,positionFile,cluster_num):
    node_num = orindata.shape[1]
    filepath = positionFile
    file = open(filepath, "r", encoding="utf8")
    txt = file.read().split("\n")
    data_res = []
    for linetxt in txt:
        linedata = linetxt.split(" ")
        data_res.append(list(map(float, linedata[1:3])))
    data_res = data_res[0:node_num]

    np.random.seed(5)
    x = np.array(data_res)

    centers, clusters = k_means(x, cluster_num)

    clusters_data = []
    for c in clusters:
        # print("#####")
        nums_res = []
        for point in clusters[c]:
            # print(data_res.index([point[0],point[1]]),[point[0],point[1]])
            nums_res.append(data_res.index([point[0], point[1]]))
        clusters_data.append(nums_res)
    # print(clusters_data)
    return  clusters_data