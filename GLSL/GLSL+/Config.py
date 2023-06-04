import argparse   #步骤一
from processed.IBRL_load import get_IBRL_data
from kmeans_paa import kmeans_paa
import numpy as np

def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser(description="you should add those parameter")        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    parser.add_argument('--gpu', default=True, help="if you want to use cuda (True)")
    parser.add_argument('--dataset', default="IBRL", help="dataset name")
    parser.add_argument('--window_size', default=20, help="slid window length")
    parser.add_argument('--batch_size', default=5, help="training batch size")
    parser.add_argument('--test_time', default=200, help="number of test times")
    parser.add_argument('--margin', default=20, help="the margin of test check point")
    parser.add_argument('--inject_length', default=5, help="inject_length")
    parser.add_argument('--learning_rate', default=0.0005, help="learning_rate")
    parser.add_argument('--train_rate', default=0.6, help="train_rate")
    parser.add_argument('--epoch', default=100, help="train_rate")
    parser.add_argument('--base_gnn', default="GAT", help="base_gnn")

    parser.add_argument('--use_kmeans_paa', default=True, help="use kmeans and paa")
    parser.add_argument('--cluster_num', default=3, help="cluster_num")
    parser.add_argument('--use_cluster_block', default=0, help="cluster_num")
    parser.add_argument('--paa_out_dimension', default=2500, help="cluster_num")
    args = parser.parse_args()
    return args

config = parse_args()
IBRL_data = get_IBRL_data()
if config.use_kmeans_paa == True:
    print(IBRL_data.shape)
    split_res = kmeans_paa(IBRL_data, "rawData/IBRL/node.txt", config.cluster_num,
                           config.paa_out_dimension, 1)
    use_block = split_res[config.use_cluster_block]

def get_data_inf(datasetName):
    config = parse_args()
    if config.use_kmeans_paa == False:
        node_num = IBRL_data.shape[1]
        mode_num = IBRL_data.shape[0]
    elif config.use_kmeans_paa == True:
        node_num = use_block.shape[1]
        mode_num = use_block.shape[0]
    return node_num, mode_num

def get_inject_inf(datasetName):
    np.random.seed(5)
    config = parse_args()
    if config.use_kmeans_paa == False:
        point_num = int(IBRL_data.shape[2] * config.train_rate / 100)-1
        # print(point_num)
        IBRL_inject_point = [[x*100+50+np.random.randint(low=-20,high=20)
                              for x in range(point_num)] for i in range(4)]
        # print(IBRL_inject_point)
        IBRL_inject_mode = [[np.random.randint(low=0,high=IBRL_data.shape[0])
                              for x in range(point_num)] for i in range(4)]
        # print(IBRL_inject_mode)
        IBRL_inject_node = [[np.random.randint(low=0,high=IBRL_data.shape[1])
                              for x in range(point_num)] for i in range(4)]
        # print(IBRL_inject_node)
    elif config.use_kmeans_paa == True:
        point_num = int(use_block.shape[2] * config.train_rate / 100)-1
        # print(point_num)
        IBRL_inject_point = [[x*100+50+np.random.randint(low=-20,high=20)
                              for x in range(point_num)] for i in range(4)]
        # print(IBRL_inject_point)
        IBRL_inject_mode = [[np.random.randint(low=0,high=use_block.shape[0])
                              for x in range(point_num)] for i in range(4)]
        # print(IBRL_inject_mode)
        IBRL_inject_node = [[np.random.randint(low=0,high=use_block.shape[1])
                              for x in range(point_num)] for i in range(4)]

    return IBRL_inject_point,IBRL_inject_mode,IBRL_inject_node

def get_block_data():
    return use_block

def get_IBRL_data():
    return IBRL_data
