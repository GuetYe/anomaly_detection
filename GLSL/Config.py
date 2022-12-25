import argparse
from processed.IBRL_load import get_IBRL_data
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description="you should add those parameter")        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    parser.add_argument('--gpu', default=True, help="if you want to use cuda (True)")
    parser.add_argument('--dataset', default="IBRL", help="dataset name")
    parser.add_argument('--window_size', default=20, help="slid window length")
    parser.add_argument('--batch_size', default=5, help="training batch size")
    parser.add_argument('--test_time', default=100, help="number of test times")
    parser.add_argument('--margin', default=20, help="the margin of test check point")
    parser.add_argument('--inject_length', default=5, help="inject_length")
    parser.add_argument('--learning_rate', default=0.0005, help="learning_rate")
    parser.add_argument('--train_rate', default=0.6, help="train_rate")
    parser.add_argument('--epoch', default=100, help="train_rate")
    parser.add_argument('--base_gnn', default="GAT", help="base_gnn")
    parser.add_argument('--train_inj_deviation', default=25, help="Training anomaly injection deviation")
    parser.add_argument('--test_inj_deviation', default=40, help="Testing anomaly injection deviation")

    args = parser.parse_args()
    return args

def get_data_inf(datasetName):
    if datasetName == "IBRL":
        IBRL_data = get_IBRL_data()
        node_num = IBRL_data.shape[1]
        mode_num = IBRL_data.shape[0]
        return node_num, mode_num

def get_inject_inf(datasetName):
    np.random.seed(5)
    config = parse_args()
    if datasetName == "IBRL":
        IBRL_data = get_IBRL_data()
    point_num = int(IBRL_data.shape[2] * config.train_rate / 100)-1
    IBRL_inject_point = [[x*100+50+np.random.randint(low=-20,high=20)
                          for x in range(point_num)] for i in range(4)]
    IBRL_inject_mode = [[np.random.randint(low=0,high=IBRL_data.shape[0])
                          for x in range(point_num)] for i in range(4)]
    IBRL_inject_node = [[np.random.randint(low=0,high=IBRL_data.shape[1])
                          for x in range(point_num)] for i in range(4)]

    if datasetName == "IBRL":
        return IBRL_inject_point,IBRL_inject_mode,IBRL_inject_node

