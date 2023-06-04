import  numpy as np
import torch,os,random,time
from model.new import GNN
#from read_CIMIS import get_CIMIS_data
from util.utils import combine_graph,normalize,get_adj_wei,get_adj_maxt
from Config import parse_args,get_data_inf,get_IBRL_data,get_block_data
from util.inject_anomaly import *

config = parse_args()
use_cuda = config.gpu
device = "cpu"
if use_cuda:
    device = "cuda"

# print(test_data.shape)
# print(H_.shape,test_data.shape)
max_min_res = torch.load("processed/" + config.dataset + "/max_min.pt")
print(max_min_res)


if config.use_kmeans_paa == False:
    orin_data = get_IBRL_data()
elif config.use_kmeans_paa == True:
    orin_data = get_block_data()

node_num, mode_num = get_data_inf(config.dataset)
adj,weight = get_adj_wei()
adj_maxt = get_adj_maxt()

train_data = orin_data[:,:,0:int(orin_data.shape[2]*config.train_rate)].copy()
test_data = orin_data[:,:,int(orin_data.shape[2]*config.train_rate) - config.window_size + 1: orin_data.shape[2]].copy()

normal_check_points = np.int16(np.linspace(0 + config.margin + config.window_size,
                                           test_data.shape[2] - config.margin - config.window_size,
                                           config.test_time))
# print(normal_check_points)
anomaly_check_points = normal_check_points.copy()
np.random.seed(1)

TP = 0
FN = 0

TN = 0
FP = 0

model = GNN(mode_num,node_num, config.window_size,config.base_gnn).to(device)
model.load_state_dict(torch.load("save/para_" + config.dataset + ".pth"))
H_ = torch.load("save/save_H_" + config.dataset + ".pth").to(device)
H_ = H_[:,H_.shape[1] - 1:H_.shape[1],:].contiguous()
model.eval()
test_normal = normalize(test_data,train_data)
with torch.no_grad():
    for tc in range(test_normal.shape[2] - config.window_size):
        # print(torch.tensor(test_data[:,:,i:i+20]).unsqueeze(0).shape)
        batchcat, adjcat, weicat, batchnum = combine_graph(
            torch.tensor(test_normal[:, :, tc:tc + config.window_size]).unsqueeze(0), adj, weight)
        # print(batchcat.shape,adjcat.shape,weicat.shape,batchnum)
        out, rec, new_H = model(batchcat.to(device), adjcat.to(device), weicat.to(device), H_)
        # print(out)
        new_H = new_H.detach()
        H_ = new_H
        if tc in normal_check_points:
            # print(i)
            if torch.argmax(out[0]).item() == 0:
                TN = TN + 1
            else:
                FP = FP + 1
    print("TN",TN,"FP", FP)

    for i in range(len(anomaly_check_points)):
        print("inject point",anomaly_check_points[i])
        H_ = torch.load("save/save_H_" + config.dataset + ".pth").to(device)
        H_ = H_[:, H_.shape[1] - 1:H_.shape[1], :].contiguous()
        test_anomaly = test_data.copy()
        inject_node = np.random.randint(node_num)
        inject_modal = np.random.randint(mode_num)
        num = config.inject_length
        start_t = anomaly_check_points[i]
        inj_type = intermodal_anomaly(mode_num, test_anomaly, inject_modal, inject_node, start_t - num, start_t + config.window_size)
        if inj_type == False:
            # print("dsadsad")
            inj_type = internode_anomaly(adj_maxt.copy(),  test_anomaly, inject_modal, inject_node, start_t - num,
                                         start_t + config.window_size)
            # print("inj_type",inj_type)
            if inj_type == False:
                inj_type = np.random.randint(0, 3)
        # inject_type = np.random.randint(3)
        print("inject_node",inject_node,"inject_modal",inject_modal)
        if inj_type == 0:
            test_anomaly = scale(test_anomaly, inject_modal, inject_node,
                                 anomaly_check_points[i], anomaly_check_points[i] +config.inject_length)
        elif inj_type == 1:
            test_anomaly = mirro(test_anomaly, inject_modal, inject_node,
                                 anomaly_check_points[i], anomaly_check_points[i] +config.inject_length)
        elif inj_type == 2:
            test_anomaly = surge(test_anomaly, inject_modal, inject_node,
                                 anomaly_check_points[i], anomaly_check_points[i] +config.inject_length, max_min_res)
        elif inj_type == 3:
            # print(am_sd[inject_mode[group_num][i]][inject_node[group_num][i]][inject_point[group_num][i]:inject_point[group_num][i]+num])
            # am_sd = scale(am_sd,inject_mode[i],inject_node[i],inject_point[i],inject_point[i]+num)
            test_anomaly = decay(test_anomaly,inject_modal, inject_node,
                                 anomaly_check_points[i], anomaly_check_points[i] +config.inject_length, max_min_res)
        elif inj_type == 4:
            # print(am_sd[inject_mode[group_num][i]][inject_node[group_num][i]][inject_point[group_num][i]:inject_point[group_num][i]+num])
            # am_sd = scale(am_sd,inject_mode[i],inject_node[i],inject_point[i],inject_point[i]+num)
            test_anomaly = increase(test_anomaly,inject_modal, inject_node,
                                 anomaly_check_points[i], anomaly_check_points[i] +config.inject_length, max_min_res)
        test_anomaly = normalize(test_anomaly, train_data)
        target = False
        # print(anomaly_check_points[i])
        for tc in range(test_anomaly.shape[2] - config.window_size):
            # print(tc+config.window_size)
            batchcat, adjcat, weicat, batchnum = combine_graph(
                torch.tensor(test_anomaly[:, :, tc:tc + config.window_size]).unsqueeze(0), adj, weight)
            # print(batchcat.shape,adjcat.shape,weicat.shape,batchnum)
            out, rec, new_H = model(batchcat.to(device), adjcat.to(device), weicat.to(device), H_)
            # print(out)
            new_H = new_H.detach()
            H_ = new_H
            if tc+config.window_size >= anomaly_check_points[i] and \
                    tc+config.window_size < anomaly_check_points[i] + config.window_size:
                # print("yes",tc)
                if torch.argmax(out[0]).item() == 1:
                    target = True
            elif tc+config.window_size > anomaly_check_points[i] + config.window_size or\
                    tc+config.window_size == test_anomaly.shape[2] - config.window_size-1:
                # print("exceed", tc)
                if target == False:
                    # print("x")
                    FN = FN + 1
                else:
                    # print("y")
                    TP = TP + 1
                break
    print("TP",TP,"FN",FN)

    test_anomaly = test_data.copy()
    batchcat, adjcat, weicat, batchnum = combine_graph(
        torch.tensor(test_anomaly[:, :, 0:0 + config.window_size]).unsqueeze(0), adj, weight)
    test_start = time.time()
    for tc in range(int(config.paa_out_dimension * 0.4)):
        out, rec, new_H = model(batchcat.to(device), adjcat.to(device), weicat.to(device), H_)
    test_end = time.time()
    speed = (test_end - test_start)
    print(speed)

Prec = TP / (TP + FP)
Rec  = TP / (TP + FN)
F1   = 2 * Prec * Rec / (Prec + Rec)

resultFile = open("save/result.txt","w")
resultFile.write("Number of points without anomaly injection: %s\n" %len(normal_check_points))
resultFile.write("Number of points with anomaly injection: %s\n" %len(anomaly_check_points))
resultFile.write("TP: %s\n" %TP)
resultFile.write("FN: %s\n" %FN)
resultFile.write("TN: %s\n" %TN)
resultFile.write("FP: %s\n" %FP)
resultFile.write("Prec: %s\n" %Prec)
resultFile.write("Rec: %s\n" %Rec)
resultFile.write("F1: %s\n" %F1)
resultFile.write("Speed: %s\n" %speed)