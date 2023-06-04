import torch
from model.models import MGAT
from dataloader import get_data,get_TopK_adj
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
# torch.set_printoptions(profile="full")

train_rate = 0.8
vali_rate = 0.1
epoch = 200
learn_rate = 0.00005 #0.00003
threshold = 2.892448663711548

data_maxt = get_data()
print(data_maxt.shape)

def normalize(input_maxt,mask_Maxt):
    register = np.zeros((input_maxt.shape[0],input_maxt.shape[1],input_maxt.shape[2]))
    for maxt_num in range(mask_Maxt.shape[0]):
        # print(mask_Maxt[maxt_num].shape)
        meanvalue = np.mean(mask_Maxt[maxt_num],axis=1)
        stdvalue = np.std(mask_Maxt[maxt_num],axis=1)
        # print(meanvalue,stdvalue)
        register[maxt_num] = (input_maxt[maxt_num] - np.tile(np.array(meanvalue),[input_maxt[maxt_num].shape[1],1]).T) /\
           np.tile(np.array(stdvalue),[input_maxt[maxt_num].shape[1],1]).T
    return register


train_set = data_maxt[:,:,0:int(data_maxt.shape[2] * train_rate)]
print("train set",train_set.shape)
vali_set = data_maxt[:,:,int(data_maxt.shape[2] * train_rate):int(data_maxt.shape[2] * (train_rate + vali_rate))]
print("vali set",vali_set.shape)
orin_test_set = data_maxt[:,:,int(data_maxt.shape[2] * (train_rate + vali_rate)):int(data_maxt.shape[2])]
print("test set",orin_test_set.shape)

max_res = [float('-inf') for xp in range(train_set.shape[1])]
min_res = [float('inf') for xp in range(train_set.shape[1])]
for read_num in range(train_set.shape[0]):
    this_maxt = train_set[read_num]
    max_list = np.max(this_maxt,axis=1)
    for maxcount in range(len(max_list)):
        if max_list[maxcount] > max_res[maxcount]:

            max_res[maxcount] = max_list[maxcount]
    min_list = np.min(this_maxt,axis=1)
    for mincount in range(len(min_list)):
        if min_list[mincount] < min_res[mincount]:

            min_res[mincount] = min_list[mincount]
print("the max observation %s of modal %s of the training set" %(data_maxt.shape[1],max_res))
print("the min observation %s of modal %s of the training set" %(data_maxt.shape[1],min_res))

def test_inj(test_set,clabel):

    #!!!slow increase!!!#
    if clabel == 4:
        test_set[35][1][150] = test_set[35][1][149] + (max_res[1]-min_res[1])/8
        test_set[35][1][151] = test_set[35][1][150] + (max_res[1]-min_res[1])/8
        test_set[35][1][152] = test_set[35][1][151] + (max_res[1]-min_res[1])/8
        test_set[35][1][153] = test_set[35][1][152] + (max_res[1]-min_res[1])/8
        test_set[35][1][154] = test_set[35][1][153] + (max_res[1]-min_res[1])/8
        test_set[35][1][155] = test_set[35][1][154] + (max_res[1]-min_res[1])/8
    #!!!fast increase!!!#
    if clabel == 5:
        test_set[23][0][150] = test_set[23][0][149] + (max_res[1]-min_res[1])/5
        test_set[23][0][151] = test_set[23][0][150] + (max_res[1]-min_res[1])/5
        test_set[23][0][152] = test_set[23][0][151] + (max_res[1]-min_res[1])/5
        test_set[23][0][153] = test_set[23][0][152] + (max_res[1]-min_res[1])/5
    #!!!surge!!!#
    if clabel == 3:
    # print(test_set[35][0][100])
        test_set[35][0][100] = test_set[35][0][100] + (max_res[0]-min_res[0])
    #!!!observation turns zero!!!#
    if clabel == 2:
    # print(test_set[29][2][70])
        test_set[29][2][70] = 0
    else:
        pass

    test_set = normalize(test_set.copy(),train_set.copy())
    return test_set

def mse_loss(pred,label):
    return torch.sum((pred.squeeze(0) - label.squeeze(0)) ** 2)
def rec_loss(pred,label):
    # print(label.reshape(50 * slide_win).shape)
    nodescore_res = []
    for i in range(pred.shape[0]):
        nodescore_res.append(torch.sum(torch.abs(pred[i] - label[i])))
    # print(len(nodescore_res))
    return max(nodescore_res)

node_num = 50
measure_dim = 3
slide_win = 60


mgat = MGAT(node_num,measure_dim,slide_win,0,0.05)
mgat.load_state_dict(torch.load('save/3_dim.pt'))
F_adj_sub = torch.ones(50,3,3)
T_adj_sub = torch.ones(50,60,60)
adj = torch.ones(50,50)
# out , H_ = mgat(torch.rand(50,5,40),torch.ones(50,5,5),torch.ones(50,50),(h_1,h_2))
# print(out.shape,H_[0].shape,H_[1].shape)

while True:
    clabel = input("""
    input test type number
    1，Running tests on a test set without anomaly injection
    2，Injecting zero turn anomalies into the test set for testing
    3，Injecting surge anomalies into the test set for testing
    4，Injecting slow change anomalies into the test set for testing
    5，Injecting fast change anomalies into the test set for testing
    """)
    test_set = test_inj(orin_test_set.copy(),int(clabel))

    with torch.no_grad():

        scorelist = []
        h_1 = torch.zeros(2, 50, 32)
        lossvalue = 0
        losscount = 1
        losslist = []
        for timecount in range(test_set.shape[2] - slide_win - 1 ):
            # print(timecount,timecount + slide_win,timecount + slide_win + 1)
            now_tensor = torch.from_numpy(test_set[:,:,timecount:timecount + slide_win]).float()
            label = torch.from_numpy(test_set[:,:, timecount + slide_win + 1]).float()

            # print(now_tensor.shape,label.shape)


            out , H_ = mgat(now_tensor,F_adj_sub,T_adj_sub,adj,h_1)
            # print(H_.shape)
            score =  rec_loss(out,label)
            # print(out.shape)
            h_1 = H_.detach()

            scorelist.append(float(score.detach()))
            # if losscount % 10 == 0:
            #     h_1 = torch.zeros(2, 50, 1)
            #     h_2 = torch.zeros(2, 50, 1)
            losscount = losscount + 1
        print(scorelist)

    plt.title("testset anormaly score plot",fontsize = 20)
    plt.xlabel("time_series",fontsize = 20)
    plt.ylabel("",fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.plot([ (i+slide_win+1) for i in range(len(scorelist))],scorelist,color = "black")
    plt.plot([ (i+slide_win+1) for i in range(len(scorelist))],[ threshold for i in range(len(scorelist))],)
    plt.show()