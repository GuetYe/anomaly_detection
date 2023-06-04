import torch
from model.models import MGAT
from dataloader import get_data,get_TopK_adj
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
# torch.set_printoptions(profile="full")

####----基础参数----####
train_rate = 0.8
vali_rate = 0.1
learn_rate = 0.00005 #0.00003
grusize = 32

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
print("训练集",train_set.shape)
vali_set = data_maxt[:,:,int(data_maxt.shape[2] * train_rate):int(data_maxt.shape[2] * (train_rate + vali_rate))]
print("验证集",vali_set.shape)
test_set = data_maxt[:,:,int(data_maxt.shape[2] * (train_rate + vali_rate)):int(data_maxt.shape[2])]
print("测试集",test_set.shape)

# sd = np.array([[[1,2,3,4],[11,12,13,14]],[[5,6,6,4],[34,23,55,33]],[[1,2,3,4],[11,12,13,14]]])
# print(sd.shape)
# print(sd)
# print(normalize(sd,sd))
vali_set = normalize(vali_set.copy(),train_set.copy())

def mse_loss(pred,label):
    return torch.sum((pred.squeeze(0) - label.squeeze(0)) ** 2)
def rec_loss(pred,label):
    # print(label.reshape(50 * slide_win).shape)
    nodescore_res = []
    for i in range(pred.shape[0]):
        nodescore_res.append(torch.sum(torch.abs(pred[i] - label[i])))
    # print(len(nodescore_res))
    return max(nodescore_res)

# ###---初始化lstm_GAT预测模型---###
node_num = 50
measure_dim = 3
slide_win = 60


mgat = MGAT(node_num,measure_dim,slide_win,0,0.05)
mgat.load_state_dict(torch.load('save/3_dim.pt'))
F_adj_sub = torch.ones(50,3,3)
T_adj_sub = torch.ones(50,slide_win,slide_win)
adj = torch.ones(50,50)
# out , H_ = mgat(torch.rand(50,5,40),torch.ones(50,5,5),torch.ones(50,50),(h_1,h_2))
# print(out.shape,H_[0].shape,H_[1].shape)


with torch.no_grad():

    scorelist = []
    h_1 = torch.zeros(2, 50, grusize)
    lossvalue = 0
    losscount = 1
    losslist = []
    for timecount in range(vali_set.shape[2] - slide_win - 1 ):
        # print(timecount,timecount + slide_win,timecount + slide_win + 1)
        now_tensor = torch.from_numpy(vali_set[:,:,timecount:timecount + slide_win]).float()
        label = torch.from_numpy(vali_set[:,:, timecount + slide_win + 1]).float()

        # print(now_tensor.shape,label.shape)


        out , H_ = mgat(now_tensor,F_adj_sub,T_adj_sub,adj,h_1)
        # print(H_.shape)
        score =  rec_loss(out,label)
        # print(out.shape)
        h_1 = H_.detach()

        scorelist.append(float(score.detach()))
        # if losscount % 20 == 0:
        #     h_1 = torch.zeros(2, 50, 16)
        #     h_2 = torch.zeros(2, 50, 1)
        losscount = losscount + 1
    print(scorelist)
plt.plot([ i for i in range(len(scorelist))],scorelist)
plt.show()
print(max(scorelist))