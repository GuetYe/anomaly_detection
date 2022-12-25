import torch
import numpy as np

def combine_graph(attr_maxt, adj_maxt, wei_maxt, batch_copy):
    if(batch_copy == True):
        adj_maxt = adj_maxt.unsqueeze(0).expand(attr_maxt.shape[0], adj_maxt.shape[0],  adj_maxt.shape[1])
        wei_maxt = wei_maxt.unsqueeze(0).expand(attr_maxt.shape[0], wei_maxt.shape[0])
    for modal_num in range(attr_maxt.shape[1]):
        if modal_num == 0:
            for batch_num in range(attr_maxt.shape[0]):
                if batch_num == 0:
                    bc = attr_maxt[batch_num][modal_num]
                else:
                    bc = torch.cat([bc,attr_maxt[batch_num][modal_num]],dim=0)
            attr_combine = bc
        else:
            for batch_num in range(attr_maxt.shape[0]):
                if batch_num == 0:
                    bc = attr_maxt[batch_num][modal_num]
                else:
                    bc = torch.cat([bc,attr_maxt[batch_num][modal_num]],dim=0)
            attr_combine = torch.cat([attr_combine,bc],dim=0)
    for batch_num in range(attr_maxt.shape[0]):
        if batch_num == 0:
            adj_combine = adj_maxt[batch_num]
        else:
            adj_combine = torch.cat([adj_combine, adj_maxt[batch_num] + batch_num * attr_maxt.shape[2]], dim=1)

    for batch_num in range(attr_maxt.shape[0]):
        if batch_num == 0:
            wei_combine = wei_maxt[batch_num]
        else:
            wei_combine = torch.cat([wei_combine, wei_maxt[batch_num]], dim=0)
    batchnum = attr_maxt.shape[0]
    return attr_combine.float(), adj_combine.long(), wei_combine.float(), batchnum

# gp = torch.rand(3,3,5,5)
# # print(gp)
# adj = torch.tensor([[[0,0,0],[1,1,1]],
#                     [[1,1,1],[2,2,2]],
#                     [[2,2,2],[2,2,2]]])
# wei = torch.tensor([[3,3,3],
#                     [2,2,2],
#                     [1,1,1]])
# print(gp.shape,adj.shape,wei.shape)
# mc,ac,wc, batch_num = combine_graph(gp,adj,wei,False)
# print(mc.shape,ac.shape,wc.shape,batch_num)
# print(ac)

def normalize(input_maxt,mask_Maxt):
    register = np.zeros((input_maxt.shape[0],input_maxt.shape[1],input_maxt.shape[2]))
    meanvalue = np.mean(mask_Maxt,axis=2)
    stdvalue = np.std(mask_Maxt,axis=2)
    register = (input_maxt - np.tile(np.expand_dims(meanvalue,2),[1,1,input_maxt.shape[2]])) /\
               np.tile(np.expand_dims(stdvalue,2),[1,1,input_maxt.shape[2]])
    return register

def get_adj_wei():
    position_file = open("rawData/IBRL/node.txt", "r")
    fileDate = position_file.read()
    nodeMSG = fileDate.split("\n")
    if "" in nodeMSG:
        nodeMSG.remove("")
    check_file = open("processed/IBRL_count.txt", "r")
    considered_node = list(map(int,eval(check_file.readline().replace("\n",""))))
    considered_node.remove(18)
    nodeArray = np.ones((len(considered_node),2))
    ct = 0
    for num_index in considered_node:
        split_msg = nodeMSG[num_index - 1].strip().split(" ")
        nodeArray[ct][0] = float(split_msg[1])
        nodeArray[ct][1] = float(split_msg[2])
        ct = ct + 1
    wei_res = []
    adj_res = []
    for i in range(nodeArray.shape[0]):
        for j in range(nodeArray.shape[0]):
            distance = ((nodeArray[i][0] - nodeArray[j][0]) ** 2 + (nodeArray[i][1] - nodeArray[j][1]) ** 2) ** (1/2)
            # print(i, j ,distance)
            wei_res.append(np.array(distance))
            adj_res.append(np.array([i, j]))
    wei_res = np.array(wei_res)
    wei_res = torch.tensor(wei_res / np.max(wei_res))
    adj = torch.tensor(np.array(adj_res).T)
    return adj, wei_res

# adj, wei = get_adj_wei()
# print(adj.shape,wei.shape)
# print(wei)

def get_adj_maxt():
    position_file = open("rawData/IBRL/node.txt", "r")
    fileDate = position_file.read()
    nodeMSG = fileDate.split("\n")
    # print(nodeMSG)
    if "" in nodeMSG:
        nodeMSG.remove("")
    check_file = open("processed/IBRL_count.txt", "r")
    considered_node = list(map(int,eval(check_file.readline().replace("\n",""))))
    considered_node.remove(18)
    nodeArray = np.ones((len(considered_node),2))
    ct = 0
    for num_index in considered_node:
        split_msg = nodeMSG[num_index - 1].strip().split(" ")
        nodeArray[ct][0] = float(split_msg[1])
        nodeArray[ct][1] = float(split_msg[2])
        ct = ct + 1
    adj_res = np.zeros((nodeArray.shape[0],nodeArray.shape[0]))
    for i in range(nodeArray.shape[0]):
        for j in range(nodeArray.shape[0]):
            distance = ((nodeArray[i][0] - nodeArray[j][0]) ** 2 + (nodeArray[i][1] - nodeArray[j][1]) ** 2) ** (1/2)
            # print(i, j ,distance)
            adj_res[i][j] = distance
    adj_res = adj_res / np.max(adj_res)
    position_file.close()
    check_file.close()
    return adj_res
# adj_maxt = get_adj_maxt()
# print(adj_maxt)


def trendline(data):
    order=1
    index=[i for i in range(1,len(data)+1)]
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)
