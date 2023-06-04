import torch
import numpy as np
from Config import parse_args,get_IBRL_data
from kmeans_paa import get_splitblock_node

def combine_graph(attr_maxt,adj_maxt,wei_maxt):
    for modal_num in range(attr_maxt.shape[1]):
        if modal_num == 0:
            for batch_num in range(attr_maxt.shape[0]):
                if batch_num == 0:
                    bc = attr_maxt[batch_num][modal_num]
                else:
                    bc = torch.cat([bc,attr_maxt[batch_num][modal_num]],dim=0)
            mc = bc
        else:
            for batch_num in range(attr_maxt.shape[0]):
                if batch_num == 0:
                    bc = attr_maxt[batch_num][modal_num]
                else:
                    bc = torch.cat([bc,attr_maxt[batch_num][modal_num]],dim=0)
            mc = torch.cat([mc,bc],dim=0)
    for modenum in range(attr_maxt.shape[1]):
        for batch_num in range(attr_maxt.shape[0]):
            if batch_num == 0:
                ac = adj_maxt
            else:
                ac = torch.cat([ac, adj_maxt + batch_num * attr_maxt.shape[2]], dim=1)
        if modenum == 0:
            total_ac = ac.unsqueeze(0)
        else:
            total_ac = torch.cat([total_ac,ac.unsqueeze(0)],dim=0)
    for modenum in range(attr_maxt.shape[1]):
        for batch_num in range(attr_maxt.shape[0]):
            if batch_num == 0:
                wc = wei_maxt
            else:
                wc = torch.cat([wc,wei_maxt], dim=0)
        weight = wc.unsqueeze(0)
        if modenum == 0:
            total_wei = weight
        else:
            total_wei = torch.cat([total_wei,weight],dim=0)
    batchcat = mc
    adjcat = total_ac
    weicat = total_wei
    batchnum = attr_maxt.shape[0]
    return batchcat.float(), adjcat.long(), weicat.float(), batchnum

# gp = torch.rand(2,3,52,20)
# adj = torch.tensor([[0,1,2],[4,5,6]])
# wei = torch.tensor([3,3,3])
# print(gp.shape,adj.shape,wei.shape)
# mc,ac,wc, batch_num = combine_graph(gp,adj,wei)
# print(mc.shape,ac.shape,wc.shape,batch_num)
# print(ac,wc)

def normalize(input_maxt,mask_Maxt):
    register = np.zeros((input_maxt.shape[0],input_maxt.shape[1],input_maxt.shape[2]))
    # for maxt_num in range(mask_Maxt.shape[0]):
        # print(mask_Maxt[maxt_num].shape)
    meanvalue = np.mean(mask_Maxt,axis=2)
    stdvalue = np.std(mask_Maxt,axis=2)
    # stdvalue = stdvalue + (stdvalue==0)*np.full((stdvalue.shape[0],stdvalue.shape[1]),0.00001)
    # print(meanvalue.shape,stdvalue.shape)
    # print(np.tile(np.expand_dims(meanvalue,2),[1,1,input_maxt.shape[2]]))
    register = (input_maxt - np.tile(np.expand_dims(meanvalue,2),[1,1,input_maxt.shape[2]])) /\
               np.tile(np.expand_dims(stdvalue,2),[1,1,input_maxt.shape[2]])
    return register

# def normalize(input_maxt,mask_Maxt):
#     register = np.zeros((input_maxt.shape[0],input_maxt.shape[1],input_maxt.shape[2]))
#     # for maxt_num in range(mask_Maxt.shape[0]):
#         # print(mask_Maxt[maxt_num].shape)
#     maxvalue = np.max(mask_Maxt,axis=2)
#     minvalue = np.min(mask_Maxt,axis=2)
#     # print(maxvalue,minvalue)
#     register = (input_maxt - np.tile(np.expand_dims(minvalue,2),[1,1,input_maxt.shape[2]])) /\
#                (np.tile(np.expand_dims(maxvalue,2),[1,1,input_maxt.shape[2]]) -
#                 np.tile(np.expand_dims(minvalue,2),[1,1,input_maxt.shape[2]]))
#     return register

# sd = np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
# print(sd)
# outsd = normalize(sd,sd)
# print(outsd)

def get_adj_wei():
    config = parse_args()
    position_file = open("rawData/IBRL/node.txt", "r")
    fileDate = position_file.read()
    nodeMSG = fileDate.split("\n")
    # print(nodeMSG)
    if "" in nodeMSG:
        nodeMSG.remove("")
    check_file = open("processed/IBRL_count.txt", "r")
    considered_node = list(map(int,eval(check_file.readline().replace("\n",""))))
    # print(considered_node)
    nodeArray = np.ones((len(considered_node),2))
    ct = 0
    for num_index in considered_node:
        split_msg = nodeMSG[num_index - 1].strip().split(" ")
        nodeArray[ct][0] = float(split_msg[1])
        nodeArray[ct][1] = float(split_msg[2])
        ct = ct + 1
    if config.use_kmeans_paa == True:
        split_node_num = get_splitblock_node(get_IBRL_data(),"rawData/IBRL/node.txt",config.cluster_num)
        use_node = split_node_num[config.use_cluster_block]
        nodeArray = nodeArray[use_node]
        # print(nodeArray.shape)
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
    # print(att_res)
    return adj, wei_res

# adj, wei = get_adj_wei()
# print(adj.shape,wei.shape)
# print(wei)

def get_adj_maxt():
    config = parse_args()
    position_file = open("rawData/IBRL/node.txt", "r")
    fileDate = position_file.read()
    nodeMSG = fileDate.split("\n")
    # print(nodeMSG)
    if "" in nodeMSG:
        nodeMSG.remove("")
    check_file = open("processed/IBRL_count.txt", "r")
    considered_node = list(map(int,eval(check_file.readline().replace("\n",""))))
    # print(considered_node)
    nodeArray = np.ones((len(considered_node),2))
    ct = 0
    for num_index in considered_node:
        split_msg = nodeMSG[num_index - 1].strip().split(" ")
        nodeArray[ct][0] = float(split_msg[1])
        nodeArray[ct][1] = float(split_msg[2])
        ct = ct + 1
    if config.use_kmeans_paa == True:
        split_node_num = get_splitblock_node(get_IBRL_data(),"rawData/IBRL/node.txt",config.cluster_num)
        use_node = split_node_num[config.use_cluster_block]
        nodeArray = nodeArray[use_node]
    # print(nodeArray)
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
