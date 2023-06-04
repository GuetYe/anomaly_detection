import numpy as np
from prim import prim
from torch_geometric.data import Data
import torch
from tqdm import tqdm



#####----
exclude = []
upper = 28675
lower = 14285
datalen = 5000
measure_dim = 3
sensornum = 52

def get_similar(orinlist,checklist,data_register):
    check_array = np.array(checklist)
    res = []
    for orin in orinlist:
        mask = abs(check_array - orin)
        # print(mask,np.argmin(mask))
        res.append(data_register[np.argmin(mask)])
    # print(res)
    return res

# get_similar(ab,ssd)


voltage_register = np.zeros((sensornum,measure_dim,datalen))
model_orient_res = np.zeros((measure_dim,sensornum,datalen))
def get_IBRL_data():
    countfile = open("processed/IBRL_count.txt", "rb")
    idlist = list(map(int, eval(countfile.readline().decode("utf8").replace("\n", ""))))
    numlist = list(map(int, eval(countfile.readline().decode("utf8").replace("\n", ""))))

    sample_point = np.int16(np.linspace(lower,upper,datalen))
    # print(sample_point)
    rowcount = 0
    for filename in idlist:
        if filename not in exclude:
            # print(filename)
            readfile = open("orther/cut_IBRL/" + str(filename) + ".txt", "rb")
            data_epoch = []
            data_register = []
            for linenum in range(numlist[idlist.index(filename)]):
                linedata = readfile.readline()
                data_epoch.append(int(linedata.decode("utf8").replace("\n","").split(" ")[2]))
                # print(list(map(float,linedata.decode("utf8").replace("\n","").split(" ")[4:6] +
                #       [linedata.decode("utf8").replace("\n","").split(" ")[7]])))
                data_register.append(list(map(float,linedata.decode("utf8").replace("\n","").split(" ")[4:6] +
                      [linedata.decode("utf8").replace("\n","").split(" ")[7]])))
            # print(len(data_epoch),len(data_register))
            serial_data = get_similar(sample_point,data_epoch,data_register)
            # print(len(serial_data))
            # print(len(serial_data),len(serial_data[0]))
            # print(rowcount)
            # print(np.array(serial_data).T.shape)
            voltage_register[rowcount] = np.array(serial_data).T
            rowcount = rowcount + 1
    for model_num in range(measure_dim):
        model_orient_res[model_num] = voltage_register[:, model_num, :]
    return model_orient_res
# dataset=get_data()
# print(dataset.shape)

# print(dataset[0][1][0:10])
# print(dataset[0][2][0:10])

TopK = 10
def get_TopK_adj():
    countF = open("processed/IBRL_count.txt")
    selectNodeList = list(map(int, list(eval(countF.read().split("\n")[0]))))
    countF.close()

    NodeMsgList = []
    nodemsgF = open("rawData/IBRL/node.txt")
    allmsg = nodemsgF.read()
    msglist = allmsg.split("\n")
    for i in range(len(msglist)):
        # print(msglist[i])
        rowList = msglist[i].split(" ")
        if int(rowList[0]) in selectNodeList and int(rowList[0]) not in exclude:
            # print("yes")
            NodeMsgList.append([float(rowList[0]), float(rowList[1]), float(rowList[2])])
    nodemsgF.close()
    NodeMsgArray = np.array(NodeMsgList)
    # print(NodeMsgArray,NodeMsgArray.shape)

    NodeADJ = np.zeros([NodeMsgArray.shape[0], NodeMsgArray.shape[0]])
    for startNum in range(NodeMsgArray.shape[0]):
        for targetNum in range(NodeMsgArray.shape[0]):
            # print(startNum," to ",targetNum)
            # print(NodeMsgArray[startNum],NodeMsgArray[targetNum])
            distance = ((NodeMsgArray[startNum][1] - NodeMsgArray[targetNum][1]) ** 2 + \
                        (NodeMsgArray[startNum][2] - NodeMsgArray[targetNum][2]) ** 2) ** (1 / 2)
            # print(distance)
            NodeADJ[startNum][targetNum] = distance
    # print(NodeADJ)

    for i in range(len(NodeADJ)):
        # print(NodeADJ[i])
        listres = NodeADJ[i].copy()
        Topvalue = np.sort(NodeADJ[i])[len(NodeADJ[i]) - TopK:]
        for num in range(len(listres)):
            if listres[num] not in Topvalue:
                listres[num] = 0
            else:
                listres[num] = 1
        NodeADJ[i] = listres
    return NodeADJ

def get_Gen_adj():
    countF = open("processed/IBRL_count.txt")
    selectNodeList = list(map(int, list(eval(countF.read().split("\n")[0]))))
    countF.close()

    NodeMsgList = []
    nodemsgF = open("rawData/IBRL/node.txt")
    allmsg = nodemsgF.read()
    msglist = allmsg.split("\n")
    for i in range(len(msglist)):
        # print(msglist[i])
        rowList = msglist[i].split(" ")
        if int(rowList[0]) in selectNodeList and int(rowList[0]) not in exclude:
            # print("yes")
            NodeMsgList.append([float(rowList[0]), float(rowList[1]), float(rowList[2])])
    nodemsgF.close()
    NodeMsgArray = np.array(NodeMsgList)
    # print(NodeMsgArray,NodeMsgArray.shape)
    total_adj_dict = {}
    for startNum in range(NodeMsgArray.shape[0]):
        target_patch = {}
        for targetNum in range(NodeMsgArray.shape[0]):
            # print(startNum," to ",targetNum)
            # print(NodeMsgArray[startNum],NodeMsgArray[targetNum])
            distance = ((NodeMsgArray[startNum][1] - NodeMsgArray[targetNum][1]) ** 2 + \
                        (NodeMsgArray[startNum][2] - NodeMsgArray[targetNum][2]) ** 2) ** (1 / 2)
            target_patch.update({str(targetNum):distance})
            # print(distance)
        # print(target_patch)
        total_adj_dict.update({str(startNum):target_patch})
    # print(total_adj_dict)
    path = prim(total_adj_dict,"0")
    # print(np.array(path).shape)
    # print(np.array(path))
    # print(np.array(path).T)
    return np.array(path).T


