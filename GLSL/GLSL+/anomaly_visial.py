import  numpy as np
import matplotlib.pyplot as plt
import torch,os
from model.new import GNN
from util.utils import combine_graph,normalize,get_adj_wei
from processed.IBRL_load import get_IBRL_data
from Config import parse_args,get_data_inf,get_block_data

config = parse_args()
use_cuda = config.gpu
device = "cpu"

if use_cuda:
    device = "cuda"


H_ = torch.load("save/save_H_" + config.dataset + ".pth").to(device)
H_ = H_[:,H_.shape[1] - 1:H_.shape[1],:].contiguous()

max_min_res = torch.load("processed/" + config.dataset + "/max_min.pt")
print(max_min_res)


orin_data = get_block_data()

node_num, mode_num = get_data_inf(config.dataset)
adj,weight = get_adj_wei()

print(adj.shape,weight.shape)


train_data = orin_data[:,:,0:int(orin_data.shape[2]*config.train_rate)].copy()
test_data = orin_data[:,:,int(orin_data.shape[2]*config.train_rate) - config.window_size + 1: orin_data.shape[2]].copy()

print(train_data.shape, test_data.shape)
#11，45
inj_mode = 0
inj_node = 11
inj_time = 525  #150异常  #1450正常

nm_test_data = normalize(test_data,train_data)

model = GNN(mode_num,node_num, config.window_size,"GAT").to(device)
model.load_state_dict(torch.load("save/para_" + config.dataset + ".pth"))

nm_result_res = []
mid_nm = np.array(config.window_size)
rec0_nm = np.array(config.window_size)
rec1_nm = np.array(config.window_size)
rec2_nm = np.array(config.window_size)
nm_inp  = np.array(config.window_size)
check_time = inj_time + 10
with torch.no_grad():
    for i in range(nm_test_data.shape[2] - config.window_size):
        # print(torch.tensor(nm_test_data[:,:,i:i+20]).unsqueeze(0).shape)
        batchcat, adjcat, weicat, batchnum = combine_graph(torch.tensor(
            nm_test_data[:,:,i:i+config.window_size]).unsqueeze(0),adj,weight)
        # print(batchcat.shape,adjcat.shape,weicat.shape,batchnum)
        out, rec, new_H = model(batchcat.to(device), adjcat.to(device), weicat.to(device), H_)
        # print(rec.shape)
        new_H = new_H.detach()
        H_ = new_H
        if (i + config.window_size)==check_time:

            # mid_nm = mid[0][inj_node].cpu().numpy()
            rec_data = rec[0 * node_num + inj_node].cpu().numpy()
            rec0_nm = rec[0 * node_num + inj_node].cpu().numpy()
            nm_inp = batchcat[0 * node_num + inj_node].cpu().numpy()
            # rec0_nm = rec_data
            # plt.plot([xray for xray in range(20)],batchcat[0 * node_num + inj_node].cpu().numpy())
            # plt.show()
            # plt.plot([xray for xray in range(20)],rec0_nm)
            # plt.show()

            rec_data = rec[1 * node_num + inj_node].cpu().numpy()
            rec1_nm = rec_data

            rec_data = rec[2 * node_num + inj_node].cpu().numpy()
            rec2_nm = rec_data
        if torch.argmax(out[0]).item() == 1:
            print(i + config.window_size, torch.argmax(out[0]), "异常")
        else:
            print(i + config.window_size, torch.argmax(out[0]))
        nm_result_res.append(torch.argmax(out[0]).item())

ct = inj_time
# change shaplet
# for i in range(200):
#     test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct-1] - 0.001
#     ct = ct + 1

# change min
for i in range(5):
    test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct - 1] + \
                                              (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / 35
    ct = ct + 1

for i in range(5):
    test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct - 1] - \
                                              (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / 35
    ct = ct + 1

# mu = 60
# for i in range(15):
#     test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct - 1] + \
#                                               (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / mu
#     mu = mu + 1
#     ct = ct + 1
#
# mu = 240
# for i in range(15):
#     test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct - 1] + \
#                                               (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / mu
#     mu = mu
#     ct = ct + 1
#
# test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct - 1] - \
#                                               (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / mu

# change sudden
# for i in range(5):
#     test_data[inj_mode][inj_node][ct] = test_data[inj_mode][inj_node][ct] + \
#                                               (max_min_res[0][inj_mode]-max_min_res[1][inj_mode]) / 2
#     ct = ct + 1



H_ = torch.load("save/save_H_" + config.dataset + ".pth").to(device)
H_ = H_[:,H_.shape[1] - 1:H_.shape[1],:].contiguous()
am_test_data = normalize(test_data,train_data)

result_res = []
mid_am = np.array(config.window_size)
rec0_am = np.array(config.window_size)
rec1_am = np.array(config.window_size)
rec2_am = np.array(config.window_size)

import time
st = time.time()

with torch.no_grad():
    for i in range(am_test_data.shape[2] - config.window_size):
        # print(torch.tensor(am_test_data[:,:,i:i+20]).unsqueeze(0).shape)
        batchcat, adjcat, weicat, batchnum = combine_graph(torch.tensor(
            am_test_data[:,:,i:i+config.window_size]).unsqueeze(0),adj,weight)
        # print(batchcat.shape,adjcat.shape,weicat.shape,batchnum)
        out, rec, new_H = model(batchcat.to(device), adjcat.to(device), weicat.to(device), H_)
        new_H = new_H.detach()
        H_ = new_H
        if (i + config.window_size)==check_time:

            # mid_am = mid[0][inj_node].cpu().numpy()
            rec_data = rec[0 * node_num + inj_node].cpu().numpy()
            rec0_am = rec_data

            rec_data = rec[1 * node_num + inj_node].cpu().numpy()
            rec1_am = rec_data

            rec_data = rec[2 * node_num + inj_node].cpu().numpy()
            rec2_am = rec_data
            # break
        if torch.argmax(out[0]).item() == 1:
            print(i + config.window_size, torch.argmax(out[0]), "anomaly")
        else:
            print(i + config.window_size, torch.argmax(out[0]))
        result_res.append(torch.argmax(out[0]).item())
et = time.time()
print(st-et)

def set_x_space_show():
    ax = plt.gca()
    x_space = plt.MultipleLocator(2)
    ax.xaxis.set_major_locator(x_space)
    plt.legend()
    # plt.show()

plt.figure(figsize=(16, 8))


plt.subplot2grid((2,2),(0,1))

plt.xlabel("timestamp")
plt.ylabel("value")
plt.title("(b) anomaly injection in %s-%s timestamps of test set "
          "of modal 1 observed on node 45" %(inj_time,inj_time+5))
td = normalize(train_data, train_data)
td_size = td[inj_mode][inj_node].shape[0] - config.window_size
plt.plot([i for i in range(td_size)],
         td[inj_mode][inj_node][0:td_size], label="train")

consider = result_res
for number in range(len(consider)):
    if number != 0 and number != len(consider)-1:
        if(consider[number] == 1 and consider[number-1] == 0 and
                consider[number+1] == 0):
            # print("remove")
            consider[number]=0

plt.plot([i + td_size for i in range(am_test_data[inj_mode][inj_node].shape[0])],
         am_test_data[inj_mode][inj_node], linewidth=1, label="test")
plt.plot([i + td_size + config.window_size for i in range(len(consider))],
         np.array(consider) - 2, label="predict status")
plt.legend()
# plt.show()


plt.subplot2grid((2,2),(0,0))
plt.xlabel("timestamp")
plt.ylabel("value")
plt.title("(a) normal data flow of modal 1 observed on node 45")
nm_norm = normalize(orin_data, orin_data)
# plt.plot([i for i in range(nm_norm[inj_mode][inj_node].shape[0])],
#          nm_norm[inj_mode][inj_node])
plt.plot([i for i in range(td_size)],
         nm_norm[inj_mode][inj_node][0:td_size], label="train")
plt.plot([i for i in range(td_size, nm_norm[inj_mode][inj_node].shape[0])],
         nm_norm[inj_mode][inj_node][td_size: nm_norm[inj_mode][inj_node].shape[0]], linewidth=1, label="test")

consider = nm_result_res
for number in range(len(consider)):
    if number != 0 and number != len(consider)-1:
        if(consider[number] == 1 and consider[number-1] == 0 and
                consider[number+1] == 0):
            # print("remove")
            consider[number]=0

plt.plot([i + int(config.paa_out_dimension * 0.6) for i in range(len(consider))], np.array(consider) - 2, color="green", label="predict status")
plt.legend()
# plt.show()

# plt.subplot2grid((2,2),(1,0))
#
# plt.xlabel("hidden layer dimension")
#
# plt.ylabel("value")
# plt.title("(c) change of hidden layer in %s timestamp of modal 1 observed on node 45" %check_time)
# plt.plot([i for i in range(mid_nm.shape[0])], mid_nm, color="blue", label="hidden layer with normal input")
# plt.plot([i for i in range(mid_am.shape[0])], mid_am, color="red", label="hidden layer with abnormal input")
# set_x_space_show()

plt.subplot2grid((2,2),(1,1))
plt.xlabel("timestamp")
plt.ylabel("value")
plt.title("(d) change of 1th-model %sth-node reconstruction data in %s-timestamp" %(inj_node, check_time))
plt.plot([i for i in range(nm_inp.shape[0])], nm_inp, color="green", label="normal input")
plt.plot([i for i in range(rec0_nm.shape[0])], rec0_nm, color="blue", label="reconstruction with normal input")
plt.plot([i for i in range(rec0_am.shape[0])], rec0_am, color="red", label="reconstruction with abnormal input")
set_x_space_show()
# plt.tight_layout()
# plt.savefig("context_am.svg", bbox_inches='tight', pad_inches=0.0, format="svg", dpi = 300)
plt.show()
# plt.title("change of 2th-model %s-node reconstructiondata" %inj_node)
# plt.plot([i for i in range(rec1_nm.shape[0])], rec1_nm, color="blue", label="normal")
# plt.plot([i for i in range(rec1_am.shape[0])], rec1_am, color="red", label="abnormal")
# set_x_space_show()
#
# plt.title("change of 3th-model %s-node reconstructiondata" %inj_node)
# plt.plot([i for i in range(rec2_nm.shape[0])], rec2_nm, color="blue", label="normal")
# plt.plot([i for i in range(rec2_am.shape[0])], rec2_am, color="red", label="abnormal")
# set_x_space_show()