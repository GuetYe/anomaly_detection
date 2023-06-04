import torch,random
from model.models import MGAT
from dataloader import get_data,get_TopK_adj
import numpy as np

####----基础参数----####
train_rate = 0.8
vali_rate = 0.1
threshold = 2.8924484252929688
slow_ = 14
fast_ = 9
sudden_ = 1.01

slow_inj_time = 8
fast_inj_time = 4

test_time = 150


data_maxt = get_data()
print(data_maxt.shape)

def normalize(input_maxt,mask_Maxt):
    register = np.zeros((input_maxt.shape[0],input_maxt.shape[1],input_maxt.shape[2]))
    for maxt_num in range(mask_Maxt.shape[0]):
        # print(mask_Maxt[maxt_num].shape)
        meanvalue = np.mean(mask_Maxt[maxt_num],axis=1)
        stdvalue = np.std(mask_Maxt[maxt_num],axis=1)
        # print(meanvalue,stdvalue)
        # print(np.tile(np.array(meanvalue),[input_maxt[maxt_num].shape[1],1]).T.shape)
        register[maxt_num] = (input_maxt[maxt_num] - np.tile(np.array(meanvalue),[input_maxt[maxt_num].shape[1],1]).T) /\
           np.tile(np.array(stdvalue),[input_maxt[maxt_num].shape[1],1]).T
    return register


train_set = data_maxt[:,:,0:int(data_maxt.shape[2] * train_rate)]
print("训练集",train_set.shape)
vali_set = data_maxt[:,:,int(data_maxt.shape[2] * train_rate):int(data_maxt.shape[2] * (train_rate + vali_rate))]
print("验证集",vali_set.shape)
test_set = data_maxt[:,:,int(data_maxt.shape[2] * (train_rate + vali_rate)):int(data_maxt.shape[2])]
print("测试集",test_set.shape)

max_res = [float('-inf') for xp in range(train_set.shape[1])]
min_res = [float('inf') for xp in range(train_set.shape[1])]
for read_num in range(train_set.shape[0]):
    this_maxt = train_set[read_num]
    max_list = np.max(this_maxt,axis=1)
    for maxcount in range(len(max_list)):
        if max_list[maxcount] > max_res[maxcount]:
            # print(max_list[maxcount],"大于",max_res[maxcount])
            max_res[maxcount] = max_list[maxcount]
    min_list = np.min(this_maxt,axis=1)
    for mincount in range(len(min_list)):
        if min_list[mincount] < min_res[mincount]:
            # print(min_list[mincount],"小于",min_res[mincount])
            min_res[mincount] = min_list[mincount]
print("训练集中%s种模态的最大值为%s" %(data_maxt.shape[1],max_res))
print("训练集中%s种模态的最小值为%s" %(data_maxt.shape[1],min_res))


# ###---初始化lstm_GAT预测模型---###
node_num = 50
measure_dim = 3
slide_win = 60


mgat = MGAT(node_num,measure_dim,slide_win,0,0.05)
mgat.load_state_dict(torch.load('save/3_dim.pt'))

def rec_loss(pred,label):
    nodescore_res = []
    for i in range(pred.shape[0]):
        nodescore_res.append(torch.sum(torch.abs(pred[i] - label[i])))
    return max(nodescore_res)

def detact_start(input_set,slide_win,inj_point):
    delay = 8
    target_true = 0
    F_adj_sub = torch.ones(50, 3, 3)
    T_adj_sub = torch.ones(50, slide_win, slide_win)
    adj = torch.ones(50,50)
    with torch.no_grad():
        h_1 = torch.zeros(2, 50, 32)
        scorelist = []
        for timecount in range(input_set.shape[2] - slide_win - 1):
            # print(timecount,timecount + slide_win,timecount + slide_win + 1)
            now_tensor = torch.from_numpy(input_set[:, :, timecount:timecount + slide_win]).float()
            label = torch.from_numpy(input_set[:, :, timecount + slide_win + 1]).float()
            out, H_ = mgat(now_tensor,F_adj_sub,T_adj_sub,adj,h_1)
            score = rec_loss(out, label)
            h_1 = H_.detach()
            # print(out.shape)
            if score > threshold and inj_point<= (timecount + slide_win + 1) < (inj_point + delay):
                # print(score, "anomaly",timecount + slide_win + 1)
                target_true = 1
                scorelist.append(float(score.detach()))

        return target_true,scorelist

def normal_start(input_set,slide_win,inj_point):
    delay = 1
    target_true = 1
    F_adj_sub = torch.ones(50, 3, 3)
    T_adj_sub = torch.ones(50, slide_win, slide_win)
    adj = torch.ones(50,50)
    with torch.no_grad():
        h_1 = torch.zeros(2, 50, 32)
        scorelist = []
        for timecount in range(input_set.shape[2] - slide_win - 1):
            # print(timecount,timecount + slide_win,timecount + slide_win + 1)
            now_tensor = torch.from_numpy(input_set[:, :, timecount:timecount + slide_win]).float()
            label = torch.from_numpy(input_set[:, :, timecount + slide_win + 1]).float()
            out, H_ = mgat(now_tensor,F_adj_sub,T_adj_sub,adj,h_1)
            score = rec_loss(out, label)
            h_1 = H_.detach()
            # print(out.shape)
            if score > threshold and inj_point<= (timecount + slide_win + 1) < (inj_point + delay):
                # print(score, "anomaly",timecount + slide_win + 1)
                target_true = 0
                scorelist.append(float(score.detach()))

        return target_true,scorelist

log_file = open("anomaly_detection/log_hard.txt","w",encoding="utf8")
log_file.write("###---训练集中%s种模态的最大值为%s---###\n" %(data_maxt.shape[1],max_res))
log_file.write("###---训练集中%s种模态的最小值为%s---###\n" %(data_maxt.shape[1],min_res))
log_file.write("###___阈值%s___###\n" %threshold)

#####有错-注入总长度为：300-slidewin才对，前40不动
inject_point = list(map(round,np.linspace(0+slide_win+1,test_set.shape[2]-1,num=test_time).tolist()))
# print(len(inject_point))
# random.shuffle(inject_point)
inject_anomaly = inject_point[0:int(test_time*0.5)]
inject_normal = inject_point[int(test_time*0.5):test_time]
print(len(inject_anomaly))
print(len(inject_normal))

target_anomaly = []
target_normal = []
for time_num in range(len(inject_anomaly)):
    random_node = random.randint(0,node_num - 1)
    random_measure = random.randint(0, measure_dim - 1)
    random_win = inject_anomaly[time_num]
    print(time_num," nodenum:",random_node,"modenum:",random_measure,"winnum:",random_win)
    if random_win < 290:
        inject_mode = random.randint(0,3)
    else:
        inject_mode = random.randint(0,1)
    # print(inject_mode)

    if inject_mode == 0:
        print("注入值变0异常")
        input_set = test_set.copy()
        log_file.write("注入值变0异常|注入节点%s模态%s时刻%s,原值%s,现值0" %(random_node,random_measure,random_win,
                                                           input_set[random_node][random_measure][random_win]))
        input_set[random_node][random_measure][random_win] = 0
        input_set = normalize(input_set.copy(), train_set.copy())
        target_true , score_list = detact_start(input_set, slide_win, random_win)
        log_file.write("!!!##检测对%s 分%s\n" %(target_true,str(score_list)))
        print(target_true)

    elif inject_mode == 1:
        print("注入值骤变异常")
        input_set = test_set.copy()
        log_file.write("注入值骤变异常|注入节点%s模态%s时刻%s,原值%s" %(random_node,random_measure,random_win,
                                                           input_set[random_node][random_measure][random_win]))
        input_set[random_node][random_measure][random_win] = input_set[random_node][random_measure][random_win] + \
                                         random.uniform((max_res[random_measure] - min_res[random_measure]),
                                        (max_res[random_measure] - min_res[random_measure]) * sudden_)
        log_file.write(",现值%s" %input_set[random_node][random_measure][random_win])
        input_set = normalize(input_set.copy(), train_set.copy())
        target_true,score_list = detact_start(input_set, slide_win, random_win)
        log_file.write("!!!##检测对%s 分%s\n" % (target_true,str(score_list)))
        print(target_true)
    elif inject_mode == 2:
        print("注入缓慢增加异常")
        input_set = test_set.copy()
        log_file.write("注入缓慢增加异常|注入节点%s模态%s时刻%s-%s" %(random_node,random_measure,random_win,random_win + 7
                                                           ))
        inc_value = (max_res[random_measure] - min_res[random_measure]) / slow_
        for ct in range(slow_inj_time):
            input_set[random_node][random_measure][random_win + ct + 1] = \
                input_set[random_node][random_measure][random_win + ct] + inc_value
        log_file.write("现值"+str(input_set[random_node][random_measure][random_win:random_win+6]))
        input_set = normalize(input_set.copy(), train_set.copy())
        target_true,score_list = detact_start(input_set, slide_win, random_win)
        log_file.write("!!!##检测对%s 分%s\n" % (target_true,str(score_list)))
        print(target_true)

    elif inject_mode == 3:
        print("注入快速增加异常")
        log_file.write("注入快速增加异常|注入节点%s模态%s时刻%s-%s" %(random_node,random_measure,random_win,random_win + 5
                                                           ))
        input_set = test_set.copy()
        inc_value = (max_res[random_measure] - min_res[random_measure]) / fast_
        for ct in range(fast_inj_time):
            input_set[random_node][random_measure][random_win + ct + 1] = \
                input_set[random_node][random_measure][random_win + ct] + inc_value
        log_file.write("现值" + str(input_set[random_node][random_measure][random_win:random_win + 4]))

        input_set = normalize(input_set.copy(), train_set.copy())
        target_true ,score_list= detact_start(input_set, slide_win, random_win)
        log_file.write("!!!##检测对%s 分%s\n" % (target_true,str(score_list)))
        print(target_true)
    target_anomaly.append(target_true)
    if len(target_anomaly) != time_num + 1>0:
        print("sssssss",len(target_anomaly))

for time_num in range(len(inject_normal)):
    random_node = random.randint(0,node_num - 1)
    random_measure = random.randint(0, measure_dim - 1)
    random_win = inject_normal[time_num]
    print(time_num," nodenum:",random_node,"modenum:",random_measure,"winnum:",random_win)
    log_file.write("不注入异常|注入节点%s模态%s时刻%s" % (random_node, random_measure, random_win))
    input_set = test_set.copy()
    input_set = normalize(input_set.copy(), train_set.copy())
    target_true, score_list = normal_start(input_set, slide_win, random_win)
    log_file.write("!!!##检测对%s \n" % (target_true))
    print(target_true)
    target_normal.append(target_true)
#     target_false_res.append(target_false)
print(sum(target_anomaly),sum(target_normal))
log_file.write(str(sum(target_anomaly))+" "+str(sum(target_normal))+"\n")
log_file.write("精准率:"+str(sum(target_anomaly)/ (int(test_time*0.5)-sum(target_anomaly)+sum(target_anomaly))))
log_file.write("召回率:"+str(sum(target_anomaly)/ ((test_time - int(test_time*0.5))-sum(target_normal)+sum(target_anomaly))))

log_file.close()

