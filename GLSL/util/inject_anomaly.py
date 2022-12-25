import  numpy as np

def mirro(orin_data,mode_num,node_num,start_t,end_t):
    new_data = orin_data.copy()
    # print(new_data)
    for i in range(start_t,end_t):
        # print(new_data[modal_num].shape)
        # print(new_data[modal_num])
        if new_data[mode_num][node_num][i] == 0:
            new_data[mode_num][node_num][i]=new_data[mode_num][node_num][i]+0.00001
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i] * (-1)
    # print("mirro#",new_data[mode_num][node_num][start_t:end_t])
    # print(new_data)
    return new_data

def scale(orin_data,mode_num,node_num,start_t,end_t):
    new_data = orin_data.copy()
    # print(new_data)
    # print(new_data[mode_num][node_num][start_t:end_t])
    scale_choice = [0.5, 1.5, 2]
    for i in range(start_t,end_t):
        # print(i)
        num = np.random.randint(0,2)
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i] * scale_choice[num]
    # print("scale#",new_data[mode_num][node_num][start_t:end_t])
    return new_data

def surge(orin_data,mode_num,node_num,start_t,end_t,max_min_res):
    new_data = orin_data.copy()
    # print(new_data)
    # print(new_data[mode_num][node_num][start_t:end_t])
    num = np.random.randint(0, 1)
    for i in range(start_t,end_t):
        # print(i)
        if num == 0:
            new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i] + \
                                              (max_min_res[0][mode_num] - max_min_res[1][mode_num])
        elif num == 1:
            new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i] - \
                                              (max_min_res[0][mode_num] - max_min_res[1][mode_num])
    # print("surge#",new_data[mode_num][node_num][start_t:end_t])
    return new_data

def decay(orin_data, mode_num, node_num, start_t, end_t, max_min_res, config):
    new_data = orin_data.copy()
    for i in range(start_t, end_t):
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i - 1] - \
                                          (max_min_res[0][mode_num] - max_min_res[1][mode_num]) / config.test_inj_deviation
    for i in range(end_t, end_t + (end_t - start_t)):
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i - 1] + \
                                          (max_min_res[0][mode_num] - max_min_res[1][mode_num]) / config.test_inj_deviation
    print("d#",new_data[mode_num][node_num][start_t:end_t])
    return new_data

def increase(orin_data, mode_num, node_num, start_t, end_t, max_min_res, config):
    new_data = orin_data.copy()
    for i in range(start_t, end_t):
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i - 1] + \
                                          (max_min_res[0][mode_num] - max_min_res[1][mode_num]) / config.test_inj_deviation
    for i in range(end_t, end_t + (end_t - start_t)):
        new_data[mode_num][node_num][i] = new_data[mode_num][node_num][i - 1] - \
                                          (max_min_res[0][mode_num] - max_min_res[1][mode_num]) / config.test_inj_deviation
    # print("i#",new_data[mode_num][node_num][start_t:end_t])
    return new_data


def trendline(data):
    order=1
    index=[i for i in range(1,len(data)+1)]
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)

def intermodal_anomaly(mode_num , series_data, inject_mode_num,inject_node_num, start_time, end_time):
    trend = trendline(series_data[inject_mode_num][inject_node_num][start_time:end_time])
    for search_num in range(mode_num):
        if search_num != inject_mode_num:
            correlation = np.corrcoef(series_data[search_num][inject_node_num][start_time:end_time].copy(),
                                      series_data[inject_mode_num][inject_node_num][start_time:end_time].copy())
            # print(search_num, sub_search, correlation[0][1])
            target = np.random.uniform()
            if target > 0.5 and 0.8 < correlation[0][1] <= 1 :
                # print("inter-mode positive correlation")
                if trend < 0:
                    inj_type = 4
                elif trend > 0:
                    inj_type = 3
                return inj_type
            elif target > 0.5 and -0.8 > correlation[0][1] >= -1 :
                # print("inter-mode negative correlation")
                if trend < 0:
                    inj_type = 4
                elif trend > 0:
                    inj_type = 3
                return inj_type
    return False

def internode_anomaly(adj_maxt , series_data, inject_mode_num,inject_node_num, start_time, end_time):
    trend = trendline(series_data[inject_mode_num][inject_node_num][start_time:end_time])
    # print(adj_maxt[inject_node_num])
    sort_nodes = adj_maxt[inject_node_num].copy()
    sort_nodes.sort()
    # print(sort_nodes)
    reference_node = list(adj_maxt[inject_node_num]).index(sort_nodes[1])
    # print(reference_node)
    correlation = np.corrcoef(series_data[inject_mode_num][reference_node][start_time:end_time].copy(),
                              series_data[inject_mode_num][inject_node_num][start_time:end_time].copy())
    # print("correlation", correlation)
    target = np.random.uniform()
    if target > 0.7 and 0.6 < correlation[0][1] <= 1:
        # print("inter-node positive correlation")
        if trend < 0:
            inj_type = 4
        elif trend > 0:
            inj_type = 3
        return inj_type
    elif target > 0.7 and -0.6 > correlation[0][1] >= -1:
        # print("inter-node negative correlation")
        if trend < 0:
            inj_type = 4
        elif trend > 0:
            inj_type = 3
        return inj_type
    return False