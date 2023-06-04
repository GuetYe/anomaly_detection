# from matplotlib import pyplot as plt
from processed.IBRL_load import get_data,get_TopK_adj,get_Gen_adj
# import sranodec as anom
import numpy as np
import random

def add_noise(orin_data):
    new_data = orin_data.copy()
    # print(new_data)
    for modal_num in range(new_data.shape[0]):
        # print(new_data[modal_num].shape)
        # print(new_data[modal_num])
        noise_maxt = np.random.normal(loc=0,
                                      scale=np.mean(new_data[modal_num])/5,
                                      size=new_data[modal_num].shape)
        # print(noise_maxt)
        new_data[modal_num] = new_data[modal_num] + noise_maxt
        # print(new_data[modal_num])
    # print(new_data)
    return new_data

def scale(orin_data):
    new_data = orin_data.copy()
    # print(new_data)
    scale_choice = [0.5, 0.8, 1.5, 2]
    for modal_num in range(new_data.shape[0]):
        # print(new_data[modal_num].shape)
        # print(new_data[modal_num])
        new_data[modal_num] = new_data[modal_num] * scale_choice[random.randint(0,3)]
        # print(new_data[modal_num])
    # print(new_data)
    return new_data

def mirro(orin_data):
    new_data = orin_data.copy()
    # print(new_data)
    scale_choice = [0.5, 0.8, 1.5, 2]
    for modal_num in range(new_data.shape[0]):
        # print(new_data[modal_num].shape)
        # print(new_data[modal_num])
        new_data[modal_num] = new_data[modal_num] **(-1)
    # print(new_data)
    return new_data



def trans_IBRL_dataset():
    data = get_data()
    noise_dataset = add_noise(data)
    mirro_dataset = mirro(data)
    scale_dataset = scale(data)
    return data,noise_dataset,mirro_dataset,scale_dataset

