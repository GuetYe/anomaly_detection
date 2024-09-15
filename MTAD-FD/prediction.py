from model.mymodel import myFormer, meltFormer
from data.dataset.mydata import sensor_data
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from data.dataset.RevIN import RevIN
import random

revin_layer = RevIN(3, affine=True)

def f1Score(y_hat, y_true):
    tp = np.sum(y_hat*y_true)
    fp = np.sum(y_hat*(1-y_true))
    fn = np.sum((1-y_hat)*y_true)
    # print(tp,fp,fn)
    
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    # print(p,r)
    
    f1 = 2*p*r/(p+r)
    # f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    
    return p,r,f1#np.mean(f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=150, help='input sequence length')
    # parser.add_argument('--label_len', type=int, default=20, help='start token length')
    parser.add_argument('--pred_len', type=int, default=150, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')#时间长度有关（除以采样数）512
    parser.add_argument('--d_ff', type=int, default=2, help='dimension of fcn')#通道数有关
    parser.add_argument('--fac_T', action='store_true', default=True, help='whether to apply factorized temporal interaction')
    parser.add_argument('--fac_C', action='store_true', default=True, help='whether to apply factorized channel interaction')
    parser.add_argument('--sampling', type=int, default=2, help='the number of downsampling in factorized temporal interaction')
    parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')#网络层数
    parser.add_argument('--rev', action='store_true', default=False, help='whether to apply RevIN')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj=torch.ones(size=(51, 51)).to(device)
    former=myFormer(args, adj).to(device)
    # former=meltFormer(args, adj).to(device)
    revin_layer.to(device)
    file_path="./data/dataset"
    data_test=sensor_data(file_path, mode='test')
    param = torch.load("./dwt_weights/model_56.pth")#dwt70|56 mv39 -tem10 -gcn3
    former.load_state_dict(param,strict = False)

 
    i=0
    select = random.sample(range(9), 7)
    acc=[]
    Pre=[]
    Rec=[]
    former.eval()
    # print(select)
    for wi_te,da_te,la_te in data_test:    
        # i+=1  #i=0
        # if i-1 not in select:
        #     continue
        
        # print(i)
        # i+=1
        da_te=torch.FloatTensor(da_te).to(device)
        # la_te=torch.FloatTensor(la_te).to(device)
        wi_te=torch.FloatTensor(wi_te).to(device)
        da_te=revin_layer(da_te.permute(0,2,1), 'norm').permute(0,2,1)
        with torch.no_grad():
            output=former(da_te)
        output=revin_layer(output.permute(0,2,1), 'denorm').permute(0,2,1)
        da_te=revin_layer(da_te.permute(0,2,1), 'denorm').permute(0,2,1)

        dra = abs(output-da_te)
        dra[:,0,:] = dra[:,0,:]/8.5
        dra[:,1,:] = dra[:,1,:]/13.5
        cha=dra.cpu().detach().numpy()
        out1=np.where(cha>=0.38,1,0)#0.435
        p,r,f1=f1Score(out1, la_te)
        Pre.append(p)
        Rec.append(r)
        acc.append(f1)
        # print("f1:",f1)

    
        wi_te=wi_te.cpu().detach().numpy()
        da_te=da_te.cpu().detach().numpy()
        output=output.cpu().detach().numpy()
        if i==0:
            win_tem=wi_te
            data_tem=output
            da_tem=da_te
            label_tem=la_te
            i=1
        else:
            win_tem=np.append(win_tem,wi_te,axis=2)
            data_tem=np.append(data_tem,output,axis=2)
            da_tem=np.append(da_tem,da_te,axis=2)
            label_tem=np.append(label_tem,la_te,axis=2)  
    
    # print(la_te)
    print(">>>p,r,f1:",np.array(Pre).mean(),np.array(Rec).mean(),np.array(acc).mean())



