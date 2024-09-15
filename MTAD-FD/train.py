from model.mymodel import myFormer, meltFormer
from data.dataset.mydata import sensor_data
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from data.dataset.RevIN import RevIN

revin_layer = RevIN(3, affine=True)

def f1Score(y_hat, y_true):
    tp = np.sum(y_hat*y_true)
    fp = np.sum(y_hat*(1-y_true))
    fn = np.sum((1-y_hat)*y_true)
    # print(tp,fp,fn)
    
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    # print(p)
    
    f1 = 2*p*r/(p+r)
    # f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    
    return f1#np.mean(f1)


 
class My_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        sub = x - y
        sub[:,0,:] = sub[:,0,:]/8.5
        sub[:,1,:] = sub[:,1,:]/13.5
        return torch.mean(torch.pow(sub, 2))
 
# #使用：
# criterion = My_loss()
# loss = criterion(outputs, targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=300, help='input sequence length')
    # parser.add_argument('--label_len', type=int, default=20, help='start token length')
    parser.add_argument('--pred_len', type=int, default=300, help='prediction sequence length')
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
    # former=myFormer(args, adj).to(device)
    former=meltFormer(args, adj).to(device)
    revin_layer.to(device)
    file_path="./data/dataset"
    data_train=sensor_data(file_path,stride=300, mode='train')
    data_test=sensor_data(file_path, mode='test')
    criterion = My_loss()#torch.nn.MSELoss()#torch.nn.BCELoss()
    model_optim = torch.optim.Adam(former.parameters(), lr=0.001)
    
    epochs=100
    bloss=0.1
    history=[]
    for epoch in range(epochs):
        former.train()
        train_loss=[]
        test_loss=[]
        i=1
        for wi_tr,da_tr,la_tr in data_train:   
            model_optim.zero_grad() 
            wi_tr=torch.FloatTensor(wi_tr).to(device)
            
            da_tr=torch.FloatTensor(da_tr).to(device)
            # la_tr=torch.FloatTensor(la_tr).to(device)
            da_tr=revin_layer(da_tr.permute(0,2,1), 'norm').permute(0,2,1)
            output=former(da_tr)
            output=revin_layer(output.permute(0,2,1), 'denorm').permute(0,2,1)
            da_tr=revin_layer(da_tr.permute(0,2,1), 'denorm').permute(0,2,1)

            
            loss = criterion(output, wi_tr)
            loss.requires_grad_(True)
            loss.backward()
            model_optim.step()
            # e_loss.append(loss.item())        
            train_loss.append(loss.item())

            # wi_tr=wi_tr.cpu().detach().numpy()
            # da_tr=da_tr.cpu().detach().numpy()
            # output=output.cpu().detach().numpy()
            # if i:
            #     win_tem=wi_tr
            #     data_tem=output
            #     da_tem=da_tr
            #     label_tem=la_tr
            #     i=0
            # else:
            #     win_tem=np.append(win_tem,wi_tr,axis=2)
            #     data_tem=np.append(data_tem,output,axis=2)
            #     da_tem=np.append(da_tem,da_tr,axis=2)
            #     label_tem=np.append(label_tem,la_tr,axis=2)    

        train_loss=np.array(train_loss).mean()
        print("train_loss:",train_loss)

        i=1
        former.eval()
        for wi_te,da_te,la_te in data_test:    
            model_optim.zero_grad() 
            da_te=torch.FloatTensor(da_te).to(device)
            # la_te=torch.FloatTensor(la_te).to(device)
            wi_te=torch.FloatTensor(wi_te).to(device)
            da_te=revin_layer(da_te.permute(0,2,1), 'norm').permute(0,2,1)
            with torch.no_grad():
                output=former(da_te)
            output=revin_layer(output.permute(0,2,1), 'denorm').permute(0,2,1)
            da_te=revin_layer(da_te.permute(0,2,1), 'denorm').permute(0,2,1)

            # np.where(output>=1,1,0)
            loss = criterion(output, wi_te)
            dra = abs(output-da_te)
            dra[:,0,:] = dra[:,0,:]/8.5
            dra[:,1,:] = dra[:,1,:]/13.5
            cha=dra.cpu().detach().numpy()
            out1=np.where(cha>=0.3,1,0)
            acc = f1Score(out1, la_te)
            # acc = np.mean(out1 == la_te.cpu().detach().numpy())
            test_loss.append(loss.item())
        
            # wi_te=wi_te.cpu().detach().numpy()
            # da_te=da_te.cpu().detach().numpy()
            # output=output.cpu().detach().numpy()
            # if i:
            #     win_tem=wi_te
            #     data_tem=output
            #     da_tem=da_te
            #     label_tem=la_te
            #     i=0
            # else:
            #     win_tem=np.append(win_tem,wi_te,axis=2)
            #     data_tem=np.append(data_tem,output,axis=2)
            #     da_tem=np.append(da_tem,da_te,axis=2)
            #     label_tem=np.append(label_tem,la_te,axis=2)  
        
        # print(la_te)
        test_loss=np.array(test_loss).mean()
        print("test_loss:",test_loss)
        print("acc:",acc)
        history.append([train_loss,test_loss])
        if bloss > test_loss:
            bloss = test_loss
            print(">>>>>loss:{}".format(bloss))
            torch.save(former.state_dict(), "./-gcn_weights/model_{}.pth".format(epoch)) 
    
    plt.figure()
    plt.plot(history)
    plt.savefig("./record/loss_gcn-.png")
        



    # dra = abs(data_tem-da_tem)
    # dra[:,0,:] = dra[:,0,:]/8.5
    # dra[:,1,:] = dra[:,1,:]/13.5
    
    # plt.figure()
    # plt.plot(win_tem[0].T)
    # plt.figure()
    # plt.plot(data_tem[0].T)
    # plt.figure()
    # plt.plot(da_tem[0].T)
    # plt.figure()
    # plt.plot(dra[0].T)
    # plt.figure()
    # plt.plot(label_tem[0].T)
    # plt.show()




