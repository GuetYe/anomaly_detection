import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphAttentionLayer, SpGraphAttentionLayer
# from layers import GraphAttentionLayer, SpGraphAttentionLayer

class MGAT(nn.Module):
    def __init__(self,node_num, sensor_num, slide_win, dropout, alpha):

        super(MGAT, self).__init__()
        self.node_num = node_num
        self.sensor_num = sensor_num
        self.slide_win = slide_win
        self.gru_hid_size = 32

        self.feature_attentions = [GraphAttentionLayer(slide_win, slide_win, dropout=dropout, alpha=alpha, concat=False) for _ in range(node_num)]
        for i, attention in enumerate(self.feature_attentions):
            self.add_module('Fattention_{}'.format(i), attention)

        self.time_attentions = [GraphAttentionLayer(sensor_num, sensor_num, dropout=dropout, alpha=alpha, concat=False) for _ in range(node_num)]
        for i, attention in enumerate(self.time_attentions):
            self.add_module('Tattention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(sensor_num, sensor_num, dropout=dropout, alpha=alpha, concat=False)
        self.out_att1 = GraphAttentionLayer(sensor_num, sensor_num, dropout=dropout, alpha=alpha, concat=False)
        self.out_att2 = GraphAttentionLayer(sensor_num * 2, sensor_num, dropout=dropout, alpha=alpha, concat=False)

        self.grus = [nn.GRU( slide_win * 3, self.gru_hid_size,2,batch_first=True) for _ in range(node_num)]
        for i, gru_ in enumerate(self.grus):
            self.add_module('lstms_{}'.format(i), gru_)

        self.Fc = [nn.Sequential(
            nn.Linear(self.gru_hid_size, self.gru_hid_size), nn.ReLU(),
            nn.Linear(self.gru_hid_size, self.gru_hid_size), nn.ReLU(),
            nn.Linear(self.gru_hid_size, 1)
        ) for _ in range(node_num)]
        for i, fc_ in enumerate(self.Fc):
            self.add_module('fcs_{}'.format(i), fc_)


    def forward(self, x, Feature_adj_sub,Time_adj_sub, adj,H_pre):

        for count in range(x.shape[0]):
            
            if count == 0:
                Fat = self.feature_attentions[count](x[count],Feature_adj_sub[count])
                
                Tat = self.time_attentions[count](x[count].T,Time_adj_sub[count])
                
                cat_at = torch.cat([Fat,Tat.T,x[count]],dim = 1).unsqueeze(0)
                
                gru_tensor,cat_H = self.grus[count](cat_at,H_pre[:,count:count+1].contiguous()
                                                         )
                
                cat_tensor = self.Fc[count](gru_tensor)
               
            else:
                Fat = self.feature_attentions[count](x[count],Feature_adj_sub[count])
               
                Tat = self.time_attentions[count](x[count].T,Time_adj_sub[count])
                
                cat_at = torch.cat([Fat,Tat.T,x[count]],dim = 1).unsqueeze(0)
               
                gru_tensor,this_H = self.grus[count](cat_at,H_pre[:,count:count+1].contiguous()
                                                           )
                this_tensor = self.Fc[count](gru_tensor)
                cat_tensor = torch.cat([cat_tensor,this_tensor],dim=0)
                
                cat_H = torch.cat([cat_H, this_H], dim=1)
        
        out = self.out_att(cat_tensor.squeeze(2),adj)
        out1 = self.out_att1(cat_tensor.squeeze(2),adj)
        finalout = self.out_att2(torch.cat([out,out1],dim=1),adj)
        
        return  finalout ,cat_H

