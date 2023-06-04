import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphAttentionLayer, SpGraphAttentionLayer
import numpy as np
class MGAT(nn.Module):
    def __init__(self):

        super(MGAT, self).__init__()
        self.lstm = nn.LSTM(10,2,2,batch_first=True)
        self.L = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10,1)
        )
    def forward(self,x,h):
        # print(x.shape,h[0].shape)
        out,H_ = self.lstm(x,h)
        out = self.L(out)
        # print(out.shape,H_[0].shape)

        return out,H_
lstm_len = 5
md = MGAT()
datamaxt = np.array([[(i * 10 + xp) for xp in range(10)] for i in range(0,20)])
label = np.array([i for i in range(100)])
# print(datamaxt.shape)
# print(datamaxt[0])
# print(label)
###########每个时刻10个值现在有20个时刻
# h_1 = torch.zeros(2,1,2).float()
# h_2 = torch.zeros(2,1,2).float()
optimizer_G = torch.optim.Adam(md.parameters(), lr=0.0007)

for epochnum in range(1000):
    h_1 = torch.zeros(2, 1, 2).float()
    h_2 = torch.zeros(2, 1, 2).float()
    for i in range(20 - lstm_len):

        print(torch.from_numpy(datamaxt[i:i+5]).unsqueeze(0).shape)
        out,H_ = md(torch.from_numpy(datamaxt[i:i+5]).unsqueeze(0).float(),(h_1,h_2))
        # print(out.shape)
        # print("s",torch.from_numpy(label)[i].unsqueeze(0))
        loss = F.mse_loss(out.squeeze(0).float(),torch.from_numpy(label)[i].unsqueeze(0).float())

        h_1 = H_[0].detach()
        h_2 = H_[1].detach()

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
    print(loss)
h_1 = torch.zeros(2,1,2).float()
h_2 = torch.zeros(2,1,2).float()
with torch.no_grad():
    for i in range(20 - lstm_len):
        out, H_ = md(torch.from_numpy(datamaxt[i:i + 5]).unsqueeze(0).float(), (h_1, h_2))
        # print(out.shape)
        # print("s",torch.from_numpy(label)[i].unsqueeze(0))
        print("out",out)
        print("label",label[i])
        loss = F.mse_loss(out.squeeze(0), torch.from_numpy(label)[i].unsqueeze(0))
        h_1 = H_[0].detach()
        h_2 = H_[1].detach()
