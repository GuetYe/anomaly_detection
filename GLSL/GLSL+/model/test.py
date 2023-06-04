import torch_geometric,torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, modenum,nodenum,slid_win):
        super().__init__()
        self.nn = nn.M
        self.gin = gnn.GINConv(self.nn, train_eps=True)

    def forward(self, x,adj):
       out = self.gin(x,adj)
       print(out.shape)

###IBRL 2batch test###
graph1_x = torch.rand(312,20)
graph1_a = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
# H_ = torch.ones(2,2,16)
model = GNN(3,52,20)
model(graph1_x.float(),graph1_a)
# print(cf_out.shape,rec_out.shape,new_H.shape)

# ###IBRL 1batch test###
# graph1_x = torch.rand(156,20)
# graph1_a = torch.tensor([[[0,1],[2,3]],
#                          [[0,1],[2,3]],
#                          [[0,1],[2,3]]])
# H_ = torch.ones(2,1,6)
# model = GNN(3,52,20)
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,H_)
# print(cf_out.shape,rec_out.shape,new_H.shape)

# ###CIMIS 2batch test###
# graph1_x = torch.rand(912,20)
# graph1_a = torch.tensor([[[0,1,60,61],[2,3,61,62]] for i in range(8)])
# H_ = torch.ones(2,2,16)
# model = GNN(8,57,20)
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,H_)
# print(cf_out.shape,rec_out.shape,new_H.shape)

# ###CIMIS 1batch test###
# graph1_x = torch.rand(456,20)
# graph1_a = torch.tensor([[[0,1],[2,3]] for i in range(8)])
# H_ = torch.ones(2,1,6)
# model = GNN(8,57,20)
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,H_)
# print(cf_out.shape,rec_out.shape,new_H.shape)




# import torch_geometric,torch
# import torch_geometric.nn as gnn
# import torch.nn as nn
# from torch.autograd import Variable
#
#
# class GNN(nn.Module):
#     # this version use nin as hidden instead of nout, resulting a larger model
#     def __init__(self):
#         super().__init__()
#
#         # self.wParam = Variable(torch.ones(2, 2), requires_grad=True)
#         # self.pParam = Variable(torch.ones(2, 2), requires_grad=True)
#         # self.wParam2 = Variable(torch.ones(2, 2), requires_grad=True)
#         # self.pParam2 = Variable(torch.ones(2, 2), requires_grad=True)
#         # self.wParam3 = Variable(torch.ones(2, 2), requires_grad=True)
#         # self.pParam3 = Variable(torch.ones(2, 2), requires_grad=True)
#         self.wParam = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#         self.pParam = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#         self.wParam2 = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#         self.pParam2 = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#         self.wParam3 = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#         self.pParam3 = nn.Parameter(torch.ones(2, 2), requires_grad=True)
#
#
#     def forward(self, x):
#         out = x.mm(self.wParam)+self.pParam
#         out = out.mm(self.wParam2) + self.pParam2
#         out = out.mm(self.wParam3) + self.pParam3
#         return out
#
# inp = torch.tensor([
#         # [[1,2],[3,4]],
#         [[7,8],[9,10]],
#        ])
#
# lb = torch.tensor([
#         # [[0,0],[0,2]],
#         [[12,13],[14,15]],
#        ])
#
# cr = nn.MSELoss()
# model = GNN()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.4)
# for i in range(100):
#     for ct in range(inp.shape[0]):
#         # print(inp[ct])
#         # print(lb[ct])
#         out=model(inp[ct].float())
#         print(out,lb[ct].float())
#         loss = cr(out,lb[ct].float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#
#
#
