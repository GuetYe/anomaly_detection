import torch_geometric,torch
import torch_geometric.nn as gnn
import torch.nn as nn
from torch.autograd import Variable


class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, modenum,nodenum,slid_win):
        super().__init__()
        self.modenum = modenum
        self.nodenum = nodenum
        self.slid_win = slid_win
        self.emb_size = 4
        self.gat_head = 6
        self.gru_emb_size = 16
        self.use_bias = True
        self.gnns1f = nn.ModuleList([
            gnn.GATConv(slid_win, 32, heads=self.gat_head, bias= self.use_bias)
              for i in range(modenum)])
        self.gnns2f = nn.ModuleList([
            gnn.GATConv(32 * self.gat_head, self.emb_size, bias= self.use_bias)
              for i in range(modenum)])

        self.catParam = nn.ParameterList([
            nn.Parameter(torch.ones(2 * self.emb_size, 2 * self.emb_size), requires_grad=True)
            for i in range(modenum)
        ])
        # self.catParam = nn.ModuleList([nn.Parameter(torch.ones(2 * self.emb_size,2 * self.emb_size))
        #                                for i in range(modenum)])
        self.regnn1f = nn.ModuleList([
            gnn.GATConv(slid_win, 32, heads=self.gat_head, bias= self.use_bias)
              for i in range(modenum)])
        self.regnn2f = nn.ModuleList([
            gnn.GATConv(32*self.gat_head, slid_win, bias= self.use_bias)
              for i in range(modenum)])
        self.gru = nn.GRU(2 * self.emb_size, self.gru_emb_size, num_layers=2,batch_first=True)
        self.cf_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nodenum * self.gru_emb_size, 512, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(512, 64, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(64, 2, bias= self.use_bias),
            nn.Sigmoid()
        )
        self.node_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(slid_win,256, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(256,32, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(32,self.emb_size, bias= self.use_bias)
        )
        self.mid_linear = nn.Sequential(
            nn.Linear(2 * self.emb_size, 128, bias= self.use_bias), nn.ReLU(),
            nn.Linear(128, 128, bias= self.use_bias), nn.ReLU(),
            nn.Linear(128, slid_win, bias= self.use_bias)
        )

    def forward(self, x,adj, H_):
        # print(x.shape,  self.nodenum,self.modenum,self.slid_win)
        batch = int(x.shape[0] / (self.nodenum * self.modenum))
        # print(batch)
        for mode_num in range(self.modenum):
            # print(mode_num * self.nodenum * batch, mode_num * self.nodenum * batch
            #       + self.nodenum * batch)
            mode_tensor = x[mode_num * self.nodenum * batch: mode_num * self.nodenum * batch
                                                             + self.nodenum * batch]
            # print(mode_tensor.shape)
            gnnout = self.gnns1f[mode_num](mode_tensor, adj[mode_num])
            # print(gnnout.shape)
            gnnout = self.gnns2f[mode_num](gnnout, adj[mode_num])
            # print(gnnout.shape)
            lineout = self.node_feature(mode_tensor)
            gnn_line_cat = torch.cat([gnnout,lineout],dim=1)
            # print(gnn_line_cat.shape)
            if mode_num == 0:
                cat_tensor = gnn_line_cat.mm(self.catParam[mode_num])
            else:
                cat_tensor = cat_tensor + gnn_line_cat.mm(self.catParam[mode_num])
        # print(cat_tensor.shape)
        mid_result = cat_tensor.reshape(batch, self.nodenum, cat_tensor.shape[1])
        # print(mid_result.shape)
        gru_out, new_H = self.gru(mid_result, H_)
        # print(gru_out.shape)
        cf_out = self.cf_linear(gru_out)
        # print(cf_out.shape)

        re_line = self.mid_linear(cat_tensor)
        # print(re_line.shape)

        for mode_num in range(self.modenum):
            recon = self.regnn1f[mode_num](re_line, adj[mode_num])
            # print(recon.shape)
            recon = self.regnn2f[mode_num](recon, adj[mode_num])
            # print(recon.shape)
            if mode_num == 0:
                rec_out = recon
            else:
                rec_out = torch.cat([rec_out,recon],dim=0)
        # print(rec_out.shape)
        return cf_out,rec_out,new_H


###IBRL 2batch test###
# graph1_x = torch.rand(312,20)
# graph1_a = torch.tensor([[[0,1,60,61],[2,3,61,62]],
#                          [[0,1,60,61],[2,3,61,62]],
#                          [[0,1,60,61],[2,3,61,62]]])
# H_ = torch.ones(2,2,6)
# model = GNN(3,52,20)
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,H_)
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
# H_ = torch.ones(2,2,6)
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

