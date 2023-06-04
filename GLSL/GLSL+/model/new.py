import torch_geometric,torch
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=False, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid,
                                     n_hid if i<nlayer-1 else nout,
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x

class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, modenum,nodenum,slid_win,base_gnn):
        super().__init__()
        self.modenum = modenum
        self.nodenum = nodenum
        self.slid_win = slid_win
        self.emb_size = 4
        self.gat_head = 6
        self.gru_emb_size = 16
        self.use_bias = True
        self.base_gnn = base_gnn
        if base_gnn == "GAT":
            self.gnns1f = nn.ModuleList([
                gnn.GATConv(slid_win, 32, heads=self.gat_head, bias= self.use_bias)
                  for i in range(modenum)])
            self.gnns2f = nn.ModuleList([
                gnn.GATConv(32 * self.gat_head, self.emb_size, bias= self.use_bias)
                  for i in range(modenum)])
            self.regnn1f = nn.ModuleList([
                gnn.GATConv(slid_win, 32, heads=self.gat_head, bias=self.use_bias)
                for i in range(modenum)])
            self.regnn2f = nn.ModuleList([
                gnn.GATConv(32 * self.gat_head, slid_win, bias=self.use_bias)
                for i in range(modenum)])
        elif base_gnn == "GCN":
            self.gnns1f = nn.ModuleList([
                gnn.GCNConv(slid_win, 32 * self.gat_head, bias= self.use_bias)
                  for i in range(modenum)])
            self.gnns2f = nn.ModuleList([
                gnn.GCNConv(32 * self.gat_head, self.emb_size, bias= self.use_bias)
                  for i in range(modenum)])
            self.regnn1f = nn.ModuleList([
                gnn.GCNConv(slid_win, 32 * self.gat_head, bias=self.use_bias)
                for i in range(modenum)])
            self.regnn2f = nn.ModuleList([
                gnn.GCNConv(32 * self.gat_head, slid_win, bias=self.use_bias)
                for i in range(modenum)])
        elif base_gnn == "GIN":
            self.gnns1f = nn.ModuleList([
                gnn.GINConv(MLP(slid_win, 32 * self.gat_head, 2, False, bias=self.use_bias))
                  for i in range(modenum)])
            self.gnns2f = nn.ModuleList([
                gnn.GINConv(MLP(32 * self.gat_head, self.emb_size, 2, False, bias=self.use_bias))
                  for i in range(modenum)])
            self.regnn1f = nn.ModuleList([
                gnn.GINConv(MLP(slid_win, 32 * self.gat_head, 2, False, bias=self.use_bias))
                for i in range(modenum)])
            self.regnn2f = nn.ModuleList([
                gnn.GINConv(MLP(32 * self.gat_head, slid_win, 2, False, bias=self.use_bias))
                for i in range(modenum)])
        elif base_gnn == "SuperGAT":
            self.gnns1f = nn.ModuleList([
                gnn.SuperGATConv(slid_win, 32, heads=self.gat_head, bias= self.use_bias)
                  for i in range(modenum)])
            self.gnns2f = nn.ModuleList([
                gnn.SuperGATConv(32 * self.gat_head, self.emb_size, bias= self.use_bias)
                  for i in range(modenum)])
            self.regnn1f = nn.ModuleList([
                gnn.SuperGATConv(slid_win, 32, heads=self.gat_head, bias=self.use_bias)
                for i in range(modenum)])
            self.regnn2f = nn.ModuleList([
                gnn.SuperGATConv(32 * self.gat_head, slid_win, bias=self.use_bias)
                for i in range(modenum)])
        self.catParam = nn.ParameterList([
            nn.Parameter(torch.ones(2 * self.emb_size, 2 * self.emb_size), requires_grad=True)
            for i in range(modenum)
        ])
        self.combineParam = nn.ParameterList([nn.Parameter(torch.ones(1))
                                       for i in range(2)])


        self.gru = nn.GRU(2 * self.emb_size, self.gru_emb_size, num_layers=2,batch_first=True)
        self.cf_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nodenum * self.gru_emb_size, 256, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(256, 64, bias= self.use_bias),
            nn.ReLU(),
            nn.Linear(64, 2, bias= self.use_bias),
        )
        self.node_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(slid_win,128, bias= self.use_bias),nn.ReLU(),
            nn.Linear(128,self.emb_size, bias= self.use_bias)
        )
        self.re_node_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(slid_win,128, bias= self.use_bias),nn.ReLU(),
            nn.Linear(128,slid_win, bias= self.use_bias)
        )
        self.mid_linear = nn.Sequential(
            nn.Linear(2 * self.emb_size, 128, bias= self.use_bias), nn.ReLU(),
            nn.Linear(128, slid_win, bias= self.use_bias)
        )
        self.readout1f = nn.Linear(slid_win, 1)
        self.readout2f = nn.Sequential(
            nn.Linear(modenum * nodenum, 2),
        )
        self.rec_linear = nn.Linear(slid_win * 2, slid_win)

    def forward(self, x,adj, wei, H_):
        # print(x.shape,  self.nodenum,self.modenum,self.slid_win)
        batch = int(x.shape[0] / (self.nodenum * self.modenum))
        # print(batch)
        for mode_num in range(self.modenum):
            # print(mode_num * self.nodenum * batch, mode_num * self.nodenum * batch
            #       + self.nodenum * batch)
            mode_tensor = x[mode_num * self.nodenum * batch: mode_num * self.nodenum * batch
                                                             + self.nodenum * batch]
            # print(mode_tensor.shape)
            if self.base_gnn == "GAT":
                gnnout = self.gnns1f[mode_num](mode_tensor, adj[mode_num])
                # print(gnnout.shape, adj[mode_num].shape, att[mode_num].shape)
                gnnout = self.gnns2f[mode_num](gnnout, adj[mode_num])
            elif self.base_gnn == "GCN":
                gnnout = self.gnns1f[mode_num](mode_tensor, edge_index=adj[mode_num], edge_weight=wei[mode_num])
                # print(gnnout.shape, adj[mode_num].shape, att[mode_num].shape)
                gnnout = self.gnns2f[mode_num](gnnout, edge_index=adj[mode_num], edge_weight=wei[mode_num])
            elif self.base_gnn == "GIN":
                gnnout = self.gnns1f[mode_num](mode_tensor, edge_index=adj[mode_num])
                # print(gnnout.shape, adj[mode_num].shape, att[mode_num].shape)
                gnnout = self.gnns2f[mode_num](gnnout, edge_index=adj[mode_num])
            elif self.base_gnn == "SuperGAT":
                gnnout = self.gnns1f[mode_num](mode_tensor, adj[mode_num])
                # print(gnnout.shape, adj[mode_num].shape, att[mode_num].shape)
                gnnout = self.gnns2f[mode_num](gnnout, adj[mode_num])
            # print(gnnout.shape)
            lineout = self.node_feature(mode_tensor)
            # print(lineout.shape)
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
        cf_res = self.cf_linear(gru_out)
        # print(cf_out.shape)

        re_line = self.mid_linear(cat_tensor)
        # print("reline",re_line.shape)

        for mode_num in range(self.modenum):
            if self.base_gnn == "GAT":
                regnn = self.regnn1f[mode_num](re_line, adj[mode_num])
                # print(recon.shape)
                regnn = self.regnn2f[mode_num](regnn, adj[mode_num])
            elif  self.base_gnn == "GCN":
                regnn = self.regnn1f[mode_num](re_line, edge_index=adj[mode_num], edge_weight=wei[mode_num])
                # print(recon.shape)
                regnn = self.regnn2f[mode_num](regnn, edge_index=adj[mode_num], edge_weight=wei[mode_num])
            # print(recon.shape)
            elif  self.base_gnn == "GIN":
                regnn = self.regnn1f[mode_num](re_line, edge_index=adj[mode_num])
                # print(recon.shape)
                regnn = self.regnn2f[mode_num](regnn, edge_index=adj[mode_num])
            # print(recon.shape)
            elif self.base_gnn == "SuperGAT":
                regnn = self.regnn1f[mode_num](re_line, adj[mode_num])
                # print(recon.shape)
                regnn = self.regnn2f[mode_num](regnn, adj[mode_num])
            cat_regnn_relink = torch.cat([regnn , self.re_node_feature(re_line)],dim=1)
            if mode_num == 0:
                rec_out = cat_regnn_relink
            else:
                rec_out = torch.cat([rec_out,cat_regnn_relink],dim=0)
        # print(rec_out.shape)
        rec_out = self.rec_linear(rec_out)
        readout = self.readout1f(rec_out)
        readout = readout.reshape(int(readout.shape[0] / (self.modenum*self.nodenum)),self.modenum*self.nodenum)
        readout = self.readout2f(readout)
        cf_out = torch.sigmoid(cf_res * self.combineParam[0] + readout * self.combineParam[1])
        # print(cf_out.shape)
        return cf_out,rec_out,new_H

###IBRL 2batch test###
# graph1_x = torch.rand(312,20)
# graph1_a = torch.tensor([[[0,1,60,61],[2,3,61,62]],
#                          [[0,1,60,61],[2,3,61,62]],
#                          [[0,1,60,61],[2,3,61,62]]])
# graph1_at = torch.tensor([[1.5,2.0,2.0,2.0],
#                           [1.5,2.0,2.0,2.0],
#                           [1.5,2.0,2.0,2.0],])
# H_ = torch.ones(2,2,16)
# model = GNN(3,52,20,"SuperGAT")
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,graph1_at,H_)
# print(cf_out.shape,rec_out.shape,new_H.shape)

# graph1_x = torch.rand(255,20)
# graph1_a = torch.tensor([[[0,1,18,19],[2,3,19,20]],
#                          [[0,1,18,19],[2,3,19,20]],
#                          [[0,1,18,19],[2,3,19,20]]])
# graph1_at = torch.tensor([[1.5,2.0,2.0,2.0],
#                           [1.5,2.0,2.0,2.0],
#                           [1.5,2.0,2.0,2.0],])
# H_ = torch.ones(2,5,16)
# model = GNN(3,16,20,"SuperGAT")
# cf_out,rec_out,new_H = model(graph1_x,graph1_a,graph1_at,H_)
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

