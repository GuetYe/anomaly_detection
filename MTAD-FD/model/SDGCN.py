import torch
import torch.nn as nn
import torch.nn.functional as F


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        z=x.permute(2, 0, 1)
        z1 = F.normalize(z, dim=2)
        score = torch.matmul(z1, z1.permute(0, 2, 1))
        # T, N, M = score.shape    
        score = F.softmax(score, dim=-1)#self.dropout(F.softmax(score.view(T,N*M), dim=-1).view(T,N,M))  # the sum of each row is 1; (b*t, N, N)

        return score

class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.weight = nn.Parameter(torch.randn(51,3,3)/(3*3))
        self.fusion = nn.ModuleList([Fusion(seq_len=51) for _ in range(6)])

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        N, M, T = x.shape
        # spatial_attention = self.SAt(x)  # (batch, T, N, N)
        x=x.permute(2, 0, 1)  # (T, N, M)
        # # print(x.shape)
        # adj=self.sym_norm_Adj_matrix.mul(spatial_attention)
        # a=0
        # x_out = []
        # for i in range(3):
        #     k=1
        #     for j in range(3):
        #         if i==j:
        #             continue
        #         if k:
        #             f=self.fusion[a](x[:,:,j],x[:,:,i])
        #             k=0
        #         else:
        #             f=f+self.fusion[a](x[:,:,j],x[:,:,i])
        #     x_out.append(f)
        # x = torch.stack(x_out, dim=-1)
        # # print(x.shape)


        self.sym_norm_Adj_matrix = F.softmax(self.sym_norm_Adj_matrix, dim=-1)
        adj=self.sym_norm_Adj_matrix.expand(T, N, N)

        return torch.einsum("bhi,pii->bhi",torch.einsum("phh,bhi->bhi", adj, x),self.weight).permute(1, 2, 0)#F.gelu(self.Theta(torch.matmul(spatial_attention, x)).permute(1, 2, 0))
    

class Fusion(nn.Module):
    def __init__(self, seq_len):
        super(Fusion, self).__init__()
        self.proj_q1 = nn.Linear(seq_len, seq_len, bias=False)
        self.proj_k2 = nn.Linear(seq_len, seq_len, bias=False)
        self.proj_v2 = nn.Linear(seq_len, seq_len, bias=False)
        
    def forward(self, x1, x2):
        
        q1 = self.proj_q1(x1)
        k2 = self.proj_k2(x2).T
        v2 = self.proj_v2(x2)
        
        attn = torch.matmul(k2, q1)# / self.k_dim**0.5
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(v2, attn)
 
        return output



if __name__ == '__main__':
    sagcn=spatialAttentionGCN(torch.ones(size=(51, 51)), in_channels=3, out_channels=3, dropout=0.0)
    in_data=torch.randn(size=(51, 3, 300))
    print(in_data.shape)
    print(sagcn(in_data).shape)

    # sagcn=Fusion(seq_len=300)
    # in_data=torch.randn(size=(51, 300))
    # print(in_data.shape)
    # print(sagcn(in_data,in_data).shape)

