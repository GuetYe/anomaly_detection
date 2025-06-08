import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import CrossAttention

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
        score = F.softmax(score, dim=-1)

        return score


class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, seq_len, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)
        self.weight = nn.Parameter(torch.randn(sym_norm_Adj_matrix.shape[0],in_channels,in_channels)/(in_channels*in_channels))
        self.fusion = CrossAttention(seq_len, in_channels)

    def forward(self, x):
        spatial_attention = self.SAt(x)  # (batch, T, N, N)
        # x=x.permute(2, 0, 1)  # (T, N, M)
        # print(x.shape)
        adj=torch.mul(self.sym_norm_Adj_matrix,spatial_attention)

        x=self.fusion(x)

        x=x.permute(2, 0, 1)

        return torch.einsum("bhi,pii->bhi",torch.einsum("phh,bhi->bhi", adj, x),self.weight).permute(1, 2, 0)#F.gelu(self.Theta(torch.matmul(spatial_attention, x)).permute(1, 2, 0))
    

if __name__ == '__main__':
    sagcn=spatialAttentionGCN(torch.ones(size=(51, 51)).to("cuda"), seq_len=300, in_channels=3, out_channels=3, dropout=0.0).to("cuda")
    # sagcn=sagcn.to("cuda")
    in_data=torch.randn(size=(51, 3, 300)).to("cuda")
    print(in_data.shape)
    print(sagcn(in_data).shape)

    # sagcn=Fusion(seq_len=300)
    # in_data=torch.randn(size=(51, 300))
    # in_data1=torch.randn(size=(51, 2, 300))
    # # print(in_data.shape)
    # print(torch.matmul(torch.randn(size=(51,3,300)), torch.randn(size=(300, 300))).shape)

