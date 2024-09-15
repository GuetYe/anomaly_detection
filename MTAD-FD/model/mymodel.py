import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from model.MTSMixer import MixerBlock, ChannelProjection
from model.SDGCN import spatialAttentionGCN
from model.attention import FourierBlock, CrossAttention, FourierAttention,TemporalBlock
import argparse
import math
from thop import profile
from thop import clever_format


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x

class myFormer(nn.Module):
    def __init__(self, configs, adj):
        super().__init__()
        self.Embed=TokenEmbedding(3,3)
        self.dwt = DWT1DForward(wave='db1', J=1)
        self.idwt = DWT1DInverse(wave='db1')
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C, configs.sampling, configs.norm) for _ in range(configs.e_layers)
        ])
        self.tre_sagcn=nn.ModuleList([spatialAttentionGCN(adj, in_channels=3, out_channels=3, dropout=0.0) for _ in range(configs.e_layers)])
        self.tre_project = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        self.tre_lin = nn.ModuleList([nn.Linear(configs.seq_len, configs.seq_len) for _ in range(configs.e_layers)])
        self.norm = nn.LayerNorm(configs.seq_len) if configs.norm else None
        self.fou_att = nn.ModuleList([FourierBlock(fou_len=int(configs.seq_len/2)+1) for _ in range(configs.e_layers)])
        self.cro_att=nn.ModuleList([CrossAttention(seq_len1=configs.seq_len, seq_len2=configs.seq_len, k_dim=50, v_dim=50, num_heads=5) for _ in range(configs.e_layers)])
        self.sea_sagcn=nn.ModuleList([spatialAttentionGCN(adj, in_channels=3, out_channels=3, dropout=0.0) for _ in range(configs.e_layers)])
        self.sea_lin = nn.ModuleList([nn.Linear(configs.seq_len, configs.seq_len) for _ in range(configs.e_layers)])
        self.sea_project = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        self.projection = ChannelProjection(configs.seq_len*2, configs.pred_len*2, configs.enc_in, configs.individual)
        # self.sigmoid = nn.Sigmoid()
        # # self.arbiter=nn.Linear(3, 1)
        # # self.weights1 = nn.Parameter(torch.randn(size))
        # self.tr_project = nn.Parameter(torch.randn(3, 150,150)/(150*150))
        # self.tr_project1 = nn.Parameter(torch.randn(3,150))

    def forward(self, x):

        trend, season = self.dwt(x)
        
        for mlps,t_sagcn,t_lin in zip(self.mlp_blocks,self.tre_sagcn,self.tre_lin):
            trend = mlps(trend)
            trend = t_sagcn(trend)
            trend = t_lin(trend)
        # trend = torch.einsum("bhi,pii->bhi", trend, self.tr_project)#+self.tr_project1
        # trend = torch.einsum("bhi,hio->bho", trend.permute(2, 0, 1), self.tr_project).permute(1, 2, 0)
        # print(self.tre_project.weight)
        # trend = self.tre_project(trend)

        y = season[0]
        y=self.Embed(y)
        # y = self.norm(season[0])
        for f_att,c_att,s_sagcn,s_lin in zip(self.fou_att,self.cro_att,self.sea_sagcn,self.sea_lin):
            y = y+f_att(y)
            y = self.norm(y)
            y = y+s_sagcn(y)
            y = s_lin(y)
            y = self.norm(y)
            # y = lin(y)
            # y = y+torch.einsum("bhi,pii->bhi", y, self.tr_project)
            # y = self.norm(y)
        # y = self.sea_project(y)
        season = [y]
        idwt_data = self.idwt((trend, season))
        idwt_data=self.projection(idwt_data)
        # # idwt_data = self.arbiter(idwt_data.permute(2, 0, 1))
        # # idwt_data = self.sigmoid(idwt_data)#.permute(1, 2, 0)

        return idwt_data#torch.squeeze(idwt_data)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2))

    def forward(self, x):
        # padding on the both ends of time series
        # front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        # end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        # x = torch.cat([front, x, end], dim=1)
        x = self.avg(x)#.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class meltFormer(nn.Module):
    def __init__(self, configs, adj):
        super().__init__()
        self.Embed=TokenEmbedding(3,3)
        self.dwt = DWT1DForward(wave='db1', J=1)
        self.idwt = DWT1DInverse(wave='db1')
        self.decomp = series_decomp(kernel_size=31)
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C, configs.sampling, configs.norm) for _ in range(configs.e_layers)
        ])
        self.tre_sagcn=nn.ModuleList([spatialAttentionGCN(adj, in_channels=3, out_channels=3, dropout=0.0) for _ in range(configs.e_layers)])
        # self.tre_project = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        self.tre_lin = nn.ModuleList([nn.Linear(configs.seq_len, configs.seq_len) for _ in range(configs.e_layers)])
        self.norm = nn.LayerNorm(configs.seq_len) if configs.norm else None
        # self.fou_att = nn.ModuleList([FourierBlock(fou_len=int(configs.seq_len/2)+1) for _ in range(configs.e_layers)])
        self.tem_att = nn.ModuleList([TemporalBlock(tem_len=configs.seq_len) for _ in range(configs.e_layers)])
        # self.cro_att=nn.ModuleList([CrossAttention(seq_len1=configs.seq_len, seq_len2=configs.seq_len, k_dim=50, v_dim=50, num_heads=5) for _ in range(configs.e_layers)])
        self.sea_sagcn=nn.ModuleList([spatialAttentionGCN(adj, in_channels=3, out_channels=3, dropout=0.0) for _ in range(configs.e_layers)])
        self.sea_lin = nn.ModuleList([nn.Linear(configs.seq_len, configs.seq_len) for _ in range(configs.e_layers)])
        # self.sea_project = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        self.projection = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)


    def forward(self, x):

        # trend, season = self.dwt(x)
        season, trend = self.decomp(x)
        
        for mlps,t_sagcn,t_lin in zip(self.mlp_blocks,self.tre_sagcn,self.tre_lin):
            trend = mlps(trend)
            trend = t_sagcn(trend)
            trend = t_lin(trend)

        # y = season[0]
        y = season
        y=self.Embed(y)
        for t_att,s_sagcn,s_lin in zip(self.tem_att,self.sea_sagcn,self.sea_lin):
            y = y+t_att(y)
            y = self.norm(y)
            y = y+s_sagcn(y)
            y = s_lin(y)
            y = self.norm(y)

        # season = [y]
        season = y
        # idwt_data = self.idwt((trend, season))
        idwt_data = season + trend
        idwt_data=self.projection(idwt_data)

        return idwt_data


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
    parser.add_argument('--individual', action='store_true', default=True, help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')#网络层数
    parser.add_argument('--rev', action='store_true', default=False, help='whether to apply RevIN')

    args = parser.parse_args()
    # former=myFormer(args, torch.ones(size=(51, 51)))
    former=meltFormer(args, torch.ones(size=(51, 51)))
    # output=former(torch.randn(size=(51, 3, 300)))
    # print(output.shape)
    # 使用thop分析模型的运算量和参数量
    input = torch.randn(size=(51, 3, 300))  # 随机生成一个输入张量，这个尺寸应该与模型输入的尺寸相匹配
    flops, params = profile(former, inputs=(input,))

    # 将结果转换为更易于阅读的格式
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")


    # data_points = torch.tensor([
    #     [[1, 2, 3, 4, 5],
    #      [6, 7, 8, 9, 10]],

    #     [[11, 12, 13, 14, 15],
    #      [16, 17, 18, 19, 20]]
    # ], dtype=torch.float32)

    # # self.decomp = series_decomp(kernel_size)
    # # seasonal_enc, trend_enc = self.decomp(x_enc)
    # mov_avg = moving_avg(kernel_size=3, stride=1)
    # output=mov_avg(data_points)
    # print(output)

