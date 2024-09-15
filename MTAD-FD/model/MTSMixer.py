import torch
import torch.nn as nn
from torchinfo import summary
import argparse

# 纠正：数据第二和第三维反了，更正LayerNorm为数据长度

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
        self.AAvgP = nn.AdaptiveAvgPool1d(input_dim)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling) :
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)])
        self.Avg = nn.AdaptiveAvgPool1d(input_dim)

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))
        x = self.merge(x.shape, x_samp)
        # x = self.Avg(x)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()
        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(tokens_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y)
        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y.transpose(1, 2)).transpose(1, 2)

        return y

class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        # self.dropouts = nn.ModuleList()
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-2]):
                x_out.append(self.linears[idx](x[:, idx, :]))
            x = torch.stack(x_out, dim=1)
        else: 
            x = self.linears(x)

        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C, configs.sampling, configs.norm) for _ in range(configs.e_layers)
        ])   
        self.projection = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)

    def forward(self, x):
        for block in self.mlp_blocks:
            x = block(x)
        x = self.projection(x)

        return x
    

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
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')#网络层数
    parser.add_argument('--rev', action='store_true', default=False, help='whether to apply RevIN')    

    args = parser.parse_args()
    model = MixerBlock(args.seq_len, args.enc_in, args.d_model, args.d_ff, args.fac_T, args.fac_C, args.sampling, args.norm)
    # # criterion = nn.MSELoss()
    # model = Model(args)
    # # model_optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # model.to(device)    
    # # model_optim.zero_grad()
    # # print(model)
    outputs = model(torch.ones(size=(51, 3, 300)))
    print(outputs.shape)
    # outputs = model.lin(torch.ones(size=(48, 3, 26)))
    # print(outputs.shape)
    # loss = criterion(outputs, torch.ones(size=(1, 20, 51*3)).to(device))
    # loss.backward()
    # model_optim.step()
    # summary(model, input_size=(1, 20, 51*3))
    # for blk in model:
    #     X = blk(X)
    #     print(blk.__class__.__name__, 'output shape:\t', X.shape)
    # sa=Spatial_Attention_layer(0.0)
    # in_data=torch.randn(size=(4, 3, 2))
    # print(in_data)
    # print(sa(in_data))

