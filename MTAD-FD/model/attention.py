import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, fou_len):
        super(FourierBlock, self).__init__()
        # print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # self.fc1 = nn.Linear(seq_len, seq_len)

        # self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(
        #     self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        self.scale = (1 / (fou_len * fou_len))
        # self.weights = nn.Parameter(torch.ones(3, 150, 150)/(150*150))
        # self.weights1 = nn.Parameter(self.scale * torch.ones((3, 76, 76), dtype=torch.cfloat))
        # self.weight = nn.Parameter(torch.ones(3, 150, 150)/(150*150))
        self.weight1 = nn.Parameter(self.scale * torch.ones((3, fou_len, fou_len), dtype=torch.cfloat))
        # self.max_pool = nn.MaxPool1d(3,stride=2, return_indices=True) 
        # self.unpool = nn.MaxUnpool1d(3, stride=2)
        self.w_k = nn.Parameter(self.scale * torch.ones((3, fou_len, fou_len), dtype=torch.cfloat))
        self.w_q = nn.Parameter(self.scale * torch.ones((3, fou_len, fou_len), dtype=torch.cfloat))
        self.w_v = nn.Parameter(self.scale * torch.ones((3, fou_len, fou_len), dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,pio->bho", input, weights)#"bhi,pii->bhi"

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=2)
        kx_ft = self.compl_mul1d(x_ft, self.w_k)
        qx_ft = self.compl_mul1d(x_ft, self.w_q)
        vx_ft = self.compl_mul1d(x_ft, self.w_v)

        kx_ft = kx_ft.transpose(1, 2)
        qk = torch.matmul(qx_ft, kx_ft)
        # qk = torch.einsum("phi,bih->phh", qx_ft, kx_ft)

        # v = torch.fft.irfft(vx_ft, n=x.size(-1))
        a = F.softmax(abs(qk), dim=-1)
        a = torch.complex(a, torch.zeros_like(a))
        y = torch.matmul(a, vx_ft)
        y = torch.fft.irfft(y, n=x.size(-1))

        # # xq_ft = torch.fft.rfft(xq, dim=2)

        # # size = [B, L, H, E]
        # # N, M, L = q.shape
        # # x = self.fc1(q)
        # # x = self.compl_mul1d(q, self.weights)
        # # xq = self.compl_mul1d(q, self.weight)
        # # x = q.permute(0, 2, 3, 1)
        # # Compute Fourier coefficients
        # x_ft = torch.fft.rfft(x, dim=2)
        # # xq_ft = torch.fft.rfft(xq, dim=2)
        # # Perform Fourier neural operations
        # # out_ft = torch.zeros(N, M, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        # # print(x_ft[:, :, self.index])
        # out_ft=self.compl_mul1d(x_ft, self.weight1)
        # # output, indices = self.max_pool(abs(xq_ft))
        # # print(indices)
        # # # print(in_ft.shape)
        # # for wi, i in enumerate(self.index):
        # #     # print(wi,i)
        # #     out_ft[:, :, i] = in_ft[:, :, wi]
        # # # Return to time domain
        # y = torch.fft.irfft(out_ft, n=x.size(-1))
        return y
    

# ########## fourier layer #############
class TemporalBlock(nn.Module):
    def __init__(self, tem_len):
        super(TemporalBlock, self).__init__()
        # print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.scale = (1 / (tem_len * tem_len))

        self.w_k = nn.Parameter(self.scale * torch.ones((3, tem_len, tem_len), dtype=torch.float32))
        self.w_q = nn.Parameter(self.scale * torch.ones((3, tem_len, tem_len), dtype=torch.float32))
        self.w_v = nn.Parameter(self.scale * torch.ones((3, tem_len, tem_len), dtype=torch.float32))


    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,pio->bho", input, weights)#"bhi,pii->bhi"

    def forward(self, x):

        kx = self.compl_mul1d(x, self.w_k)
        qx = self.compl_mul1d(x, self.w_q)
        vx = self.compl_mul1d(x, self.w_v)

        kx = kx.transpose(1, 2)
        qk = torch.matmul(qx, kx)

        a = F.softmax(qk, dim=-1)
        y = torch.matmul(a, vx)

        return y


class CrossAttention(nn.Module):
    def __init__(self, seq_len1, seq_len2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(seq_len1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(seq_len2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(seq_len2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, seq_len1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_m1, seq_len1 = x1.size()
        seq_m2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_m1, self.num_heads, self.k_dim)#.permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_m2, self.num_heads, self.k_dim).permute(0, 1, 3, 2)#.permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_m2, self.num_heads, self.v_dim)#.permute(0, 2, 1, 3)
        
        attn = torch.matmul(k2, q1) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(v2, attn).view(batch_size, seq_m1, -1)#.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_m1, -1)
        output = self.proj_o(output)
        
        return output

    
# ########## fourier layer #############
class FourierAttention(nn.Module):
    def __init__(self, seq_len):
        super(FourierAttention, self).__init__()
        # print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.fc1 = nn.Linear(seq_len, seq_len)
        self.fc = nn.Linear(seq_len, seq_len)

        self.max_pool = nn.MaxPool1d(3,stride=2, return_indices=True) 
        self.unpool = nn.MaxUnpool1d(3, stride=2)

        self.scale = (1 / (76 * 76))
        # self.weights = nn.Parameter(self.scale * torch.ones((3, 26, 26), dtype=torch.float32))
        self.fq = nn.Linear(seq_len//4, seq_len//4)
        self.weights_q = nn.Parameter(torch.ones(3, 150, 150)/(150*150))
        self.weights_v = nn.Parameter(torch.ones(3, 150, 150)/(150*150))
        self.weights_o = nn.Parameter(torch.ones(3, 37, 37)/(37*37))
        self.weights1 = nn.Parameter(self.scale * torch.ones((3, 76, 76), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,pio->bho", input, weights)

    def forward(self, q):
        # size = [B, L, H, E]
        N, M, L = q.shape
        xq = self.compl_mul1d(q, self.weights_q)#self.fc(q)
        xv = self.compl_mul1d(q, self.weights_v)#self.fc1(q)
        xq_ft = torch.fft.rfft(xq, dim=2)
        xv_ft = torch.fft.rfft(xv, dim=2)

        output, indices = self.max_pool(abs(xq_ft))
        output = self.compl_mul1d(output, self.weights_o)
        # unp = self.unpool(output, indices)
        att=F.pad(self.unpool(output, indices), (0, 1))
        att = torch.softmax(att, dim=-1)
        # att = torch.complex(att, torch.zeros_like(att))
        x_ft=torch.mul(att, xv_ft)
        x_ft = torch.softmax(abs(x_ft), dim=-1)
        x_ft = torch.complex(x_ft, torch.zeros_like(x_ft))
        # Perform Fourier neural operations
        out_ft=self.compl_mul1d(x_ft, self.weights1)

        # Return to time domain
        x = torch.fft.irfft(out_ft, n=q.size(-1))
        return x

if __name__=='__main__':
    # f=FourierBlock(seq_len=50, modes=10, mode_select_method='random')
    # print(f(torch.ones(size=(48, 3, 50))).shape)

    fa=FourierAttention(seq_len=50)
    print(fa(torch.ones(size=(51, 3, 50))).shape)    

    # pool = nn.MaxPool1d(2, stride=2, return_indices=True)
    # unpool = nn.MaxUnpool1d(2, stride=2)
    # input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
    # output, indices = pool(input)
    
    # print(torch.take(input, indices))
    # print(output, indices)
    # print(unpool(output, indices))


