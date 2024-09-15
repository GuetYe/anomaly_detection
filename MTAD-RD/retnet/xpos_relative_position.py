# Copyright (c) 2022 Microsoft
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)
import torch
import torch.nn as nn
from util.base import mprint

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    # mprint(4, x.shape,prefix="pos emb")
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    if x.shape[-1]%2 == 1:
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
    x = torch.stack((-x2, x1), dim=-1)

    return x.flatten(-2)

def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    t = (rotate_every_two(x) * sin)
    return (x * cos[:, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]]


class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        tmp = (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)    # (head_dim / 2,)
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]                 # 张量的序列长度
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x



# test
if __name__ == "__main__":
    x = torch.eye(5).unsqueeze(0)   # 4*4的单位矩阵，再升高维度 [1,4,4]
    print(x.shape)
    xpos = XPOS(5)
    x_rot = xpos(x)                 # [1,4,4]
    # apply reverse
    x_rot_rev = xpos.forward(x)
    print(x_rot @ x_rot_rev.transpose(-1, -2))




    # x = torch.eye(4).unsqueeze(0).unsqueeze(0)   # 4*4的单位矩阵，再升高维度 [1,4,4]
    # print(x.shape)
    # xpos = XPOS(4)
    # x_rot = xpos(x)                 # [1,4,4]
    # # apply reverse
    # x_rot_rev = xpos.forward(x)
    # print(x_rot @ x_rot_rev.transpose(-1, -2))


    # t = torch.arange(1, 25)
    # print(t)
    # t = t.reshape(2, 3, 4)
    # print(t.shape)
    # print(t)
    # t = t.reshape(6, 4)
    # print(t.shape)
    # print(t)
    # t = t.reshape(2, 3, 4)
    # print(t.shape)
    # print(t)



    # s = t.shape
    # print(s)
    # print(t.size(2))
    # t = t.view(-1, t.size(2))
    # print(t.shape)
    # print(t)

    # # 还原回原来的维度
    # t = t.view(s)
    # print(t.shape)
    # print(t)

    # x  = torch.arange(1, 25)
    x  = torch.rand(2, 100, 9)
    x  = torch.arange(1, 13).reshape(2, 2, 3)
    # x  = torch.arange(1, 25).reshape(2, 3, 4)
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    if x.shape[-1]%2 == 1:
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)

    print(x.shape,x1.shape,x2.shape)
    print(x)
    print(x1)
    print(x2)
    x = torch.stack((-x2, x1), dim=-1)
    print(x.shape)
    print(x)

    r = x.flatten(-2)


