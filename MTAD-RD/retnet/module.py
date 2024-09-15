import torch.nn as nn
import argparse
from retnet.retnet import RetNetBlock,RetNet
from retnet.discriminatorNet import DiscriminatorNet
from util.base import mprint
class MTAD_RD(nn.Module):
    """ 网络模型 """
    def __init__(self,opt:argparse.Namespace):
        super(MTAD_RD, self).__init__()
        self.retnet = RetNet(hidden_size=opt.hidden_sizes,sequence_len=opt.windows, double_v_dim=opt.double_v_dim)
        self.discriminatorNet = DiscriminatorNet(layers=opt.b_layers, ins_node_feature=opt.retnet_output_dim + 2, ins_node_size=opt.node_sizes)

    def feature_extraction(self):
        return self.retnet

    def forward(self, X, A, L):
        Y_retnet = self.retnet(X,A)
        Y_ins, Y_dis, Ni= self.discriminatorNet.forward_batch(Y_retnet, L)
        return Y_ins, Y_dis, Ni
