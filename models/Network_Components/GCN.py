# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from toolss.utils import sym_norm_Adj


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x, shape_len):
        if shape_len == 3:
            x = torch.einsum('hw, bwc->bhc', (A, x))
        elif shape_len == 4:
            x = torch.einsum('hw, bwtc->bhtc', (A, x))
        return x.contiguous()


class linear(nn.Module):
    ''''''

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out, bias)

    def forward(self, x):
        return F.leaky_relu(self.mlp(x), inplace=True)


class mixpropGCN(nn.Module):
    '''
    '''

    def __init__(self, in_dim, out_dim, gdep, dropout_prob=0, alpha=0.3, norm_adj=None):
        super(mixpropGCN, self).__init__()
        self.nconv = gconv()
        self.mlp = linear((gdep + 1) * in_dim, out_dim)
        self.gdep = gdep
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.norm_adj = norm_adj

    def forward(self, x, norm_adj=None):
        if norm_adj == None:
            norm_adj = self.norm_adj
        h = x
        out = [x]

        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(norm_adj, h, len(x.shape))
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        if self.dropout_prob > 0:
            ho = F.dropout(ho, self.dropout_prob)
        return ho

if __name__ == '__main__':

    c_in = 2
    c_out = 64
    gdep = 2
    dropout=False
    alpha=0.3
    node_num = 50

    input = torch.randn((100, node_num, c_in))

    degree = 3
    prob = float(degree) / (node_num - 1)                      #
    A = np.tril((np.random.rand(node_num, node_num) < prob).astype(float), k=-1)      #
    norm_adj = sym_norm_Adj(A)

    GCN_model = mixpropGCN(c_in, c_out, gdep, dropout, alpha)

    output = GCN_model(input, A)

    print(output)
