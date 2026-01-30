#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
from models.Network_Components.GraphGRU import GraphGRU

class Generator(nn.Module):


    def __init__(self,
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 latent_dim: int = 8,
                 hidden_dim: int = 64,
                 gcn_depth: int = 2,
                 pre_norm_adj = None,
                 # type
                 genertor_type='',
                 base_dist_type='gaussian',
                 logger=None,
                 device='cuda:0',
                 ):
        super().__init__()

        self.node = node
        self.time = time
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.generator_type = genertor_type
        self.device = device

        # self.dec_attention = AttentionBlock(hidden_dim, num_heads=4)
        if genertor_type == 'GraphGRU':
            self.GraphGRU = GraphGRU(latent_dim,
                                     hidden_dim,
                                     pre_norm_adj,
                                     gcn_depth=gcn_depth,
                                     dropout_type='None',
                                     dropout_prob=0,
                                     alpha=0.3)
            self.recon_FC = nn.Linear(hidden_dim, input_dim)
            self.pred_FC = nn.Linear(hidden_dim, input_dim)


    def forward(self, zs, x_domain_embedding, domain_embedding,
                          delta=None):
        batch, node, time, latent_dim = zs.shape
        hidden_state = torch.zeros((batch, node, self.hidden_dim), device=zs.device)

        x_recon = []
        for t in range(time):
            zt = zs[:, :, t]
            given = zt
            output, hidden_state = self.GraphGRU(given, hidden_state)
            xt = self.recon_FC(output)
            if t == 6:
                xt = self.pred_FC(output)
            x_recon.append(xt)

        x_recons = torch.stack(x_recon[:-1], dim=2)
        x_next = x_recon[-1].unsqueeze(2)
        return x_recons, x_next

