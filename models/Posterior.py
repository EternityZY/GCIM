#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
from torch import nn
import torch.distributions as D

from models.Network_Components.GraphGRU import GraphGRU
from models.Network_Components.MLP import NLayerLeakyMLP
from toolss.utils import reparameterize


class Posterior(nn.Module):

    def __init__(self,
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 latent_dim: int = 8,
                 domain_num: int = -1,
                 hidden_dim: int = 64,
                 pre_adj=None,
                 gcn_depth:int=1,
                 dropout_prob=0.4,
                 posterior_type='GraphGRU',
                 input_fusion=True,
                 random_sampling=True,
                 ):
        super().__init__()

        self.node = node
        self.time = time
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.pre_adj = pre_adj
        self.posterior_type = posterior_type
        self.input_fusion = input_fusion
        self.random_sampling = random_sampling


        self.input_encoder = NLayerLeakyMLP(in_features=input_dim,
                                            out_features=hidden_dim,            # hidden_dim
                                            num_layers=gcn_depth,
                                            hidden_dim=hidden_dim)

        dim = hidden_dim
        if posterior_type == 'GraphGRU':
            self.GraphGRU = GraphGRU(dim,
                                     hidden_dim,
                                     pre_adj,
                                     gcn_depth=gcn_depth,
                                     dropout_type='None',
                                     dropout_prob=dropout_prob,
                                     alpha=0.3)
        self.zt_dist = nn.Linear(hidden_dim, latent_dim*2)

    def log_qz(self, zs, mus, logvars):
        q_dist = D.Normal(mus, torch.exp(logvars / 2))  #
        log_qz = q_dist.log_prob(zs).sum(-1)

        return log_qz


    def forward(self, x):

        batch, node, time, in_dim = x.shape
        x_feature = self.input_encoder(x)

        zs = []
        mus = []
        logvars = []

        # step1: infer latent variables
        hidden_state = torch.zeros((batch, node, self.hidden_dim), device=x.device)
        for t in range(time):
            current_obs = x_feature[:, :, t, :]         # B, N, T, D

            if self.posterior_type == 'GraphGRU':
                output, hidden_state = self.GraphGRU(current_obs, hidden_state)

                dist = self.zt_dist(output)
                mu = dist[:, :, :self.latent_dim]
                logvar = dist[:, :, self.latent_dim:]
                zt = reparameterize(mu, logvar, random_sampling=self.random_sampling)

                zs.append(zt)
                mus.append(mu)
                logvars.append(logvar)

        # [B, N, T, K]
        zs = torch.stack(zs, dim=2)
        mus = torch.stack(mus, dim=2)
        logvars = torch.stack(logvars, dim=2)

        return zs, mus, logvars









