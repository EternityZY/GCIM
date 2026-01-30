#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.Domain_Adapter import DomainAdapter
from models.Generator import Generator
from models.Generator import Generator
from models.Network_Components.GCN import mixpropGCN
from models.Network_Components.Gumbel_Softmax import gumbel_sigmoid
from models.Network_Components.MLP import NLayerLeakyMLP
from models.Posterior import Posterior
from models.Prior import Prior
import torch.nn.init as init

from models.Prior import Prior


class GCIM(nn.Module):

    def __init__(self,
                 # global
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 latent_dim: int = 8,
                 hidden_dim: int = 64,

                 # adapter
                 domain_num: int = -1,

                 # posterior
                 posterior_type='GraphGRU',
                 pre_adj=None,
                 pre_norm_adj=None,
                 gcn_depth=1,
                 dropout_prob=0.4,
                 input_fusion=True,
                 random_sampling=True,

                 # spline
                 spline_bin=20,
                 spline_bound=5,
                 layer_num=2,
                 spline_order='linear',

                 # prior
                 prior_type='spline',

                 # generator
                 generator_type='spline',
                 # noise
                 z_noise_dist_type='specific-domain',
                 base_dist_type='gaussian',

                 # prediction
                 prediction_type='delta_z',  # [delta_z,delta_x]

                 # pretrain
                 use_warm_start=False,
                 logger='',
                 device='cuda:0',
                 ):
        super().__init__()

        self.node = node
        self.time = time
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        ################################## Causal Graph  ##################################
        self.pre_adj = pre_adj.to(torch.int)
        self.pre_norm_adj = pre_norm_adj

        self.G_intra = nn.Parameter(torch.randn(latent_dim, latent_dim), requires_grad=True)

        condition_dim = []
        G_inters = []
        for i in range(self.node):
            Ui_num = torch.sum(self.pre_adj[:, i]>0).item()
            condition_dim.append(Ui_num*latent_dim)
            G_inters.append(nn.Parameter(torch.randn(latent_dim, Ui_num, latent_dim), requires_grad=True))
        self.G_inters = nn.ParameterList(G_inters)

        self.pre_adj = self.pre_adj.to(device)

        ################################## Domain Adapter  ##################################
        if domain_num == -1:
            self.domain_num = self.latent_dim * 2 + 5
        else:
            self.domain_num = domain_num
        self.domain_adapter = DomainAdapter(node=node,
                                            time=time,
                                            input_dim=input_dim,
                                            domain_num=domain_num,
                                            hidden_dim=hidden_dim,
                                            domain_embedding_dim=hidden_dim // 2,
                                            )

        #######################################  Posterior #####################################

        self.posterior = Posterior(node=node,
                                   time=time,
                                   input_dim=input_dim,
                                   latent_dim=latent_dim,
                                   domain_num=domain_num,
                                   hidden_dim=hidden_dim,
                                   pre_adj=self.pre_norm_adj,
                                   gcn_depth=gcn_depth,
                                   dropout_prob=dropout_prob,
                                   posterior_type=posterior_type,
                                   input_fusion=input_fusion,
                                   random_sampling=random_sampling)

        #######################################  Prior #######################################

        self.prior = Prior(node=node,
                           time=time,
                           input_dim=input_dim,
                           latent_dim=latent_dim,
                           domain_num=domain_num,
                           domain_dim=hidden_dim // 2,
                           hidden_dim=hidden_dim,
                           condition_dim=condition_dim,
                           # spline
                           count_bins=spline_bin,
                           spline_bound=spline_bound,
                           layer_num=layer_num,
                           spline_order=spline_order,

                           # type
                           prior_type=prior_type,
                           noise_dist_type=z_noise_dist_type,
                           base_dist_type=base_dist_type,
                           # pretrain
                           use_warm_start=use_warm_start,
                           logger=logger,
                           device=device,
                           )

        # self.delta_es = nn.Linear(latent_dim, latent_dim)
        self.delta_es = NLayerLeakyMLP(latent_dim, latent_dim, layer_num, hidden_dim)
        #######################################  generator #######################################

        self.generator_type = generator_type
        self.generator = Generator(node=node,
                                   time=time,
                                   input_dim=input_dim,
                                   latent_dim=latent_dim,
                                   hidden_dim=hidden_dim,
                                   gcn_depth=gcn_depth,
                                   pre_norm_adj=pre_norm_adj,
                                   # type
                                   genertor_type=generator_type,
                                   # pretrain
                                   logger=logger,
                                   device=device,
                                   )

        # self.weight_init()

    def weight_init(self):
        for m in self.modules():
            self.kaiming_init(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x_true, scalar=None, mode='pred_last'):

        batch, node, time, input_dim = x_true.shape

        ####################################### domain_adapter to get domain info  #######################################

        x_domain_class, x_domain_embedding, domain_class = self.domain_adapter(x_true)
        domain_embedding = self.domain_adapter.domain_embedding
        noise_dist = self.prior.init_base_dist(self.domain_adapter.domain_embedding)

        ####################################### initial the causal graph  #######################################
        G_intra = gumbel_sigmoid(self.G_intra, temperature=1, mode='sigmoid')

        G_inters = []
        for i in range(node):
            G_inters.append(gumbel_sigmoid(self.G_inters[i], temperature=1, mode='sigmoid'))

        ####################################### posterior to get  z_dist #######################################

        # [B, N, T, K]
        zs_est, mus_est, logvars_est = self.posterior(x_true)
        log_qz = self.posterior.log_qz(zs_est, mus_est, logvars_est)

        ####################################### prior to get log_pz #######################################
        ezs_est, log_pz = self.prior.forward(zs_est, G_intra, G_inters, self.pre_adj,
                                             x_domain_class, x_domain_embedding, domain_embedding)

        ############################### generate x with zs_est #############################
        log_px = torch.zeros_like(log_pz)

        if mode=='pred_last':

            # es_sample = self.delta_es(ezs_est[:,:,-1:])
            es_sample = None
            zs_prior_sample, _ = self.prior.inverse_last(zs_est.shape,
                                                    G_intra, G_inters, self.pre_adj,
                                                    x_domain_class, x_domain_embedding, domain_embedding,
                                                    es_sample, zs_est[:,:,-1:])

            zs_est = torch.cat([zs_est, zs_prior_sample], dim=2)
            x_est, x_next = self.generator(zs_est,
                                           x_domain_embedding,
                                           domain_embedding,
                                           )

        ########################################### return result #########################################
        return x_est, x_next, domain_class, \
               zs_est, mus_est, logvars_est, \
               log_px, log_qz, log_pz
