#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.distributions as D
import torch.nn.init as init
from torch.nn import functional as F
from models.Domain_Adapter import DomainAdapter
from models.Inverible_NNs.SplineTransform import ComponentWiseSpline
from models.Network_Components.MLP import NLayerLeakyMLP
from toolss.utils import reparameterize


class Prior(nn.Module):


    def __init__(self,
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 latent_dim: int = 8,
                 domain_num: int = 20,
                 domain_dim: int = 32,
                 condition_dim = [],
                 hidden_dim: int = 64,
                 count_bins: int = 20,
                 spline_bound=10,
                 layer_num=2,
                 spline_order='linear',
                 prior_type='spline',
                 noise_dist_type='mlp',
                 base_dist_type='gaussian',
                 use_warm_start=False,
                 spline_pth='',
                 logger=None,
                 device='cpu',
                 random_sampling=True,
                 ):
        super().__init__()

        self.node = node
        self.time = time
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.base_dist_type = base_dist_type
        self.noise_dist_type = noise_dist_type

        self.device = device

        # zitk = fk(eitk; conditioner(z_{i,t-1}, z_{U(i), t-1}))
        self.interpreters = []

        self.parent_fusion = NLayerLeakyMLP(latent_dim + latent_dim, hidden_dim, 1, hidden_dim)
        self.prior_type = prior_type
        if prior_type == 'spline':
            for k in range(latent_dim):
                fk = ComponentWiseSpline(input_dim=1,
                                         count_bins=count_bins,
                                         bound=spline_bound,
                                         order=spline_order,
                                         condition=True,
                                         hidden_dim=hidden_dim,  # conditioner
                                         layer_num=layer_num,    # conditioner
                                         condition_dim=hidden_dim)  # condition

                self.interpreters.append(fk)
            self.interpreters = nn.ModuleList(self.interpreters)

            self.noise_dist_type = noise_dist_type
            self.domain_num = domain_num
            if noise_dist_type == 'mlp':
                self.noise_dist_FC = nn.Linear(domain_dim, latent_dim * 2)
        elif prior_type == 'gaussian':
            for k in range(latent_dim):
                prior_zt_FC = NLayerLeakyMLP(hidden_dim + hidden_dim + domain_dim, 2, 1, hidden_dim)
                self.interpreters.append(prior_zt_FC)
            self.interpreters = nn.ModuleList(self.interpreters)
            self.random_sampling = random_sampling
            if noise_dist_type == 'mlp':
                self.noise_dist_FC = nn.Linear(domain_dim, latent_dim * 2)


    def init_base_dist(self, domain_embedding=None):
        """
        :param domain_embedding: [R, C]
        :return:
        """
        if self.base_dist_type == 'gaussian':
            if self.noise_dist_type in ['standard']:
                noise_dist = D.MultivariateNormal(torch.zeros(self.latent_dim, device=self.device),
                                                       torch.eye(self.latent_dim, device=self.device))
                self.noise_dist = noise_dist

                return noise_dist

            elif self.noise_dist_type == 'mlp':
                dists = self.noise_dist_FC(domain_embedding)  # [R, input_dim*2]
                mus = dists[:, :self.latent_dim]
                logvars = dists[:, self.latent_dim:]
                noise_dist = D.Normal(mus, torch.exp(logvars * 0.5))
                # noise_dist = D.Normal(mus, 1)

                self.noise_dist = noise_dist

                return noise_dist

    def noise_dist_prob(self, ez, x_domain_index, x_domain_embedding, domain_embedding):
        if self.noise_dist_type == 'mlp':
            x_domain_index = x_domain_index.unsqueeze(2).unsqueeze(-1)
            prob_ez = (self.noise_dist.log_prob(ez.unsqueeze(-2)) * x_domain_index).sum(-2).sum(-1)

            return prob_ez

        elif self.noise_dist_type == 'standard':

            # [B, N, T]
            prob_ez = self.noise_dist.log_prob(ez)
            return prob_ez

    def noise_dist_sample(self, shape, x_domain_index, x_domain_embedding, domain_embedding):
        batch, node, time, latent_dim = shape
        if self.noise_dist_type == 'mlp':

            ez = self.noise_dist.rsample((batch, node, time)) * x_domain_index.unsqueeze(2).unsqueeze(-1)
            ez = ez.sum(-2)
            return ez, torch.zeros(ez.shape[:-1], device=ez.device)

        elif self.noise_dist_type == 'standard':

            ez = self.noise_dist.rsample((batch, node, time))

            return ez


    def forward(self, zs, G_intra, G_inters, pre_adj, x_domain_class, x_domain_embedding, domain_embedding):

        batch, node, time, latent_dim = zs.shape
        init_z = torch.zeros((batch, node, 1, latent_dim), device=zs.device)

        zs = torch.cat([init_z, zs], dim=2)

        # [B, N, T, K, 2]
        zs = zs.unfold(dimension=2, size=2, step=1)

        # [B, T, N, K, 2]
        zs = zs.transpose(2, 1).reshape(-1, node, latent_dim, 2)
        zts = zs[..., 1]
        zHs = zs[..., 0]

        ezs = []
        sum_log_abs_det_jacobian = 0

        for k in range(latent_dim):
            ztk = zts[..., k:k+1]  # [B*T, N, 1]

            intra = zHs * G_intra[k]  #

            inters = []
            for i in range(node):

                adj_zHx = zHs[:, torch.nonzero(pre_adj[:, i]).squeeze()]        # [B, Ui, K] * [Ui, K]
                if len(adj_zHx.shape)==2:
                    adj_zHx = adj_zHx.unsqueeze(1)
                inter = (adj_zHx * G_inters[i][k].unsqueeze(0)).sum(1, keepdim=True)
                inters.append(inter)

            inters = torch.cat(inters, dim=1)
            condition = torch.cat([intra, inters], dim=-1)
            condition = self.parent_fusion(condition)

            # zitk = fk(eitk; conditioner(z_{i,t-1}, z_{U(i), t-1}))
            # [B*T, N, 1] [B*T, N, K+Nk]  =》 [B*T, N, 1]， [B*T, N]
            etk, logabsdet = self.interpreters[k].forward(x=ztk, condition=condition)

            sum_log_abs_det_jacobian += logabsdet
            ezs.append(etk)

        ezs = torch.cat(ezs, dim=-1)
        # [B, T-1, N, K] -> [B, N, T-1, K]
        ezs = ezs.reshape(batch, -1, node, latent_dim).transpose(1, 2)
        # [B, T-1, N] -> [B, N ,T-1]
        sum_log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch, -1, node).transpose(1, 2)

        # eitk ~ P(eik|uir)
        prob_ez = self.noise_dist_prob(ezs, x_domain_class, x_domain_embedding, domain_embedding)

        prob_pz = prob_ez + sum_log_abs_det_jacobian

        # zs -> ezs -> e_standard
        # logp(z) = logp(ez) + logabsdet_ez
        # logp(z) = logp(e_standard) + logabsdet_standard + logabsdet_ez
        return ezs, prob_pz


    def inverse_last(self, shape, G_intra, G_inters, pre_adj,
                     x_domain_class, x_domain_embedding, domain_embedding,
                     ezs=None, z_last=None):

        batch, node, time, latent_dim = shape

        if ezs is None:
            shape = z_last.shape
            ezs, ezs_logabsdet = self.noise_dist_sample(shape, x_domain_class, x_domain_embedding, domain_embedding)
        else:
            ezs_logabsdet = 0

        zH = z_last[:,:, -1]             # [B, N, K]
        et = ezs[:,:, -1, :]             # [B, N, K]

        zt = []
        zt_logabsdet=0
        for k in range(latent_dim):
            etk = et[..., k:k+1]        # [B, N]

            # [B, N, K]
            intra = zH * G_intra[k]
            inters = []
            for i in range(node):
                adj_zHx = zH[:, torch.nonzero(pre_adj[:, i]).squeeze()]  # [B, Ui, K] * [Ui, K]
                if len(adj_zHx.shape)==2:
                    adj_zHx = adj_zHx.unsqueeze(1)

                inter = (adj_zHx * G_inters[i][k].unsqueeze(0)).sum(1, keepdim=True)
                inters.append(inter)

            inters = torch.cat(inters, dim=1)
            condition = torch.cat([intra, inters], dim=-1)
            condition = self.parent_fusion(condition)

            # [B, N, 1], [B, N] torch.zeros_like(condition)
            ztk, z_logabsdet = self.interpreters[k].inverse(u=etk, condition=condition)

            zt.append(ztk)
            zt_logabsdet += z_logabsdet

        # [B, N, K]
        zt = torch.cat(zt, dim=-1)
        return zt.unsqueeze(2), zt_logabsdet.unsqueeze(2)
