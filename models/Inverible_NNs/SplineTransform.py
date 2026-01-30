#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import (Tuple)
import torch.distributions as D
from models.Inverible_NNs.Invertible_Framework import Transform
from models.Inverible_NNs.conditioner import DenseNN
from models.Inverible_NNs.splines import _monotonic_rational_spline
from models.Network_Components.MLP import NLayerLeakyMLP


class ComponentWiseSpline(Transform):

    def __init__(
            self,
            input_dim: int,
            count_bins: int = 8,
            bound: int = 3.,
            order: str = 'linear',
            condition: bool = False,
            hidden_dim: int = 64,
            layer_num: int = 1,
            condition_dim: int = 64,
    ) -> None:
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            count_bins: The number of bins that each can have their own weights.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order

        Modified from Neural Spline Flows: https://arxiv.org/pdf/1906.04032.pdf
        """
        super(ComponentWiseSpline, self).__init__()
        assert order in ("linear", "quadratic")
        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.condition = condition
        self.condition_dim = condition_dim

        if self.condition == False:

            self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.count_bins))  #
            self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
            self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.count_bins - 1))

            # Rational linear splines have additional lambda parameters
            if self.order == "linear":
                self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.count_bins))
        else:
            if self.order == 'linear':
                self.condition_nn = DenseNN(
                                    self.condition_dim,
                                    [hidden_dim] * layer_num,
                                    param_dims=[
                                        input_dim * count_bins,
                                        input_dim * count_bins,
                                        input_dim * (count_bins - 1),
                                        input_dim * count_bins,
                                    ],
                                )
            else:
                self.condition_nn = DenseNN(
                                    self.condition_dim,
                                    [hidden_dim] * layer_num,
                                    param_dims=[
                                        input_dim * count_bins,
                                        input_dim * count_bins,
                                        input_dim * (count_bins - 1),
                                    ],
                                )

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_var', torch.eye(input_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, condition=None) -> Tuple[torch.Tensor, torch.Tensor]:

        """f: data x -> latent u"""
        if self.condition == True:
            self.conditioner(condition)
        u, log_detJ = self.spline_op(x)
        log_detJ = torch.sum(log_detJ, dim=-1)
        return u, log_detJ

    def inverse(self, u, condition=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """g: latent u > data x  """
        if self.condition == True:
            self.conditioner(condition)
        x, log_detJ = self.spline_op(u, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=-1)
        return x, log_detJ  #

    def conditioner(self, context):
        if self.order == "linear":
            w, h, d, l = self.condition_nn(context)

            # 调整输出参数维度
            if w.shape[-1] == self.input_dim:
                l = l.transpose(-1, -2)
            else:
                l = l.reshape(l.shape[:-1] + (self.input_dim, self.count_bins))
        else:
            w, h, d = self.conditon_nn(context)
            l = None

        #
        if w.shape[-1] == self.input_dim:
            w = w.transpose(-1, -2)
            h = h.transpose(-1, -2)
            d = d.transpose(-1, -2)
        else:
            w = w.reshape(w.shape[:-1] + (self.input_dim, self.count_bins))
            h = h.reshape(h.shape[:-1] + (self.input_dim, self.count_bins))
            d = d.reshape(d.shape[:-1] + (self.input_dim, self.count_bins - 1))

        self.unnormalized_widths = w
        self.unnormalized_heights = h
        self.unnormalized_derivatives = d

        # Rational linear splines have additional lambda parameters
        #
        if self.order == "linear":
            self.unnormalized_lambdas = l


    def spline_op(
            self,
            x: torch.Tensor,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)


        if self.order == 'linear':
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None

        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)  # bound=5
        return y, log_detJ

    def log_prob(self, x):

        z, log_detJ = self.forward(x)  #
        logp = self.base_dist.log_prob(z) + log_detJ  #
        return logp


if __name__ == '__main__':
    x_true = torch.randn((32, 50, 12, 1))
    condition = torch.randn((32, 50, 12, 64))

    base_dist = D.MultivariateNormal(torch.zeros(1), torch.eye(1))

    g = ComponentWiseSpline(input_dim=1,
                            count_bins=128,
                            bound=10,
                            condition=True,
                            layer_num=2,
                            condition_dim=64)

    ex_true = base_dist.rsample(x_true.shape[:-1])

    print(base_dist.log_prob(ex_true).sum(-1).sum(-1).mean())

    obs_est, logabsdet_f = g.inverse(ex_true, condition)

    ex_est, logabsdet_i = g.forward(obs_est, condition)

    prob1 = base_dist.log_prob(ex_true)
    prob2 = base_dist.log_prob(ex_est)

    print(torch.sum((ex_true - ex_est).abs()))
    print(torch.sum(logabsdet_f + logabsdet_i))
    print(torch.sum(prob1 - prob2))

    ex_est, logabsdet_i = g.forward(x_true, condition)

    obs_est, logabsdet_f = g.inverse(ex_est, condition)

    prob1 = base_dist.log_prob(ex_true)
    prob2 = base_dist.log_prob(ex_est)

    print(torch.sum(x_true - obs_est))
    print(torch.sum(logabsdet_f + logabsdet_i))
    print(torch.sum(prob1 - prob2))