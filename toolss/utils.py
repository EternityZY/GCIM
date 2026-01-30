#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import math
import mkl_random
from scipy.sparse import linalg
from torch.optim.lr_scheduler import MultiStepLR
import colorsys
import random


class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        """

        :optimizer: TODO
        :milestones: TODO
        :gamma: TODO
        :last_epoch: TODO
        :min_lr: TODO

        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxNormalization:
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        data = 1. * (data - self.min) / (self.max - self.min)
        data = 2. * data - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1) / 2
        data = data * (self.max - self.min) + self.min
        return data


class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    D = np.sum(W, axis=1)
    D = np.diag(D ** -0.5)
    D[np.isnan(D)] = 0.
    D[np.isinf(D)] = 0.
    sym_norm_Adj_matrix = np.dot(D, W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, D)

    return sym_norm_Adj_matrix


def asym_norm_Adj(W):
    '''
    compute  normalized Adj matrix
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    D = np.diag(1.0 / np.sum(W, axis=1))  #
    D[np.isinf(D)] = 0.
    D[np.isnan(D)] = 0.
    norm_Adj_matrix = np.dot(D, W)  #

    return norm_Adj_matrix  #


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    sym_norm_Adj_matrix = np.dot(d_mat_inv_sqrt, adj)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, d_mat_inv_sqrt)
    return sym_norm_Adj_matrix.astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_inv[np.isnan(d_inv)] = 0.
    d_mat = sparse.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def kl_normal_log(mu, logvar, mu_prior, logvar_prior):
    var = logvar.exp()
    var_prior = logvar_prior.exp()

    element_wise = 0.5 * (
                torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.mean(-1)  #

    kl = torch.mean(kl, dim=1)  #
    kl = torch.mean(kl, dim=0)  #

    return kl


def kl_normal(mu, var, mu_prior, var_prior):

    element_wise = 0.5 * (
                torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.sum(-1)  # 对dim维度求和

    kl = torch.mean(kl, dim=1)  # 对node维度求平均
    kl = torch.mean(kl, dim=0)  # 对batch维度求平均
    return kl

def kl_sample(log_qz, log_pz, mode='mean'):
    '''
    KL(P||Q) = ∑x P(x)*log(P(x)/Q(x))  = ∑x P(x)*[logP(x)-logQ(x)]
    :param log_qz:
    :param log_pz:
    :return:
    '''
    if len(log_pz) ==2:
        kld = kl_normal_log(log_qz[0], log_qz[1],
                            log_pz[0], log_pz[1]).sum()
    else:
        if mode == 'sum':
            kld = (log_qz - log_pz).sum(-1).sum(-1)
        elif mode == 'mean':
            kld = (log_qz - log_pz)
    return kld.mean()

def reparameterize(mean, logvar, random_sampling=True):
    if random_sampling:
        eps = torch.randn_like(mean)
        std = torch.exp(0.5*logvar)
        # std = 1
        z = mean + eps*std
        return z
    else:
        return mean


def make_saved_dir(saved_dir, use_time=3):
    """
    :param saved_dir:
    :return: {saved_dir}/{%m-%d-%H-%M-%S}
    """
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if use_time == 1:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d_%H:%M'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
    elif use_time == 2:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

    return saved_dir


def pickle_write(file_name, data):
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()


def pickle_read(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def getDistance(point1, point2):

    lat1, lng1, lat2, lng2 = point1[0], point1[1], point2[0], point2[1]
    def rad(d):
        return d * math.pi / 180.0
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) *
                                math.cos(radLat2) *
                                pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s

if __name__ == '__main__':
    mat = np.random.randn(5, 5)
    print(sym_adj(mat))
    print(asym_norm_Adj(mat))