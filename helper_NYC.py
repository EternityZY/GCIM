#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from toolss.metrics import masked_mae_torch, masked_mape_torch, masked_rmse_torch, metric, metric_all, compute_mcc
from toolss.utils import StepLR2, kl_sample


class Trainer():
    def __init__(self,
                 model,
                 base_lr,
                 weight_decay,
                 milestones,
                 lr_decay_ratio,
                 min_learning_rate,
                 max_grad_norm,
                 num_for_target,
                 num_for_predict,
                 scaler,
                 device,
                 loss_weight,
                 rec=True,
                 pred=True,
                 correlation='Pearson',
                 train_mode='pred_last'
                 ):
        self.scaler = scaler
        self.model = model

        self.device = device
        self.max_grad_norm = max_grad_norm
        self.loss_weight = loss_weight
        self.train_mode = train_mode

        self.rec = rec
        self.pred = pred
        self.correlation = correlation

        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)

        self.scheduler = StepLR2(optimizer=self.optimizer,
                                 milestones=milestones,
                                 gamma=lr_decay_ratio,
                                 min_lr=min_learning_rate)

        self.SmoothL1loss = nn.SmoothL1Loss(reduction='mean')
        self.scaler = scaler
        self.num_for_target = num_for_target
        self.num_for_predict = num_for_predict

    def train(self, x, z=None, domain=None):
        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        self.model.train()
        # with torch.autograd.set_detect_anomaly(True):
        self.optimizer.zero_grad()

        x_est, x_next, domain_class, \
        zs_est, mus_est, logvars_est, \
        log_px, log_qz, log_pz = self.model(self.scaler.transform(input_x))

        mcc, total_loss, \
        rec_loss, pred_loss, \
        kl_loss, log_px, \
        l1_loss, \
        rec_mae, rec_rmse, rec_mape, \
        pred_mae, pred_rmse, pred_mape, \
        rec_output, pred_output = self.get_loss(x, z, x_est, x_next,
                                                zs_est, mus_est, logvars_est,
                                                log_px, log_qz, log_pz)

        total_loss.backward(retain_graph=True)

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return mcc.item(), total_loss.item(), \
               rec_loss.item(), pred_loss.item(), \
               kl_loss.item(), log_px.item(), \
               l1_loss.item(), \
               rec_mae, rec_rmse, rec_mape, \
               pred_mae, pred_rmse, pred_mape, \
               rec_output, pred_output

    def eval(self, x, z=None, domain=None):
        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        self.model.eval()
        with torch.no_grad():
            x_est, x_next, domain_class, \
            zs_est, mus_est, logvars_est, \
            log_px, log_qz, log_pz = self.model(self.scaler.transform(input_x))

            mcc, total_loss, \
            rec_loss, pred_loss, \
            kl_loss, log_px, \
            l1_loss, \
            rec_mae, rec_rmse, rec_mape, \
            pred_mae, pred_rmse, pred_mape, \
            rec_output, pred_output = self.get_loss(x, z, x_est, x_next,
                                                    zs_est, mus_est, logvars_est,
                                                    log_px, log_qz, log_pz)

        return mcc.item(), total_loss.item(), \
               rec_loss.item(), pred_loss.item(), \
               kl_loss.item(), log_px.item(), \
               l1_loss.item(), \
               rec_mae, rec_rmse, rec_mape, \
               pred_mae, pred_rmse, pred_mape, \
               rec_output, pred_output

    def get_loss(self, x, z,
                 x_est, x_next,
                 zs_est, mus_est, logvars_est,
                 log_px, log_qz, log_pz):

        input_x = x[:, :, :self.num_for_predict]
        batch, node, time, input_dim = input_x.shape

        ####################################### MCC  #######################################
        mcc = torch.zeros((1))
        if z is not None:
            zt_recon = mus_est.view(batch * node * time,
                                    -1).T.detach().cpu().numpy()
            zt_true = z.view(batch * node * time, -1).T.detach().cpu().numpy()
            mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        ####################################### reconstruction loss  #######################################

        rec_loss = torch.zeros((1), device=x.device)
        rec_mae, rec_rmse, rec_mape = 0, 0, 0
        rec_output = torch.zeros_like(x[:, :, :self.num_for_predict])
        if self.rec:
            rec_x = self.scaler.inverse_transform(x_est)
            rec_loss = self.SmoothL1loss(rec_x, input_x)
            rec_mae, rec_rmse, rec_mape = metric_all(
                [rec_x[..., 0:2], rec_x[..., 2:4]],
                [x[:, :, :self.num_for_predict, 0:2],
                 x[:, :, :self.num_for_predict, 2:4]],
                NYC=True
            )

            rec_output = rec_x

        ####################################### prediction loss  #######################################
        pred_loss = torch.zeros((1), device=x.device)
        pred_mae, pred_rmse, pred_mape = 0, 0, 0
        pred_output = torch.zeros_like(x[:, :, -1:])

        if self.pred:
            x_next = self.scaler.inverse_transform(x_next)
            if self.train_mode == 'pred_last':
                pred_loss = self.SmoothL1loss(x_next[:, :, -1:], x[:, :, -1:])
                pred_mae, pred_rmse, pred_mape = metric_all(
                    [x_next[:, :, -1:, 0:2], x_next[:, :, -1:, 2:4]],
                    [x[:, :, -1:, 0:2],
                     x[:, :, -1:, 2:4], ],
                    NYC=True
                )
                pred_output = x_next
            elif self.train_mode == 'pred_all':
                pred_loss = self.SmoothL1loss(x_next, x[:, :, 1:])
                pred_mae, pred_rmse, pred_mape = metric_all(
                        [x_next[:, :, :, 0:2], x_next[:, :, :, 2:4]],
                        [x[:, :, 1:, 0:2],
                         x[:, :, 1:, 2:4]], NYC=True)

                pred_output = x_next

        ####################################### KLD loss  #######################################

        kl_loss = kl_sample(log_qz, log_pz)
        log_px = log_px.mean()

        ####################################### L1 loss  #######################################
        diversity_loss = self.model.domain_adapter.embedding_constrains()
        l1_loss = diversity_loss

        if self.model.generator_type == 'spline':
            total_loss = self.loss_weight[1] * pred_loss + \
                         self.loss_weight[2] * kl_loss - self.loss_weight[3] * log_px + \
                         self.loss_weight[4] * l1_loss
        elif self.model.generator_type in ['mlp', 'GraphGRU', ]:
            total_loss = self.loss_weight[0] * rec_loss + self.loss_weight[1] * pred_loss + \
                         self.loss_weight[2] * kl_loss + self.loss_weight[3] * log_px + \
                         self.loss_weight[4] * diversity_loss

        return mcc, total_loss, \
               rec_loss, pred_loss, \
               kl_loss, diversity_loss, \
               log_px, \
               rec_mae, rec_rmse, rec_mape, \
               pred_mae, pred_rmse, pred_mape, \
               rec_output, pred_output
