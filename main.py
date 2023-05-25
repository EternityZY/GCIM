#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import copy
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import time
import sys
import os

from torch.utils.tensorboard import SummaryWriter

from models.GCIM import GCIM

from config.config import get_logger
from datasets.datasets_NYC import load_dataset
from toolss.utils import sym_adj, sym_adj, pickle_read, asym_adj
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == '__main__':

    config_filename = 'config/config_NYC.yaml'
    with open(config_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    #
    parser = argparse.ArgumentParser(description='Train parameters')
    parser.add_argument('--latent_dim', type=int, default='8', help='latent_dim')
    parser.add_argument('--domain_num', type=int, default='20', help='domain_num')
    parser.add_argument('--hidden_dim', type=int, default='64', help='hidden_dim')
    parser.add_argument('--gcn_depth', type=int, default='2', help='gcn_depth')
    parser.add_argument('--expid', type=str, default='1', help='gcn_depth')
    parser.add_argument('--device', type=str, default='cuda:0', help='gcn_depth')
    args = parser.parse_args()
    # 遍历参数范围
    cfg['model']['latent_dim'] = args.latent_dim
    cfg['model']['hidden_dim'] = args.hidden_dim
    cfg['model']['domain_num'] = args.domain_num
    cfg['model']['gcn_depth'] = args.gcn_depth
    cfg['expid'] = args.expid
    cfg['device'] = args.device


    base_path = cfg['base_path']

    dataset_name = cfg['dataset_name']
    dataset_path = os.path.join(base_path, dataset_name)
    if cfg['data']['name'] == 'NYC':
        from datasets.datasets_NYC import load_dataset
        from ModelTest_NYC import baseline_test
        from ModelTrain_NYC import baseline_train

    log_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'],  'exp{:s}'.format(cfg['expid']), 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    save_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'],  'exp{:s}'.format(cfg['expid']), 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + datetime.now().strftime('%m-%d_%H:%M') + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level)

    confi_name = 'config{:s}_'.format(cfg['expid']) + datetime.now().strftime('%m-%d_%H:%M') + '.yaml'
    with open(os.path.join(log_dir, confi_name), 'w+') as _f:
        yaml.safe_dump(cfg, _f)

    logger.info(cfg)
    logger.info(dataset_path)
    logger.info(log_path)

    writer = None
    if cfg['train']['tensorboard']:
        tensorboard_path = os.path.join('Results', cfg['model_name'], 'exp{:s}'.format(cfg['expid']), 'tensorboard_log')
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)
        logger.info(tensorboard_path)

    device = torch.device(cfg['device'])

    dataloader = load_dataset(dataset_path,
                              cfg['data']['train_batch_size'],
                              cfg['data']['val_batch_size'],
                              cfg['data']['test_batch_size'],
                              logger=logger, device=device,
                              )

    pre_graph = None
    if cfg['model']['adj'] == 'adj':
        pre_graph = np.load(os.path.join(base_path, 'graph', 'geo_adj.npy')).astype(np.float32)

    elif cfg['model']['adj'] == 'affinity':
        pre_graph = np.load(os.path.join(base_path, 'graph', 'geo_affinity.npy')).astype(np.float32)

    adjs = [pre_graph]
    if cfg['data']['name'] == 'NYC':
        pre_adj = torch.tensor(pre_graph)
    else:
        pre_adj = torch.tensor(pre_graph - np.eye(cfg['model']['node']))

    if cfg['model']['norm_graph'] == 'sym':
        static_norm_adjs = [torch.tensor(sym_adj(adj)).to(device) for adj in adjs]
    elif cfg['model']['norm_graph'] == 'asym':
        static_norm_adjs = [torch.tensor(asym_adj(adj)).to(device) for adj in adjs]
    else:
        static_norm_adjs = [torch.tensor(adj).to(device) for adj in adjs]

    if cfg['data']['external']:
        external = pd.read_hdf(os.path.join(base_path, 'poi.h5')).values

    if cfg['model']['init_intra_graph']:
        intra_graph = np.load(os.path.join(base_path, 'G_intra.npy')).astype(np.float32)
        intra_graph = torch.tensor(intra_graph).to(device)

    if cfg['model']['init_inter_graph']:
        inters_graph = pickle_read(os.path.join(base_path, 'G_inters.pkl'))
        for i in range(len(inters_graph)):
            inters_graph[i] = inters_graph[i].to(device)

    model_name = cfg['model_name']

    val_loss_list = []
    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []

    test_loss_list = []
    test_mae_list = []
    test_mape_list = []
    test_rmse_list = []
    for runid in range(cfg['runs']):
        if cfg['model_name']=='GCIM':
            GCIM_model = GCIM(  # global
                node=cfg['model']['node'],
                time=cfg['model']['time'],
                input_dim=cfg['model']['input_dim'],
                latent_dim=cfg['model']['latent_dim'],
                hidden_dim=cfg['model']['hidden_dim'],

                # adapter
                domain_num=cfg['model']['domain_num'],

                # posterior
                posterior_type=cfg['model']['posterior_type'],
                pre_adj=pre_adj,
                pre_norm_adj=static_norm_adjs[0],

                gcn_depth=cfg['model']['gcn_depth'],
                dropout_prob=cfg['model']['dropout_prob'],
                input_fusion=cfg['model']['input_fusion'],
                random_sampling=cfg['model']['random_sampling'],

                # spline
                spline_bin=cfg['model']['spline_bin'],
                spline_bound=cfg['model']['spline_bound'],
                layer_num=cfg['model']['layer_num'],
                spline_order=cfg['model']['spline_order'],

                # prior
                prior_type=cfg['model']['prior_type'],

                # generator
                generator_type=cfg['model']['generator_type'],

                # noise
                z_noise_dist_type=cfg['model']['z_noise_dist_type'],
                base_dist_type=cfg['model']['base_dist_type'],

                # prediction
                prediction_type=cfg['model']['prediction_type'],  # [prior,delta]

                # pretrain
                use_warm_start=False,
                logger=logger,
                device=device)

        logger.info(model_name)

        if cfg['test_only']:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_test(runid,
                                                                                               GCIM_model,
                                                                                               dataloader,
                                                                                               device,
                                                                                               logger,
                                                                                               cfg,
                                                                                               writer,
                                                                                               )
        else:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_train(runid,
                                                                                                GCIM_model,
                                                                                                dataloader,
                                                                                                device,
                                                                                                logger,
                                                                                                cfg,
                                                                                                writer, )
        val_loss_list.append(mvalid_total_loss)
        val_mae_list.append(mvalid_pred_mae)
        val_mape_list.append(mvalid_pred_mape)
        val_rmse_list.append(mvalid_pred_rmse)

        test_loss_list.append(mtest_total_loss)
        test_mae_list.append(mtest_pred_mae)
        test_mape_list.append(mtest_pred_mape)
        test_rmse_list.append(mtest_pred_rmse)

    test_loss_list = np.array(test_loss_list)
    test_mae_list = np.array(test_mae_list)
    test_mape_list = np.array(test_mape_list)
    test_rmse_list = np.array(test_rmse_list)

    aloss = np.mean(test_loss_list, 0)
    amae = np.mean(test_mae_list, 0)
    amape = np.mean(test_mape_list, 0)
    armse = np.mean(test_rmse_list, 0)

    sloss = np.std(test_loss_list, 0)
    smae = np.std(test_mae_list, 0)
    smape = np.std(test_mape_list, 0)
    srmse = np.std(test_rmse_list, 0)

    logger.info('valid\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.mean(val_loss_list), np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(val_loss_list), np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list)))
    logger.info('\n\n')

    logger.info('Test\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.mean(test_loss_list), np.mean(test_mae_list), np.mean(test_rmse_list), np.mean(test_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(
        log.format(np.std(test_loss_list), np.std(test_mae_list), np.std(test_rmse_list), np.mean(test_mape_list)))
    logger.info('\n\n')
