#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from tqdm import tqdm

from ModelTest_NYC import model_val, model_test
from helper_NYC import Trainer
from toolss.metrics import record

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def baseline_train(runid,
                   model,
                   dataloader,
                   device,
                   logger,
                   cfg,
                   writer=None,
                   ):
    print("start training...", flush=True)
    save_path = os.path.join('Results', cfg['data']['name'], cfg['model_name'], 'exp{:s}'.format(cfg['expid']), 'ckpt')
    scalar = dataloader['scalar']

    engine = Trainer(model=model,
                     base_lr=cfg['train']['base_lr'],
                     weight_decay=cfg['train']['weight_decay'],
                     milestones=cfg['train']['milestones'],
                     lr_decay_ratio=cfg['train']['lr_decay_ratio'],
                     min_learning_rate=cfg['train']['min_learning_rate'],
                     max_grad_norm=cfg['train']['max_grad_norm'],
                     rec=cfg['train']['rec'],
                     pred=cfg['train']['pred'],
                     num_for_target=cfg['data']['num_for_target'],
                     num_for_predict=cfg['data']['num_for_predict'],
                     loss_weight=cfg['train']['loss_weight'],
                     scaler=scalar,
                     device=device,
                     correlation=cfg['train']['correlation']
                     )

    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.model.state_dict():
        logger.info(param_tensor + '\t' + str(engine.model.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}\n'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is {:d}\n'.format(int(nParams)))

    if cfg['train']['load_initial']:
        best_mode_path = cfg['train']['best_mode']
        logger.info("loading {}".format(best_mode_path))

        save_dict = torch.load(best_mode_path)['GCEM_state_dict']
        model_dict = engine.model.state_dict()
        pretrained_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        engine.model.load_state_dict(model_dict, strict=False)
        # engine.ECGM.load_state_dict(save_dict, strict=False)
        logger.info('model load success! {}'.format(best_mode_path))

    else:
        logger.info('Start training from scratch!')
        save_dict = dict()

    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = -1
    stable_count = 0

    logger.info('begin_epoch: {}, total_epochs: {}, patient: {}, best_val_loss: {:.4f}'.
                format(begin_epoch, epochs, tolerance, best_val_loss))

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):
        """

        """
        train_total_loss = []               # total = ELBO
        train_mcc_loss = []
        train_recon_loss = []
        train_pred_loss = []
        train_kld_loss = []
        train_logpx_loss = []
        train_l1_loss = []

        train_rec_mape = {}
        train_rec_rmse = {}
        train_rec_mae = {}

        train_rec_mae['bike'] = []
        train_rec_mae['taxi'] = []

        train_rec_rmse['bike'] = []
        train_rec_rmse['taxi'] = []

        train_rec_mape['bike'] = []
        train_rec_mape['taxi'] = []

        train_pred_mape = {}
        train_pred_rmse = {}
        train_pred_mae = {}

        train_pred_mae['bike'] = []
        train_pred_mae['taxi'] = []

        train_pred_rmse['bike'] = []
        train_pred_rmse['taxi'] = []

        train_pred_mape['bike'] = []
        train_pred_mape['taxi'] = []

        t1 = time.time()

        train_dataloder = dataloader['train']
        train_tqdm_loader = tqdm(enumerate(train_dataloder))

        for iter, (x, external) in train_tqdm_loader:

            x = x.to(device)

            mcc, total_loss, \
            rec_loss, pred_loss, \
            kl_loss, log_px, \
            l1_loss, \
            rec_mae, rec_rmse, rec_mape, \
            pred_mae, pred_rmse, pred_mape ,\
            rec_output, pred_output = engine.train(x)

            train_total_loss.append(total_loss)
            train_mcc_loss.append(mcc)
            train_recon_loss.append(rec_loss)
            train_pred_loss.append(pred_loss)
            train_kld_loss.append(kl_loss)
            train_logpx_loss.append(log_px)
            train_l1_loss.append(l1_loss)

            record(train_rec_mae, train_rec_rmse, train_rec_mape, rec_mae, rec_rmse, rec_mape, NYC=True)
            record(train_pred_mae, train_pred_rmse, train_pred_mape, pred_mae, pred_rmse, pred_mape, only_last=True, NYC=True)

            # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        engine.scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        s1 = time.time()
        mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                          engine=engine,
                                                                                          dataloader=dataloader,
                                                                                          device=device,
                                                                                          logger=logger,
                                                                                          cfg=cfg,
                                                                                          epoch=epoch)
        s2 = time.time()
        val_time.append(s2 - s1)

        mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid,
                                                                                       engine=engine,
                                                                                       dataloader=dataloader,
                                                                                       device=device,
                                                                                       cfg=cfg,
                                                                                       logger=logger,
                                                                                       mode='Train')

        mtrain_total_loss = np.mean(train_total_loss)
        mtrain_mcc_loss = np.mean(train_mcc_loss)
        mtrain_recon_loss = np.mean(train_recon_loss)
        mtrain_pred_loss = np.mean(train_pred_loss)
        mtrain_kld_loss = np.mean(train_kld_loss)
        mtrain_logpx_loss = np.mean(train_logpx_loss)
        mtrain_l1_loss = np.mean(train_l1_loss)

        mtrain_rec_bike_mae = np.mean(train_rec_mae['bike'])
        mtrain_rec_bike_mape = np.mean(train_rec_mape['bike'])
        mtrain_rec_bike_rmse = np.mean(train_rec_rmse['bike'])

        mtrain_rec_taxi_mae = np.mean(train_rec_mae['taxi'])
        mtrain_rec_taxi_mape = np.mean(train_rec_mape['taxi'])
        mtrain_rec_taxi_rmse = np.mean(train_rec_rmse['taxi'])

        mtrain_pred_bike_mae = np.mean(train_pred_mae['bike'])
        mtrain_pred_bike_mape = np.mean(train_pred_mape['bike'])
        mtrain_pred_bike_rmse = np.mean(train_pred_rmse['bike'])

        mtrain_pred_taxi_mae = np.mean(train_pred_mae['taxi'])
        mtrain_pred_taxi_mape = np.mean(train_pred_mape['taxi'])
        mtrain_pred_taxi_rmse = np.mean(train_pred_rmse['taxi'])

        if (epoch - 1) % cfg['train']['print_every'] == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(epoch, (s2 - s1)))

            log = 'Epoch: {:03d}, Train MCC: {:.4f} Learning rate: {}\n' \
                  'Train Total Loss: {:.4f}\t\t\tTrain L1 Loss: {:.4f}\n' \
                  'Train Rec Loss: {:.4f}\t\t\tTrain Pred Loss: {:.4f} \n' \
                  'Train KLD Loss: {:.4f}\t\t\tTrain Log P(x): {:.4f} \n' \
                  'Train Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'

            logger.info(log.format(epoch, mtrain_mcc_loss, str(engine.scheduler.get_lr()),
                                   mtrain_total_loss, mtrain_l1_loss,
                                   mtrain_recon_loss, mtrain_pred_loss,
                                   mtrain_kld_loss, mtrain_logpx_loss,

                                   mtrain_rec_bike_mae, mtrain_rec_bike_rmse, mtrain_rec_bike_mape,
                                   mtrain_rec_taxi_mae, mtrain_rec_taxi_rmse, mtrain_rec_taxi_mape,

                                   mtrain_pred_bike_mae, mtrain_pred_bike_rmse, mtrain_pred_bike_mape,
                                   mtrain_pred_taxi_mae, mtrain_pred_taxi_rmse, mtrain_pred_taxi_mape,
                                   ))
            logger.info('\nG_inter' + str(engine.model.G_inters[0].detach().cpu()[0, 0, :]))
            logger.info('\nG_intra' + str(engine.model.G_intra.detach().cpu()[0, :]))

        his_loss.append(mvalid_pred_loss)
        if mvalid_pred_loss < best_val_loss:

            best_val_loss = mvalid_pred_loss
            epoch_best = epoch
            stable_count = 0

            save_dict.update(GCEM_state_dict=copy.deepcopy(engine.model.state_dict()),
                             epoch=epoch_best,
                             best_val_loss=best_val_loss)

            ckpt_name = "exp{:s}_epoch{:d}_val_loss:{:.2f}_mae:{:.2f}_rmse:{:.2f}_mape:{:.2f}.pth". \
                format(cfg['expid'], epoch, mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape)
            best_mode_path = os.path.join(save_path, ckpt_name)
            torch.save(save_dict, best_mode_path)
            logger.info(f'Better model at epoch {epoch_best} recorded.')
            logger.info('Best model is : {}'.format(best_mode_path))
            logger.info('\n')

        else:
            stable_count += 1
            if stable_count > tolerance:
                break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {:.4f}, epoch:{:d}".format(round(his_loss[bestid], 4), epoch_best))

    logger.info('Start the model test phase........')
    logger.info("loading the best model for this training phase {}".format(best_mode_path))
    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['GCEM_state_dict'])

    mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                      engine=engine,
                                                                                      dataloader=dataloader,
                                                                                      device=device,
                                                                                      logger=logger,
                                                                                      cfg=cfg,
                                                                                      epoch=epoch_best)

    mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid,
                                                                                   engine=engine,
                                                                                   dataloader=dataloader,
                                                                                   device=device,
                                                                                   cfg=cfg,
                                                                                   logger=logger,
                                                                                   mode='Test')

    return mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
           mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape