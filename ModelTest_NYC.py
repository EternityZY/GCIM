#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import numpy as np
from tqdm import tqdm
from helper_NYC import Trainer
import os
import torch.nn.functional as F
from toolss.metrics import record, metric
from toolss.utils import pickle_write


def model_val(runid, engine, dataloader, device, logger, cfg, epoch):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_total_loss = []  # total = ELBO
    valid_mcc_list = []
    valid_recon_list = []
    valid_pred_list = []
    valid_kld_list = []
    valid_logpx_list = []
    valid_l1_list = []

    valid_rec_mape = {}
    valid_rec_rmse = {}
    valid_rec_mae = {}

    valid_rec_mae['bike'] = []
    valid_rec_mae['taxi'] = []


    valid_rec_rmse['bike'] = []
    valid_rec_rmse['taxi'] = []


    valid_rec_mape['bike'] = []
    valid_rec_mape['taxi'] = []

    valid_pred_mape = {}
    valid_pred_rmse = {}
    valid_pred_mae = {}

    valid_pred_mae['bike'] = []
    valid_pred_mae['taxi'] = []

    valid_pred_rmse['bike'] = []
    valid_pred_rmse['taxi'] = []

    valid_pred_mape['bike'] = []
    valid_pred_mape['taxi'] = []


    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (x, external) in val_tqdm_loader:
        # if iter >3:
        #     break

        x = x.to(device)

        mcc, total_loss, \
        rec_loss, pred_loss, \
        kl_loss, log_px, \
        l1_loss, \
        rec_mae, rec_rmse, rec_mape, \
        pred_mae, pred_rmse, pred_mape, \
        rec_output, pred_output= engine.eval(x)

        valid_total_loss.append(total_loss)
        valid_mcc_list.append(mcc)
        valid_recon_list.append(rec_loss)
        valid_pred_list.append(pred_loss)
        valid_kld_list.append(kl_loss)
        valid_logpx_list.append(log_px)
        valid_l1_list.append(l1_loss)

        record(valid_rec_mae, valid_rec_rmse, valid_rec_mape, rec_mae, rec_rmse, rec_mape, NYC=True)
        record(valid_pred_mae, valid_pred_rmse, valid_pred_mape, pred_mae, pred_rmse, pred_mape, only_last=True, NYC=True)

    mvalid_total_loss = np.mean(valid_total_loss)
    mvalid_mcc_list = np.mean(valid_mcc_list)
    mvalid_recon_list = np.mean(valid_recon_list)
    mvalid_pred_list = np.mean(valid_pred_list)
    mvalid_kld_list = np.mean(valid_kld_list)
    mvalid_logpx_list = np.mean(valid_logpx_list)
    mvalid_l1_list = np.mean(valid_l1_list)

    mvalid_rec_bike_mae = np.mean(valid_rec_mae['bike'])
    mvalid_rec_bike_mape = np.mean(valid_rec_mape['bike'])
    mvalid_rec_bike_rmse = np.mean(valid_rec_rmse['bike'])

    mvalid_rec_taxi_mae = np.mean(valid_rec_mae['taxi'])
    mvalid_rec_taxi_mape = np.mean(valid_rec_mape['taxi'])
    mvalid_rec_taxi_rmse = np.mean(valid_rec_rmse['taxi'])


    mvalid_pred_bike_mae = np.mean(valid_pred_mae['bike'])
    mvalid_pred_bike_mape = np.mean(valid_pred_mape['bike'])
    mvalid_pred_bike_rmse = np.mean(valid_pred_rmse['bike'])

    mvalid_pred_taxi_mae = np.mean(valid_pred_mae['taxi'])
    mvalid_pred_taxi_mape = np.mean(valid_pred_mape['taxi'])
    mvalid_pred_taxi_rmse = np.mean(valid_pred_rmse['taxi'])


    log = 'Epoch: {:03d}, Valid MCC: {:.4f}\n' \
          'Valid Total Loss: {:.4f}\t\t\tValid L1 Loss: {:.4f}\n' \
          'Valid Rec Loss: {:.4f}\t\t\tValid Pred Loss: {:.4f} \n' \
          'Valid KLD Loss: {:.4f}\t\t\tValid Log P(x): {:.4f} \n' \
          'Valid Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Valid Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Valid Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Valid Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'

    logger.info(log.format(epoch, mvalid_mcc_list,
                           mvalid_total_loss, mvalid_l1_list,
                           mvalid_recon_list, mvalid_pred_list,
                           mvalid_kld_list, mvalid_logpx_list,

                           mvalid_rec_bike_mae, mvalid_rec_bike_rmse, mvalid_rec_bike_mape,
                           mvalid_rec_taxi_mae, mvalid_rec_taxi_rmse, mvalid_rec_taxi_mape,

                           mvalid_pred_bike_mae, mvalid_pred_bike_rmse, mvalid_pred_bike_mape,
                           mvalid_pred_taxi_mae, mvalid_pred_taxi_rmse, mvalid_pred_taxi_mape,

                           ))

    return mvalid_total_loss, mvalid_pred_bike_mae, mvalid_pred_bike_rmse, mvalid_pred_bike_mape,


def model_test(runid, engine, dataloader, device, logger, cfg, mode='Test'):
    logger.info('Start testing phase.....')

    test_dataloder = dataloader['test']

    test_total_loss = []  # total = ELBO
    test_mcc_list = []
    test_recon_list = []
    test_pred_list = []
    test_kld_list = []
    test_logpx_list = []
    test_l1_list = []

    test_rec_mape = {}
    test_rec_rmse = {}
    test_rec_mae = {}

    test_rec_mae['bike'] = []
    test_rec_mae['taxi'] = []

    test_rec_rmse['bike'] = []
    test_rec_rmse['taxi'] = []

    test_rec_mape['bike'] = []
    test_rec_mape['taxi'] = []


    test_pred_mape = {}
    test_pred_rmse = {}
    test_pred_mae = {}

    test_pred_mae['bike'] = []
    test_pred_mae['taxi'] = []

    test_pred_rmse['bike'] = []
    test_pred_rmse['taxi'] = []

    test_pred_mape['bike'] = []
    test_pred_mape['taxi'] = []

    test_outputs_list = []
    test_metrics_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))
    for iter, (x, external) in test_tqdm_loader:
        # if iter>3:
        #     break

        x = x.to(device)

        mcc, total_loss, \
        rec_loss, pred_loss, \
        kl_loss, log_px, \
        l1_loss, \
        rec_mae, rec_rmse, rec_mape, \
        pred_mae, pred_rmse, pred_mape,\
        rec_output, pred_output = engine.eval(x)

        test_total_loss.append(total_loss)
        test_mcc_list.append(mcc)
        test_recon_list.append(rec_loss)
        test_pred_list.append(pred_loss)
        test_kld_list.append(kl_loss)
        test_logpx_list.append(log_px)
        test_l1_list.append(l1_loss)

        test_outputs_list.append(pred_output)
        test_metrics_list.append(metric(pred_output[:,:,-1:], x[:,:,-1:]))

        record(test_rec_mae, test_rec_rmse, test_rec_mape, rec_mae, rec_rmse, rec_mape, NYC=True)
        record(test_pred_mae, test_pred_rmse, test_pred_mape, pred_mae, pred_rmse, pred_mape, only_last=True, NYC=True)

    mtest_total_loss = np.mean(test_total_loss)
    mtest_mcc_list = np.mean(test_mcc_list)
    mtest_recon_list = np.mean(test_recon_list)
    mtest_pred_list = np.mean(test_pred_list)
    mtest_kld_list = np.mean(test_kld_list)
    mtest_logpx_list = np.mean(test_logpx_list)
    mtest_l1_list = np.mean(test_l1_list)

    test_metrics_list = np.array(test_metrics_list)
    mtest_metrics_list = np.mean(test_metrics_list, axis=0)

    mtest_rec_bike_mae = np.mean(test_rec_mae['bike'])
    mtest_rec_bike_mape = np.mean(test_rec_mape['bike'])
    mtest_rec_bike_rmse = np.mean(test_rec_rmse['bike'])

    mtest_rec_taxi_mae = np.mean(test_rec_mae['taxi'])
    mtest_rec_taxi_mape = np.mean(test_rec_mape['taxi'])
    mtest_rec_taxi_rmse = np.mean(test_rec_rmse['taxi'])


    mtest_pred_bike_mae = np.mean(test_pred_mae['bike'])
    mtest_pred_bike_mape = np.mean(test_pred_mape['bike'])
    mtest_pred_bike_rmse = np.mean(test_pred_rmse['bike'])

    mtest_pred_taxi_mae = np.mean(test_pred_mae['taxi'])
    mtest_pred_taxi_mape = np.mean(test_pred_mape['taxi'])
    mtest_pred_taxi_rmse = np.mean(test_pred_rmse['taxi'])


    log = 'Test MCC: {:.4f}\n' \
          'Test Total Loss: {:.4f}\t\t\tTest L1 Loss: {:.4f}\n' \
          'Test Rec Loss: {:.4f}\t\t\tTest Pred Loss: {:.4f} \n' \
          'Test KLD Loss: {:.4f}\t\t\tTest Log P(x): {:.4f} \n' \
          'Test Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Total MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(mtest_mcc_list,
                           mtest_total_loss, mtest_l1_list,
                           mtest_recon_list, mtest_pred_list,
                           mtest_kld_list, mtest_logpx_list,

                           mtest_rec_bike_mae, mtest_rec_bike_rmse, mtest_rec_bike_mape,
                           mtest_rec_taxi_mae, mtest_rec_taxi_rmse, mtest_rec_taxi_mape,

                           mtest_pred_bike_mae, mtest_pred_bike_rmse, mtest_pred_bike_mape,
                           mtest_pred_taxi_mae, mtest_pred_taxi_rmse, mtest_pred_taxi_mape,

                           mtest_metrics_list[0], mtest_metrics_list[2], mtest_metrics_list[1],
                           ))

    predicts = torch.cat(test_outputs_list, dim=0)

    if mode == 'Test':
        pred_all = predicts.cpu()
        path_save_pred = os.path.join('Results', cfg['data']['name'], cfg['model_name'], 'exp{:s}'.format(cfg['expid']), 'result_pred')
        if not os.path.exists(path_save_pred):
            os.makedirs(path_save_pred, exist_ok=True)

        name = 'exp{:s}_Test_Loss:{:.4f}_mae:{:.4f}_rmse:{:.4f}_mape:{:.4f}'. \
            format(cfg['model_name'], mtest_pred_list, mtest_metrics_list[0], mtest_metrics_list[2], mtest_metrics_list[1])
        path = os.path.join(path_save_pred, name)
        np.save(path, pred_all)
        logger.info('result of prediction has been saved, path: {}'.format(path))
        logger.info('shape: ' + str(pred_all.shape))
        if cfg['train']['visual_graph']:
            intre = engine.model.G_intra.detach().cpu().numpy()[..., 1]
            inters = engine.model.G_inters
            for i in range(len(inters)):
                inters[i] = inters[i].detach().cpu().numpy()[..., 1]

            domain_embedding = engine.model.domain_adapter.domain_embedding.detach().cpu().numpy()

            name_intra = 'exp{:s}_intra_Test_Loss:{:.4f}_mae:{:.4f}'.format(cfg['model_name'], mtest_pred_list, mtest_metrics_list[0])
            name_inters = 'exp{:s}_inters_Test_Loss:{:.4f}_mae:{:.4f}'.format(cfg['model_name'], mtest_pred_list, mtest_metrics_list[0])
            name_domain = 'exp{:s}_domain_Test_Loss:{:.4f}_mae:{:.4f}'.format(cfg['model_name'], mtest_pred_list, mtest_metrics_list[0])

            np.save(os.path.join(path_save_pred, name_intra), intre)
            pickle_write(os.path.join(path_save_pred, name_inters), inters)
            np.save(os.path.join(path_save_pred, name_domain), domain_embedding)

    return mtest_pred_list, mtest_metrics_list[0], mtest_metrics_list[2], mtest_metrics_list[1]

def baseline_test(runid,
                  model,
                  dataloader,
                  device,
                  logger,
                  cfg,
                  writer=None):

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

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path)
    engine.model.load_state_dict(save_dict['GCEM_state_dict'])
    logger.info('model load success! {}\n'.format(best_mode_path))

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

    mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid, engine, dataloader, device,
                                                                                    logger, cfg, mode='Test')

    return mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape, \
           mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape


if __name__ == '__main__':
    l = [[1,2,3], [6,4,8]]
    l = np.array(l)
    print(np.mean(l, axis=0))
