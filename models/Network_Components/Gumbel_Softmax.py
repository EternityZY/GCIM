#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch as th



def gumbel_sigmoid(logits, temperature=0.1, noise=False, hard=False, mode='sigmoid'):
    if noise :
        gumbel_noise = -th.log(-th.log(th.rand_like(logits)))
        logits = logits + gumbel_noise

    if mode=='sigmoid':
        y_soft = th.sigmoid(logits / temperature)
    elif mode == 'tanh':
        y_soft = th.tanh(logits)


    if hard:
        y_hard = th.where(y_soft > 0.5, 1, 0)
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft

    return y




