#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :easy_mixup.py
# @Time      :2020/9/21 下午7:26
# @Author    :ChangQing

import numpy as np
import torch


def image_mixup(image, label, alpha=1.0, use_cuda=True):
    if alpha > 0:
        # beta分布就是抛硬币a次正，b次反后，硬币正面概率的分布。np.random.beta(a,b)从分布里采样一个数。
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1.
    batch_size = image.size(0)

    if use_cuda:
        # 返回一个0到n-1的数组
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.rendperm(batch_size)
    mixed_image = lam * image + (1 - lam) * image[index,:]
    label_a,label_b = label,label[index]
    return mixed_image, label_a,label_b,lam


def criterion_mixup(label_a, label_b, lam):
    return lambda criterion, pred: lam * criterion(pred, label_a) + (1-lam) * criterion(pred, label_b)





