#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :eql_loss.py
# @Time     :2020/9/28 上午10:13
# @Author   :Chang Qing
import torch
import torch.nn as nn
import torch.functional as F

NUM_CLASSES = 4
LVIS_CATEGORIES = {
    "0": 0.2,
    "1": 0.2,
    "2": 0.4,
    "3": 0.2
}




class EqualizationLoss(nn.Module):

    def __init__(self,pred_class_logits,gt_classes,lambda_= 0.3):
        super(EqualizationLoss, self).__init__()
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        self.lambda_ = lambda_
        self.n_i, self.n_c = self.pred_class_logits.size()
        self.freq_info = self.get_image_count_frequency(gt_classes)


    def exclude_func(self):
        # E(r) 的实现
        # instance-level weight
        bg_ind = self.n_c
        # 对背景类别置为0，非背景类别置为1
        weight = (self.gt_classes != bg_ind).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # T(x) 的实现
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        # 对于小于 lambda_的置为1，其他为0
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def get_image_count_frequency(batch_labels):

        image_count_frequency = [None] * NUM_CLASSES
        for idx, lebel in enumerate(batch_labels):
            image_count_frequency[idx] = LVIS_CATEGORIES[str(label)]
        return torch.FloatTensor(image_count_frequency)

    def eql_loss(self):
        """
        分类 Loss 不计算任何非 ground-truth 类别中设定为稀有类别的部分。总之，这样就保证了背景不处理，前景类中，
        rare类只有它的正样本传播loss，head类（多的类）的正样本（对于rare来说是负样本）不对rare类计算loss，也就减小了对rare类的压制。
        这样一来，对于rare类来说，EQL 既忽略了rare类的负样本，又没有忽略背景类的负梯度。
        :return:
        """
        # eql loss的实现
        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)

        # wj的实现
        # 检测问题
        # eql_w = 1 - self.exclude_func() * self.threshold_func() * (1 - target)
        # 分类问题
        eql_w = 1 - self.threshold_func() * (1 - target)


        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                      reduction='none')
        return torch.sum(cls_loss * eql_w) / self.n_i

