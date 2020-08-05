#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :validation.py
# @Author   :omega
# @Time     :2020/8/4 0:23

import os
import torch
import collections

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def get_hard_examples(model, dataloader, num_classes, device):
    '''
    统计各个类别中得分最小的图片
    :param model: pron or unpron
    :param dataloader:
    :param num_classes: number of classes,
    :return: a .csv to summary_path
    '''
    if num_classes == 4:
        name_classes = ["血腥恶心", "宗教服饰", "正常", "集体"]
        name_classes = ["blood", "religion", "normal", "group"]
    elif num_classes == 11:
        name_classes = []

    prd_probas = []
    img_dirs = []
    img_labels = []
    img_dict = {}

    for image_tensors, labels, img_dics, _ in dataloader:
        # 将标签tensor转换为tensor
        labels_list = labels.tolist()
        img_labels.extend(labels_list)
        img_dirs.extend(img_dics)

        images = image_tensors.to(device)  # 数据转化为batch
        labels = labels.to(device)
        model_results = model(images)
        # 获取模型输出类别
        outputs = torch.softmax(model_results, dim=1)
        outputs = outputs.tolist()
        for i in range(len(labels_list)):
            prd_probas.append(outputs[i][labels_list[i]])

    img_dict["img_labels"] = img_labels
    img_dict["prd_probas"] = prd_probas
    img_dict["img_dirs"] = img_dirs

    df = pd.DataFrame(img_dict)

    grouped = df.groupby(by="img_labels")

    summay_path = "/home/changqing/workspaces/Overseas_classification-master/EfficientNet_Simple/summary/unpron/train/"
    for name, group in grouped:
        group = group.sort_values(by="prd_probas")
        if not os.path.exists(summay_path):
            os.makedirs(summay_path)
        group.to_csv(summay_path + "{}.csv".format(name_classes[name]))


def validation(model, val_dataset, num_classes, epoch, mode, device):
    '''
    统计验证集上的精确度与auc得分
    :param model:
    :param val_dataset:
    :param num_classes:
    :param epoch:
    :param mode:
    :param device:
    :return:
    '''
    count = 0
    eq_count = 0
    l = []
    l_preds = []
    l_labels = []
    err_img = []
    all_imgs = []
    for image_tensors, labels, img_dics, _ in val_dataset:
        # 将标签tensor转换为tensor
        labels_list = labels.tolist()
        all_imgs.extend(labels_list)

        images = image_tensors.to(device)  # 数据转化为batch
        labels = labels.to(device)
        model_results = model(images)
        # 获取模型输出类别
        outputs = torch.softmax(model_results, dim=1)
        outputs_label = torch.argmax(outputs, dim=1)

        # count来统计验证图片数量
        count += outputs_label.data.numel()
        # eq_count统计所有匹配正确的图片数量
        eq = torch.eq(labels, outputs_label)

        outputs_label = outputs_label.tolist()
        eq = eq.cpu().tolist()
        eq_count += sum(eq)

        outputs_list = outputs.cpu().tolist()
        for j in range(len(eq)):
            l_preds.append(outputs_list[j])
            l_label = [0] * len(outputs_list[j])
            l_label[labels_list[j]] = 1
            l_labels.append(l_label)
            if eq[j] == 0:
                l.append(labels_list[j])
                err_img.append(img_dics[j] + '\t' + str(labels_list[j]) + '\t' + str(outputs_label[j]))

    # with open("val_error_file.txt","w") as f:
    #     for i in range(len(err_img)):
    #         print(err_img[i])
    #         print(err_img[i],file=f)

    imgnum_per_cls = collections.Counter(all_imgs)
    errnum_per_cls = collections.Counter(l)
    assert mode in ["train", "val"], print("mode is necessary")
    if num_classes == 4:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "unpron")
        name_classes = ["blood", "religion", "normal", "group"]
    elif num_classes == 11:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pron")
        name_classes = [["class-" + str(i) for i in range(num_classes)]]

    if mode == "val":
        result_file = os.path.join(results_dir, "valresult.txt")
    else:
        result_file = os.path.join(results_dir, "trainresult.txt")

    with open(result_file, "a+") as f:
        print(('=' * 50) + "epoch:" + str(epoch) + ("=" * 50), file=f)
        print("各类图片数量： ", imgnum_per_cls, file=f)
        print("各类识别错误的图片量： ", errnum_per_cls, file=f)
        count_acc = 0
        for i in range(num_classes):
            tem_acc = (imgnum_per_cls[i] - errnum_per_cls[i]) / imgnum_per_cls[i]
            print("accuracy of {} :{:.2f}%-{}/{}".format(
                name_classes[i],
                float(tem_acc) * 100, errnum_per_cls[i], imgnum_per_cls[i]), file=f)
            count_acc += tem_acc
        print("average acc : %.2f" % (float(count_acc / num_classes) * 100) + "%", file=f)
    np_pred = np.array(l_preds)
    np_label = np.array(l_labels)
    roc_auc_score_ = roc_auc_score(np_label, np_pred)
    return float(eq_count) / float(count), l, roc_auc_score_


def validation1(net, class_num, epoch, val_dataloader):
    val_acc = 0  # 验证集上的准确性   =eq_count/val_count
    val_auc = 0  # 验证集上的auc值
    val_count = 0  # 验证集中图片的数量
    eq_count = 0  # 验证集中识别正确的数量

    all_prd_proba = []  # 存储每一张图片的预测得分
    all_onehot_label = []  # 存储每一张图片的one-hot标签   用于计算roc-auc得分

    err_num_of_everyclass = []  # 用来存储每个类别识别错的数量
    all_num_of_everyclass = [0] * class_num  # 每一个类别的图片数量

    for img_tensors, img_labels in val_dataloader:
        # 一个batch的处理
        img_tensors = img_tensors.to(device)
        img_labels = img_labels.to(device)
        img_labels_list = img_labels.tolist()
        outputs = net(img_labels)

        # 获取模型输出类别
        prd_proba = torch.softmax(outputs, dim=1)
        prd_labels = torch.argmax(outputs, dim=1)

        # 获取预测正确的图片数量
        batch_eq = torch.eq(img_labels, prd_labels).cpu().tolist()
        batch_eq_num = sum(batch_eq)
        eq_count += batch_eq_num
        val_count += img_labels.data.numel()

        batch_prd_proba = prd_proba.cpu().tolist()
        for i in range(len(batch_eq)):
            all_prd_proba.append(batch_prd_proba[i])
            onehot_label = [0] * len(batch_prd_proba[i])
            onehot_label[img_labels_list[i]] = 1
            all_onehot_label.append(onehot_label)
            if batch_eq[i] == 0:
                err_num_of_everyclass.append(img_labels_list[i])
            all_num_of_everyclass[img_labels_list[i]] += 1

    all_onehot_label = np.array(all_onehot_label)
    all_prd_proba = np.array(all_prd_proba)
    err_num_of_everyclass = collections.Count(err_num_of_everyclass)

    current_acc = float(eq_count) / val_count
    current_auc = roc_auc_score(all_onehot_label, all_prd_proba)

    # 记录日志
    with open("checkpoints/log.txt", 'w') as f:
        print("第{}个epoch的验证结果：".format(epoch))
        print("第{}个epoch的验证结果：".format(epoch), file=f)
        print("验证集总图片：" + str(val_count) + "\t" + "识别正确的图片：" + str(eq_count) + "\t" + "准确率："
              + str(current_acc) + "\t" + "auc得分：" + str(current_auc))
        print("验证集总图片：" + str(val_count) + "\t" + "识别正确的图片：" + str(eq_count) + "\t" + "准确率："
              + str(current_acc) + "\t" + "auc得分：" + str * (current_auc), file=f)
        for i in range(class_num):
            print("第{}个类别错误率为：{}（{}/{}）".format(i, float(err_num_of_everyclass[i]) / all_num_of_everyclass[i],
                                                err_num_of_everyclass[i], all_num_of_everyclass[i]))
            print("第{}个类别错误率为：{}（{}/{}）".format(i, float(err_num_of_everyclass[i]) / all_num_of_everyclass[i],
                                                err_num_of_everyclass[i], all_num_of_everyclass[i]), file=f)

    return current_acc, current_auc
