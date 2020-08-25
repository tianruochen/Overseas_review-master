#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :main.py
# @Author   :omega
# @Time     :2020/8/4 0:23

"""training code for overseas review project"""

import os
import warnings
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.util import getdata_from_dictory

from utils.util import Averager
from utils import validation
from module.dataset import DataFactory
from module.basemodel import Net

os.environ["CUDA DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Overseas review project training')
parser.add_argument('-m', '--model', default='unporn', help="model type (default:pron)")


def train(model_type="porn"):

    # *************************************读取配置，制作数据*************************************
    # 加载配置参数
    model_config = json.load(open("./config/model_config.json"))[model_type]
    print(model_config)

    batch_size = model_config["batch_size"]
    classes_num = model_config["classes_num"]
    # 获得标签id与标签name的映射字典
    # map_dict = getMapDict(model_config["class_num"])

    # 获取训练数据及验证数据列表
    train_path = model_config["train_path"]
    train_datalist = getdata_from_dictory(path=train_path)
    val_path = model_config["val_path"]
    val_datalist = getdata_from_dictory(path=val_path)
    train_num = len(train_datalist)
    val_num = len(val_datalist)

    # 根据数据列表装配数据 dataset
    train_dataset = DataFactory(train_datalist, model_config, mode="train")
    val_dataset = DataFactory(val_datalist, model_config, mode="val")

    # 根据dataset制作DataLoder
    train_dataloader = train_dataset.get_dataloader()
    val_dataloader = val_dataset.get_dataloader()

    # *************************************构建模型*************************************
    # 构建网络+将模型移入cuda+加载网络参数
    model = Net(model_type, device, model_config, mode="train").get_model()

    # 打印网络结构
    print("=" * 40 + "network architecture" + "=" * 40)
    for layer_name, layer_params in model.named_parameters():
        print("name: " + layer_name + "\t" + "shape: ", layer_params.shape)
    print("=" * 100)

    model.train()

    # 创建优化器、学习策略、损失函数
    filtered_params = []
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_params.append(param)
    if model_config["optim_type"] == "Adadelta":
        optimizer = optim.Adadelta(filtered_params, lr=0.001, rho=0.9, eps=1e-6)
    elif model_config["optim_type"] == 'SGD':
        optimizer = optim.SGD(filtered_params, lr=0.001, momentum=0.9)
    elif model_config["optim_type"] == 'adam':
        optimizer = optim.Adam(filtered_params, lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    # *************************************开始训练*************************************
    best_acc = 0.0
    best_auc = 0.0
    total_loss = 0.0
    averager = Averager()

    batch_count = math.floor(train_num / batch_size)
    for epoch in range(1, 300):
        print('*****************' + str(epoch) + 'th epoch*******************')
        batch_idx = 0
        for img_tensors, img_labels, image_dics, difficult_degree in train_dataloader:
            # 计算损失
            img_tensors = img_tensors.to(device)
            img_labels = img_labels.to(device)
            outputs = model(img_tensors)
            # CrossEntropyLoss内置了softmax,所以不用显示的增加softmax函数
            batch_loss = criterion(outputs, img_labels)
            total_loss += batch_loss
            # 统计精度
            outputs_labels = torch.argmax(outputs, dim=1)
            matched_labels = torch.eq(img_labels, outputs_labels).cpu()
            matched_numbers = matched_labels.sum()
            averager.add(matched_labels.sum(), img_labels.data.numel())

            #########################
            accuracy = averager.val()
            print("Epoch:{:3d} training batch: {:4}/{:4} --- accuracy: {:.4f} specific：[{:2}/{:2}]".format(
                epoch, batch_idx, batch_count, accuracy, matched_numbers, batch_size))
            averager.reset()
            batch_idx += 1
            #########################

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # 每一个epoch结束，在验证集上进行测试,记录在验证集上的损失，并写入日志
        model.eval()
        with torch.no_grad:
            current_acc, l, current_auc = validation(model, epoch, int(model_config["class_num"]), val_dataloader, val_num)

        # 保存模型参数后者checkpoint
        if current_acc >= best_acc or current_auc >= best_auc:
            # best_accuracy_11_class_b4_auc_adl_380.pth
            checkpoint_path = os.path.join(
                model_config["checkpoints"],
                "{}_class_{}_epoch_{}_acc_{:.4f}_auc_{:.4f}.pth".format(model_type,classes_num, epoch, current_acc, current_auc))
            checkpoint_dict = {
                "epoch": epoch,
                "loss": total_loss/train_num,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, checkpoint_path)
            best_acc = current_acc if current_acc > best_acc else best_acc
            best_auc = current_auc if current_auc > best_auc else best_auc

        print('current_acc:', current_acc, 'best_acc:', best_acc)
        print('current_auc:', current_auc, 'best_auc:', best_auc)

        scheduler.step()
        model.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    """
    混合精度训练：
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    """
    train(args.model)


