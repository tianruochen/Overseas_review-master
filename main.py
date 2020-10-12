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
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.autograd import Variable

from utils.util import getdata_from_dictory

from utils.util import Averager
from utils.validation import validation
from utils.test import test
from utils.easy_mixup import image_mixup, criterion_mixup
from module.dataset import DataFactory
from module.labelsmooth import LSR
from module.basemodel import Net

os.environ["CUDA DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Overseas review project training')
parser.add_argument('-m', '--model', default='unporn', help="model type (default:unporn)")


def train(model_type="unporn"):

    # *************************************读取配置，制作数据*************************************
    # 加载配置参数
    model_config = json.load(open("./config/model_config.json"))[model_type]
    print(model_config)

    # 获取可用的gpu数量
    gpu_nums = torch.cuda.device_count()
    if gpu_nums > 1:
        # model_config["batch_size"] = model_config["batch_size"] * (gpu_nums - 1)
        model_config["batch_size"] = 128
        cudnn.benchmark = True
    batch_size = model_config["batch_size"]
    # model_config["temp_model"] = "/home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth"
    # model_config["temp_model"] = "/home/changqing/workspaces/Overseas_review-master/checkpoints/unporn_models/unporn_class_4_epoch_96_acc_0.9121_auc_0.9882.pth"
    print(f"Use {gpu_nums} gpu, batch size: {batch_size}")
    classes_num = model_config["classes_num"]
    # 获得标签id与标签name的映射字典
    # map_dict = getMapDict(model_config["class_num"])

    # 获取训练数据及验证数据列表  [img_path, img_label]
    train_path = model_config["train_path"]
    val_path = model_config["val_path"]
    test_normal_path = "/home/changqing/workspaces/Overseas_review-master/data/normal_test.txt"
    test_unnorm_path = "/home/changqing/workspaces/Overseas_review-master/data/unnorm_test.txt"

    train_datalist = getdata_from_dictory(path=train_path)
    val_datalist = getdata_from_dictory(path=val_path)
    test_normal_datalist = getdata_from_dictory(path=test_normal_path)
    test_unnorm_datalist = getdata_from_dictory(path=test_unnorm_path)

    train_num = len(train_datalist)
    val_num = len(val_datalist)
    test_normal_num = len(test_normal_datalist)
    test_unnorm_unm = len(test_unnorm_datalist)
    print(f"Train images number : {train_num}   Val images number : {val_num}")
    print(f"Test normal images number : {test_normal_num}   Test unnorm images number : {test_unnorm_unm}")



    # 根据数据列表装配数据 dataset
    train_dataset = DataFactory(train_datalist, model_config, mode="train")
    val_dataset = DataFactory(val_datalist, model_config, mode="val")
    # test_normal_dataset = DataFactory(test_normal_datalist, model_config, mode="val")
    # test_unnorm_dataset = DataFactory(test_unnorm_datalist, model_config, mode="val")

    # 根据dataset制作DataLoder
    train_dataloader = train_dataset.get_dataloader()
    val_dataloader = val_dataset.get_dataloader()
    # test_normal_dataloader = test_normal_dataset.get_dataloader()
    # test_unnorm_dataloader = test_unnorm_dataset.get_dataloader()

    # *************************************构建模型*************************************
    # 构建网络+将模型移入cuda+加载网络参数
    classifier = Net(model_type, device, model_config, mode="train")
    model = classifier.get_model()

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
        optimizer = optim.SGD(filtered_params, lr=0.01, momentum=0.9)
    elif model_config["optim_type"] == 'adam':
        optimizer = optim.Adam(filtered_params, lr=0.001)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=0.0000001)
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80,120,150],gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Using CrossEntropyLoss......")
    criterion = LSR(device).to(device)
    print("Using label smooth ......")


    # *************************************开始训练*************************************
    model.train()
    best_acc = 0.0
    best_auc = 0.0
    total_loss = 0.0
    best_injudge = 1.0
    averager = Averager()

    # 最多保存10个模型
    keep_model_max = 10
    saved_models = deque()
    mixup = False
    if mixup:
        print("use mixup!")

    batch_count = math.floor(train_num / batch_size)
    for epoch in range(1, 300):
        print('*****************' + str(epoch) + 'th epoch*******************')
        batch_idx = 0
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        ###########################  on train  #####################################
        for img_tensors, img_labels, image_dics, difficult_degree in train_dataloader:
            # 计算损失
            img_tensors = img_tensors.to(device)
            img_labels = img_labels.to(device)

            if mixup:
                # mixup 增强
                mixup_alpha = 0.5
                # generate mixed inputs, two one-hot label vectors and mixing coefficient
                img_tensors, label_a,label_b,lam = image_mixup(img_tensors,img_labels,mixup_alpha,torch.cuda.is_available())
                # ima_tensors, label_a, label_b = map(Variable, (img_tensors,label_a, label_b))
                outputs = model(img_tensors)
                mixup_criterion = criterion_mixup(label_a, label_b, lam)
                batch_loss = mixup_criterion(criterion, outputs)
                total_loss += batch_loss
                print(
                    "Epoch:{:3d} training batch: {:4}/{:4} --- loss: {}  lr:{} ".format(
                        epoch, batch_idx, batch_count, total_loss / (batch_idx + 1),
                        optimizer.state_dict()['param_groups'][0]['lr']))

            else:
                # CrossEntropyLoss内置了softmax,所以不用显示的
                # 增加softmax函数
                outputs = model(img_tensors)
                batch_loss = criterion(outputs, img_labels)
                total_loss += batch_loss
                # 统计精度
                outputs_labels = torch.argmax(outputs, dim=1)
                matched_labels = torch.eq(img_labels, outputs_labels).cpu()
                matched_numbers = matched_labels.sum()
                averager.add(matched_labels.sum(), img_labels.data.numel())

                #########################
                accuracy = averager.val()
                print("Epoch:{:3d} training batch: {:4}/{:4} --- loss: {}  lr:{} accuracy: {:.4f} specific：[{:2}/{:2}]".format(
                    epoch, batch_idx, batch_count, total_loss/(batch_idx+1), optimizer.state_dict()['param_groups'][0]['lr'], accuracy, matched_numbers, img_labels.data.numel()))
                averager.reset()
            batch_idx += 1
            #########################

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()


        ###########################  on val and test  #####################################

        # 每一个epoch结束，在验证集上进行测试,记录在验证集上的损失，并写入日志
        model.eval()
        with torch.no_grad():
            # current_acc, l, current_auc = validation(model, train_dataloader, int(model_config["classes_num"]), epoch, device, "train")
            # if epoch > 200 and epoch % 2 == 0 or best_acc > 0.9424:
            #     validation(model, train_dataloader, int(model_config["classes_num"]), epoch,device, "train")
            # current_injudge = test(model, test_normal_datalist, test_unnorm_datalist, epoch, device, lr, total_loss)
            current_acc, l, current_auc = validation(model, val_dataloader, int(model_config["classes_num"]), epoch, device, lr, total_loss, "val")

        # 保存模型参数后者checkpoint
        if (current_acc >= 0.922 or current_acc >= best_acc or current_auc >= best_auc) and epoch >= 5: #or current_injudge <= best_injudge:
            # best_accuracy_11_class_b4_auc_adl_380.pth
            checkpoint_path = os.path.join(
                model_config["checkpoints"],
                "epoch_{}_acc_{:.4f}_auc_{:.4f}.pth".format(epoch, current_acc, current_auc))
            # checkpoint_dict = {
            #     "epoch": epoch,
            #     "loss": total_loss/train_num,
            #     "model_state_dict": model.state_dict(),
            #     "optim_state_dict": optimizer.state_dict(),
            # }

            torch.save(model.state_dict(), checkpoint_path)
            print(f"checkpoint {epoch + 1} saved!")

            saved_models.append(checkpoint_path)
            if len(saved_models) > 100:         # 最多保存20个模型
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    print(f"faild to remove {model_to_remove}")

            best_acc = current_acc if current_acc > best_acc else best_acc
            best_auc = current_auc if current_auc > best_auc else best_auc

        print('current_acc:', current_acc, 'best_acc:', best_acc)
        print('current_auc:', current_auc, 'best_auc:', best_auc)

        total_loss = 0.0

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


