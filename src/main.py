"""Overseas review"""

import os
import warnings
import argparse
import json
import collections

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.utils.util import getdata_from_dictory
from src.utils.util import getMapDict

from src.utils.util import Averager
from src.utils.validation import validation
from src.module.dataset import DataFactory
from src.module.basemodel import Net




os.environ["CUDA DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Overseas review project training')
parser.add_argument('-m', '--model', default='porn', help="model type (default:porn)")


def train(model_type="porn"):

    # 加载配置参数
    # with open("model_config.txt","r",encoding="utf-8") as load_f:
    #     str_f = load_f.read()
    #     print(str_f)
    #     model_config = json.loads(str_f)
    model_config = json.load(open("model_config/model_config.txt"))[model_type]
    print(model_config)

    # 获得标签id与标签name的映射字典
    # map_dict = getMapDict(model_config["class_num"])

    # 获取训练数据及验证数据列表
    train_path = model_config["train_path"]
    train_datalist = getdata_from_dictory(path=train_path)
    val_path = model_config["val_path"]
    val_datalist = getdata_from_dictory(path=val_path)

    # 根据数据列表装配数据 dataset
    train_dataset = DataFactory(train_datalist, model_config, mode="train")
    val_dataset = DataFactory(val_datalist, model_config, mode="val")

    # 根据dataset制作DataLoder
    train_dataloader = train_dataset.get_dataloader()
    val_dataloader = val_dataset.get_dataloader()

    # 构建网络、打印网络结构、并加载网络参数
    net = Net(model_type, model_config)
    print("=" * 40 + "network architecture" + "=" * 40)
    for layer_name, layer_params in net.named_parameters():
        print("name: " + layer_name + "\t" + "shape: ", layer_params)
    print("=" * 100)
    model = DataParallel(net).to(device)
    if model_config["best_model"]:
        model.load_state_dict(torch.load(model_config["best_model"]))
    model.trian()

    # 创建优化器、学习策略、损失函数
    filtered_params = []
    for param in filter(lambda p: p.reguired_grad, net.parameters()):
        filtered_params.append(param)
    if model_config["optim_type"] == "Adadelta":
        optimizer = optim.Adadelta(filtered_params, lr=0.01, rho=0.9, eps=1e-6)
    elif model_config["optim_type"] == 'SGD':
        optimizer = optim.SGD(filtered_params, lr=0.01, momentum=0.9)
    elif model_config["optim_type"] == 'adam':
        optimizer = optim.Adam(filtered_params, lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLr(optim=optimizer, T_max=50, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    # 开始训练
    best_acc = 0
    best_auc = 0
    averager = Averager()

    for epoch in range(1, 300):
        print('*****************' + str(epoch) + 'th epoch*******************')
        batch_idx = 0
        for img_tensors, img_labels, image_dics, difficult_degree in train_dataloader:
            #计算损失
            img_tensors = img_tensors.to(device)
            img_labels = img_labels.to(device)
            outputs = model(img_tensors)
            #CrossEntropyLoss内置了softmax,所以不用显示的增加softmax函数
            batch_loss = criterion(outputs, img_labels)

            #统计精度
            outputs_labels = torch.argmax(outputs, dim=1)
            matched_labels = torch.eq(img_labels, outputs_labels)
            averager.add(matched_labels.sum(), img_labels.data.numel())

            #########################
            if i % 50 == 0:
                accuracy = averager.val()

                print("Epoch:{:3d} training batch: {:4}/{:4} --- accuracy: {:.4f} specific：[{:2}/{:2}]".format(
                    epoch, batch_idx, batch_count, accuracy, equal_num, batch_size))
                averager.reset()
            #########################
            net.zero_grad()
            batch_loss.backward()
            optimizer.step()

        #每一个epoch结束，在验证集上进行测试,记录在验证集上的损失，并写入日志
        model.eval()
        with torch.no_grad:
            current_acc, l, current_auc = validation(net, epoch, int(model_config["class_num"]), val_dataloader)
        # best_acc = val_acc if val_acc > best_acc else best_acc
        # best_auc = val_auc if val_auc > best_auc else best_auc

        #保存模型参数后者checkpoint
        if current_acc >= best_acc or current_auc >= best_auc:
            # best_accuracy_11_class_b4_auc_adl_380.pth
            save_best_acc_path = os.path.join(
                model_config["best_acc_path"],
                "unpron_cla_{}_epoch_{}_acc_{:.4f}_auc_{:.4f}.pth".format(numer_classes, epoch, current_acc, current_auc))
            torch.save(model.state_dict(), save_best_acc_path)
            best_acc = current_acc if current_acc > best_acc else best_acc
            best_auc = current_auc if current_auc > best_auc else best_auc

        print('current_acc:', current_acc, 'best_acc:', best_acc)
        print('current_auc:', current_auc, 'best_auc:', best_auc)

        scheduler.step()
        model.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train(args.model)
