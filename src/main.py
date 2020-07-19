"""Overseas review"""

import os
import warnings
import argparse
import json
import collections

import torch
import  numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.utils.util import getdata_from_dictory
from src.utils.util import buildNetwork, getMapDict
from src.utils.util import Averager
from src.module.dataset import DatafromList


os.environ["CUDA DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Overseas review project training')
parser.add_argument('-m', '--model', default='porn_model', help="model type (default:porn)")


def validation(net, class_num, epoch,val_dataloader):
    val_acc = 0     #验证集上的准确性   =eq_count/val_count
    val_auc = 0     #验证集上的auc值
    val_count = 0   #验证集中图片的数量
    eq_count = 0    #验证集中识别正确的数量

    all_prd_proba = []   #存储每一张图片的预测得分
    all_onehot_label = []  #存储每一张图片的one-hot标签   用于计算roc-auc得分

    err_num_of_everyclass = []   #用来存储每个类别识别错的数量
    all_num_of_everyclass = [0]*class_num   #每一个类别的图片数量

    for img_tensors, img_labels in val_dataloader:
        #一个batch的处理
        img_tensors = img_tensors.to(device)
        img_labels = img_labels.to(device)
        img_labels_list = img_labels.tolist()
        outputs = net(img_labels)

        #获取模型输出类别
        prd_proba = torch.softmax(outputs,dim=1)
        prd_labels = torch.argmax(outputs, dim=1)

        #获取预测正确的图片数量
        batch_eq = torch.eq(img_labels, prd_labels).cpu().tolist()
        batch_eq_num = sum(batch_eq)
        eq_count += batch_eq_num
        val_count += img_labels.data.numel()

        batch_prd_proba = prd_proba.cpu().tolist()
        for i in range(len(batch_eq)):
            all_prd_proba.append(batch_prd_proba[i])
            onehot_label = [0]*len(batch_prd_proba[i])
            onehot_label[img_labels_list[i]] = 1
            all_onehot_label.append(onehot_label)
            if batch_eq[i] == 0:
                err_num_of_everyclass.append(img_labels_list[i])
            all_num_of_everyclass[img_labels_list[i]] += 1

    all_onehot_label = np.array(all_onehot_label)
    all_prd_proba = np.array(all_prd_proba)
    err_num_of_everyclass = collections.Count(err_num_of_everyclass)

    current_acc = float(eq_count)/val_count
    current_auc = roc_auc_score(all_onehot_label, all_prd_proba)

    #记录日志
    with open("checkpoint/log.txt",'w') as f:
        print("第{}个epoch的验证结果：".format(epoch))
        print("第{}个epoch的验证结果：".format(epoch), file=f)
        print("验证集总图片："+str(val_count)+"\t"+"识别正确的图片："+str(eq_count)+"\t"+"准确率："
              +str(current_acc)+"\t"+"auc得分："+str(current_auc))
        print("验证集总图片："+str(val_count)+"\t"+"识别正确的图片："+str(eq_count)+"\t"+"准确率："
              +str(current_acc)+"\t"+"auc得分："+str*(current_auc),file=f)
        for i in range(class_num):
            print("第{}个类别错误率为：{}（{}/{}）".format(i,float(err_num_of_everyclass[i])/all_num_of_everyclass[i],
                                                err_num_of_everyclass[i],all_num_of_everyclass[i]))
            print("第{}个类别错误率为：{}（{}/{}）".format(i, float(err_num_of_everyclass[i])/all_num_of_everyclass[i],
                                                err_num_of_everyclass[i], all_num_of_everyclass[i]),file=f)

    return current_acc, current_auc




def train(model="porn_model"):

    # 加载配置参数
    # with open("config.txt","r",encoding="utf-8") as load_f:
    #     str_f = load_f.read()
    #     print(str_f)
    #     config = json.loads(str_f)
    config = json.load(open("config.txt"))[model]
    print(config)

    # 获得标签id与标签name的映射字典
    map_dict = getMapDict(config["class_num"])

    # 获取训练数据及验证数据列表
    train_path = config["train_path"]
    train_datalist = getdata_from_dictory(path=train_path)
    val_path = config["val_path"]
    val_datalist = getdata_from_dictory(path=val_path)

    # 根据数据列表装配数据 dataset
    train_dataset = DatafromList(train_path, mode="train", inputsize=config["input_size"])
    val_dataset = DatafromList(val_path, mode="val", inputsize=config["input_size"])

    # 根据dataset制作DataLoder
    train_dataloader = DataLoader(train_dataset, batch_size=int(config["batch_size"]),
                                  shuffle=True, num_workers=max(int(config["batch_zize"]) / 2, 2))
    val_dataloader = DataLoader(val_dataset, batch_size=int(config["batch_size"]),
                                shuffle=True, num_workers=max(int(config["batch_size"]) / 2, 2))

    # 构建网络、打印网络结构、并加载网络参数
    net = buildNetwork(network_type="B4", class_num=config["class_num"])
    print("=" * 40 + "network architecture" + "=" * 40)
    for layer_name, layer_params in net.named_parameters():
        print("name: " + layer_name + "\t" + "shape: ", layer_params)
    print("=" * 100)
    net = DataParallel(net).to(device)
    if config["best_model"]:
        net.load_state_dict(torch.load(config["best_model"]))
    net.trian()

    # 创建优化器、学习策略、损失函数
    filtered_params = []
    for param in filter(lambda p: p.reguired_grad, net.parameters()):
        filtered_params.append(param)
    if config["optim_type"] == "Adadelta":
        optimizer = optim.Adadelta(filtered_params, lr=0.01, rho=0.9, eps=1e-6)
    elif config["optim_type"] == 'SGD':
        optimizer = optim.SGD(filtered_params, lr=0.01, momentum=0.9)
    elif config["optim_type"] == 'adam':
        optimizer = optim.Adam(filtered_params, lr=0.001)
    scheduler = lr_scheduler.CosineAnnealingLr(optim=optimizer, T_max=50, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    # 开始训练
    best_acc = 0
    best_auc = 0
    averager = Averager()

    for epoch in range(1, 300):
        print('*****************' + str(epoch) + 'th epoch*******************')
        for i,img_tensors, img_labels in enumerate(train_dataloader):
            #计算损失
            img_tensors = img_tensors.to(device)
            img_labels = img_labels.to(device)
            outputs = net(img_tensors)
            #CrossEntropyLoss内置了softmax,所以不用显示的增加softmax函数
            batch_loss = criterion(outputs, img_labels)

            #统计精度
            outputs_labels = torch.argmax(outputs, dim=1)
            matched_labels = torch.eq(img_labels, outputs_labels)
            averager.add(matched_labels.sum(), img_labels.data.numel())

            #########################
            if i % 50 == 0:
                accuracy = averager.val()

                print('training batch:' + str(i)+"/"+str(len(train_datalist)/int(config["batch_size"])+1) + ' accuracy:', accuracy)
                averager.reset()
            #########################
            net.zero_grad()
            batch_loss.backward()
            optimizer.step()

        #每一个epoch结束，在验证集上进行测试,记录在验证集上的损失，并写入日志
        net.eval()
        with torch.no_grad:
            val_acc, val_auc = validation(net, epoch, int(config["class_num"]), val_dataloader)
        # best_acc = val_acc if val_acc > best_acc else best_acc
        # best_auc = val_auc if val_auc > best_auc else best_auc

        #保存模型参数后者checkpoint
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config["best_acc_path"])
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config["best_auc_path"])

        net.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train(args.model)
