#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :validation.py
# @Author   :chang qing
# @Time     :2020/8/4 0:23

import os
import torch
import json
import collections
from tqdm import tqdm
import glob

import numpy as np
import pandas as pd
from PIL import Image
from module.basemodel import Net
from utils.util import getdata_from_dictory, get_tfms
from sklearn.metrics import roc_auc_score

os.environ["CUDA DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

#########################################
# 评估数据集
val_unporn_dataset = {
    "cocofun_normal": "/data1/zhaoshiyu/cocofun_normal",
    "cocofun_unnorm": "/data1/zhaoshiyu/temp/kill_image",
    #### video #########
    "cocofun_disgust_path": "/data/wangruihao/serious_data/disgusting",
    "cocofun_sensitive_path": "/data/wangruihao/serious_data/sensitive"
}
########################################

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


def validation(model, val_dataset, num_classes, epoch, device, lr, loss, mode="val"):
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
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summary/results", "unporn")
        name_classes = ["blood", "religion", "normal", "group"]
    elif num_classes == 11:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summary/results", "porn")
        name_classes = [["class-" + str(i) for i in range(num_classes)]]

    if mode == "val":
        result_file = os.path.join(results_dir, "valresult.txt")
    else:
        result_file = os.path.join(results_dir, "trainresult.txt")

    with open(result_file, "a+") as f:
        print(('=' * 50) + "epoch:" + str(epoch) + ("=" * 50), file=f)
        print("lr: " + str(lr), file=f)
        print("loss: " + str(loss), file=f)
        print("各类图片数量： ", imgnum_per_cls, file=f)
        print("各类识别错误的图片量： ", errnum_per_cls, file=f)
        count_acc = 0
        for i in range(num_classes):
            tem_acc = (imgnum_per_cls[i] - errnum_per_cls[i]) / imgnum_per_cls[i]
            print("accuracy of {} :{:.2f}%-{}/{}".format(
                name_classes[i],
                float(tem_acc) * 100, errnum_per_cls[i], imgnum_per_cls[i]), file=f)
            count_acc += tem_acc
        print("class average acc : %.2f" % (float(count_acc / num_classes) * 100) + "%" +
              "\t image average acc : %.2f" % (float(eq_count) / float(count) * 100) + "%", file=f)
    np_pred = np.array(l_preds)
    np_label = np.array(l_labels)
    roc_auc_score_ = roc_auc_score(np_label, np_pred)
    return float(eq_count) / float(count), l, roc_auc_score_


def validation1(net, class_num, epoch, val_dataloader,device):
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



def generate_logits(model_type,model_config, data_path,save_logits_dir,device):
    """
    产生训练数据的logits（将所有数据前向传播通过网络的结果记录下来）
    :return:
    """

    # # 加载配置参数
    # model_config = json.load(open("./config/model_config.json"))[model_type]

    # 构建网络+将模型移入cuda+加载网络参数
    classifier = Net(model_type, device, model_config, mode="val")


    best_model_name = model_config["best_model"].split("/")[-1]
    save_logits_name = best_model_name + "_logits.txt"
    # save_logits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/summary/logits",
    #                                data_path.split("/")[-1])
    if not os.path.exists(save_logits_dir):
        os.mkdir(save_logits_dir)

    save_logits_path = os.path.join(save_logits_dir, save_logits_name)

    invitation_list = os.listdir(data_path)
    img_count = 0
    error_img_count = 0
    invitation_count = len(invitation_list)
    print(invitation_count)
    error_invitation_count = 0
    error_invitation_list = []
    with open(save_logits_path, "w+") as f:
        # for every invitation
        for invitation in tqdm(invitation_list):
            imglist_of_invitation = glob.glob(os.path.join(data_path, invitation) + "/*jp*g")
            imglist_of_invitation.extend(glob.glob(os.path.join(data_path, invitation) + "/*png"))
            normal_invitation = True
            # for every image
            for imgpath in imglist_of_invitation:
                try:
                    img = Image.open(imgpath).convert("RGB")
                except:
                    continue
                img_count += 1
                # pred without softmax
                risk, pred = classifier.predict_img_api(img)
                pred = json.dumps(pred)
                f.write(f"{imgpath}" + "\t" + f"{pred}" + "\t" + f"{risk}" + "\n")
                if risk > 0:
                    error_invitation_list.append(invitation)
                    normal_invitation = False
                    error_img_count += 1
            if normal_invitation:
                error_invitation_count += 1
    img_acc = (img_count - error_img_count) / img_count
    invitation_acc = (invitation_count - error_invitation_count) / invitation_count
    print(f"total imgs : {img_count}, error imgs : {error_img_count}, img acc : {img_acc} ")
    print(f"total invs : {invitation_count}, error invs : {error_invitation_count}, inv acc : {invitation_acc}")



def generate_threshold():
    """根据进审率产生，产生模型在coco_norm数据集上的阈值"""
    injudge_ratio = [0.5,0.5218,0.6,0.7] #[0+i*0.05 for i in range(1,20)]

    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt"

    invitation_map = {}
    logits = []
    invit_score_list = []

    with open(suffix_file_path, 'r') as f:
        file_info_list = f.read().split("\n")

        for line in file_info_list:
            if not line:
                continue
            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = np.array(json.loads(line[1])).min(axis=0)

            # image_score_list.append(logit[2])

            if invitation_name not in invitation_map:
                invitation_map[invitation_name] = logit[None, :]
            else:
                invitation_map[invitation_name] = np.concatenate([invitation_map[invitation_name], logit[None, :]], axis=0)

        for invitation_name,invitation_logit in invitation_map.items():
            # cmd + backspace 删除当前行
            min_invitation_score = np.min(invitation_logit,axis=0)[2]
            invit_score_list.append(min_invitation_score)

        logits.append(logit)
        print(len(logits))
        invit_score_list.sort()

        thresholds = []
        for ratio in injudge_ratio:
            # image_mode_threshold = image_score_list[int(ratio*len(image_score_list))]
            thresholds.append(invit_score_list[int(ratio*len(invit_score_list))])
            # invit_mode_threshold = invit_score_list[int(ratio*len(invit_score_list))]

        # print(f"image_mode_threshold: {image_mode_threshold}   invit_mode_threshold:{invit_mode_threshold}")
        print(suffix_file_path)
        print(thresholds)



def get_injudge_ration(threshold = 0.7):
    """根据指定的阈值，产生在coco_norm数据集上的进审率"""

    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    # suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt"
    invitation_map = {}
    logits = []
    invit_score_list = []

    with open(suffix_file_path, 'r') as f:
        file_info_list = f.read().split("\n")

        for line in file_info_list:
            if not line:
                continue
            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = np.array(json.loads(line[1])).min(axis=0)

            # image_score_list.append(logit[2])

            if invitation_name not in invitation_map:
                invitation_map[invitation_name] = logit[None, :]
            else:
                invitation_map[invitation_name] = np.concatenate([invitation_map[invitation_name], logit[None, :]],
                                                                 axis=0)

        for invitation_name, invitation_logit in invitation_map.items():
            # cmd + backspace 删除当前行
            min_invitation_score = np.min(invitation_logit, axis=0)[2]
            invit_score_list.append(min_invitation_score)

        logits.append(logit)
        invit_num = len(invit_score_list)
        invit_score_list.sort()

        invit_score_list = np.array(invit_score_list)
        injudge_num = np.sum(invit_score_list < threshold)
        injudge_ratio = injudge_num / invit_num

        print(f"injudge nums: {injudge_num}, total invits: {invit_num} ,injudge ratio: {injudge_ratio}")

        # thresholds = []
        # for ratio in injudge_ratio:
        #     # image_mode_threshold = image_score_list[int(ratio*len(image_score_list))]
        #     thresholds.append(invit_score_list[int(ratio * len(invit_score_list))])
            # invit_mode_threshold = invit_score_list[int(ratio*len(invit_score_list))]

        # print(f"image_mode_threshold: {image_mode_threshold}   invit_mode_threshold:{invit_mode_threshold}")

if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "unporn"
    # mode = "val"


    # get_injudge_ration()
    generate_threshold()

    # if model_type == "unporn":
    #     dataset_name = "cocofun_unnorm"
    #     dataset_name = "cocofun_normal"
    #     cocofun_unnorm_path = val_unporn_dataset[dataset_name]
    #     model_config = json.load(open("../config/model_config.json"))[model_type]
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/checkpoints/unporn_models/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/t_max5/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/t_max5/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/t_max5/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/aug_lr_0.0001/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/aug_lr_0.001/unporn_class_4_epoch_20_acc_0.9407_auc_0.9923.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/new/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth"
    #     model_config["best_model"] = "/home/changqing/workspaces/Overseas_review-master/model/mixup/unporn_class_4_epoch_16_acc_0.9288_auc_0.9935.pth"
    #
    #     # summary_path = os.path.dirname(os.getcwd())
    #     save_logits_dir = os.path.join(os.path.dirname(os.getcwd()) + "/summary/logits/unporn",dataset_name)
    #     print(save_logits_dir)
    #     if not os.path.exists(save_logits_dir):
    #         os.mkdir(save_logits_dir)
    #     generate_logits(model_type, model_config,cocofun_unnorm_path,save_logits_dir,device)
    #
    # elif model_type == "porn":
    #     pass
    # #
