#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :gen_test.py.py
# @Time     :2020/9/30 上午11:06
# @Author   :Chang Qing
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.util import get_tfms

def generate_threshold(kill_logits, rate=0.02):
    """
    在kill image数据集上保持2%漏放率的情况下, 求得相应的阈值, 帖子维度
    :param kill_logits: model logits on kill image dataset
    :param rate: 0.02
    :return: threshold  float
    """

    suffix_file_path = kill_logits
    invitation_map = {}
    invit_score_list = []

    with open(suffix_file_path, 'r') as f:
        file_info_list = f.read().split("\n")

        for line in file_info_list:
            if not line:
                continue
            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = np.array(json.loads(line[1])).min(axis=0)

            if invitation_name not in invitation_map:
                invitation_map[invitation_name] = logit[None, :]
            else:
                invitation_map[invitation_name] = np.concatenate([invitation_map[invitation_name], logit[None, :]], axis=0)

        for invitation_name, invitation_logit in invitation_map.items():
            min_invitation_score = np.min(invitation_logit, axis=0)[2]
            invit_score_list.append(min_invitation_score)

        invit_score_list.sort(reverse=True)
        print(invit_score_list)
        invit_num = len(invit_score_list)
        print(invit_num)
        print(int(invit_num * rate))
        return invit_score_list[int(invit_num * rate)]


def get_injudge_rate(norm_logits, threshold):
    suffix_file_path = norm_logits
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
        return injudge_num, invit_num, injudge_ratio


def cut_long_img(im):
    tfms = get_tfms(380,380,"val")
    try:
        img_w, img_h = im.size
        img_w = float(img_w)
        img_h = float(img_h)
        h_w = img_h / img_w
    except:
        return -1, 0

    img_list = []
    if h_w > 2.0:
        split_len = int(img_w * 1.0)
        h_div_w = img_h / split_len
        split_num = int(min(20, np.ceil(h_div_w)))

        split_stride = int((img_h - split_len - 1) // (split_num - 1))
        for i in range(split_num):
            t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
            img_list.append(tfms(t_img.resize((380,380))).unsqueeze(0))

    elif h_w < 0.5:
        split_len = int(img_h * 1)
        h_div_w = img_w / split_len
        split_num = int(min(20, np.ceil(h_div_w)))

        split_stride = int((img_w - split_len - 1) // (split_num - 1))
        for i in range(split_num):
            t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
            img_list.append(tfms(t_img.resize((380,380))).unsqueeze(0))
    else:
        img_list.append(tfms(im.resize((380,380))).unsqueeze(0))

    return img_list


def get_logits(model, datalist, epoch, device,val_dataset):
    logits_info_list = []
    for img_path, _ in tqdm(datalist):
        # 将标签tensor转换为tensor
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        img_list = cut_long_img(img)
        pred = model(torch.cat(img_list, 0).cuda())
        # 获取模型输出类别
        pred = torch.softmax(pred, dim=1)
        pred = pred.tolist()

        info = img_path + "\t" + json.dumps(pred) + "\n"
        logits_info_list.append(info)

    #将logits写入文件
    save_logits_name = "epoch_" + str(epoch) + "_logits.txt"
    # save_logits_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/summary/logits",
    #                                data_path.split("/")[-1])
    save_logits_dir = os.path.join("./summary/logits/unporn", val_dataset)
    if not os.path.exists(save_logits_dir):
        os.mkdir(save_logits_dir)

    save_logits_path = os.path.join(save_logits_dir, save_logits_name)
    with open(save_logits_path,"w") as f:
        for logit_info in logits_info_list:
            f.write(logit_info)
    return save_logits_path


def test(model, test_normal_datalist, test_unnorm_datalist, epoch, device, lr, loss, ):
    norm_logits = get_logits(model, test_unnorm_datalist, epoch, device, val_dataset="unnorm")
    print(f"Norm logits generate done! file path is {norm_logits}")
    kill_logits = get_logits(model, test_normal_datalist, epoch, device, val_dataset="normal")
    print(f"Kill logits generate done! file path is {kill_logits}")

    threshold = generate_threshold(kill_logits, rate=0.02)
    print(threshold)

    injudge_num, invit_num, injudge_ratio = get_injudge_rate(norm_logits, threshold)
    print(injudge_ratio)
    test_results_file = "/home/changqing/workspaces/Overseas_review-master/summary/results/unporn/testresults.txt"
    with open(test_results_file, "a") as f:
        print(('=' * 50) + "epoch:" + str(epoch) + ("=" * 50))
        print("lr: " + str(lr))
        print("loss: " + str(loss))
        print(f"injudge nums: {injudge_num}, total invits: {invit_num} ,injudge ratio: {injudge_ratio}")

        print(('=' * 50) + "epoch:" + str(epoch) + ("=" * 50), file=f)
        print("lr: " + str(lr), file=f)
        print("loss: " + str(loss), file=f)
        print(f"injudge nums: {injudge_num}, total invits: {invit_num} ,injudge ratio: {injudge_ratio}", file=f)

    return injudge_ratio


