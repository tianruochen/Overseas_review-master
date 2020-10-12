#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :test.py
# @Time     :2020/9/27 下午3:50
# @Author   :Chang Qing

import os
import sys
import json
import torch
import numpy as np

import glob
from PIL import Image
from threading import Thread,Lock
from module.basemodel import Net
from tqdm import tqdm

from module.basemodel import Net
from utils.util import getdata_from_dictory, get_tfms

if (sys.version_info > (3,0)):
    import queue as Queue
else:
    import Queue

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mutex = Lock()
task_queue = Queue.Queue()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# 评估数据集
val_unporn_dataset = {
    "cocofun_normal": "/data1/zhaoshiyu/cocofun_normal",
    "cocofun_unnorm": "/data1/zhaoshiyu/temp/kill_image",
    #### video #########
    "cocofun_disgust_path": "/data/wangruihao/serious_data/disgusting",
    "cocofun_sensitive_path": "/data/wangruihao/serious_data/sensitive"
}
########################################

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
        #print(invit_score_list)
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


def remove_porn(save_logits_path):
    invitations = []
    with open("/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt") as f:
        for logit in f.readlines():
            invitations.append(logit.split("\t")[0].split('/')[-2])

    print(len(invitations))

    new_logits_info = []
    with open(save_logits_path) as f:
        for logit in f.readlines():
            if logit.split("\t")[0].split('/')[-2] in invitations:
                new_logits_info.append(logit)

    print(len(new_logits_info))
    with open(save_logits_path, "w") as f:
        for logit in new_logits_info:
            f.write(logit)

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


def get_logits(model, datalist, device, model_path , val_dataset):
    logits_info_list = []
    for img_path, _ in datalist:
        # 将标签tensor转换为tensor
        try:
            img = Image.open(img_path).convert("RGB")

        except Exception as e:
            print(e)
            continue
        img_list = cut_long_img(img)
        pred = model(torch.cat(img_list, 0).to(device))
        # 获取模型输出类别
        pred = torch.softmax(pred, dim=1)
        pred = pred.tolist()

        info = img_path + "\t" + json.dumps(pred) + "\n"
        logits_info_list.append(info)

    #将logits写入文件
    save_logits_name = model_path.split("/")[-1][15:-4] + "_logits.txt"
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


def test(model_path,test_info):
    model_type = test_info["model_type"]
    mode = test_info["mode"]
    test_normal_path = test_info["test_normal_path"]
    test_unnorm_path = test_info["test_unnorm_path"]
    test_normal_datalist = getdata_from_dictory(path=test_normal_path)
    test_unnorm_datalist = getdata_from_dictory(path=test_unnorm_path)

    model_config = json.load(open("config/model_config.json"))[model_type]
    model_config["best_model"] = model_path
    classifier = Net(model_type, device, model_config, mode)
    model = classifier.get_model()
    
    norm_logits = os.path.join("./summary/logits/unporn/normal", model_path.split("/")[-1][:-4] + "_logits.txt")
    kill_logits = os.path.join("./summary/logits/unporn/unnorm", model_path.split("/")[-1][:-4] + "_logits.txt")

    if not os.path.isfile(norm_logits):
        norm_logits = get_logits(model, test_normal_datalist, device, model_path, val_dataset="normal")
        print(f"Norm logits generate done! file path is {norm_logits}")
    if not os.path.isfile(kill_logits):
        kill_logits = get_logits(model, test_unnorm_datalist, device, model_path, val_dataset="unnorm")
        print(f"Kill logits generate done! file path is {kill_logits}")

    threshold = generate_threshold(kill_logits, rate=0.02)
    print("threshold" + str(threshold))

    injudge_num, invit_num, injudge_ratio = get_injudge_rate(norm_logits, threshold)
    print(injudge_ratio)
    test_results_file = "/home/changqing/workspaces/Overseas_review-master/summary/results/unporn/testresults.txt"
    with open(test_results_file, "a") as f:
        print("#"*60, file=f)
        print("model path: %s" % model_path, file=f)
        print("threshold: " + str(threshold), file=f)
        print(f"injudge nums: {injudge_num}, total invits: {invit_num} ,injudge ratio: {injudge_ratio}")
        print(f"injudge nums: {injudge_num}, total invits: {invit_num} ,injudge ratio: {injudge_ratio}", file=f)

    return injudge_ratio


def one_thread_process(test_info, thread_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(thread_id // 3)
    print(f"Thread {thread_id} run in {thread_id // 3}th GPU")
    count = 0
    while not task_queue.empty():
        count += 1
        mutex.acquire(60)
        if task_queue.empty():
            mutex.release()
            break
        else:
            model_path = task_queue.get()
        mutex.release()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test(model_path, test_info)

##### wrong function
def multi_thread_test(models_list, test_info, thread_num=24):
    """
    :param task_queue: models list
    :param thread_num:
    :return:
    """
    threads = []
    for model_path in models_list:
        task_queue.put(model_path)
    for thread_id in range(thread_num):
        threads.append(Thread(target=one_thread_process, args=(test_info, thread_id,)))
        threads[-1].start()
    for thread in threads:
        thread.join()
    print("All task done....")


if __name__ == "__main__":
    # killing-image 上卡0.02的漏放率，得到阈值
    models_dir = "/home/changqing/workspaces/Overseas_review-master/checkpoints/unporn_models/scheme2/"
    models_list = glob.glob(models_dir + "*.pth")
    print(f"test : {len(models_list)} models")
    test_info = {
        "model_type" : "unporn",
        "mode": "val",
        "test_normal_path": "/home/changqing/workspaces/Overseas_review-master/data/normal_test.txt",
        "test_unnorm_path": "/home/changqing/workspaces/Overseas_review-master/data/unnorm_test.txt"
    }

    # wrong function
    # multi_thread_test(models_list, test_info)

    for model_path in models_list:
        print("model path : %s " % (model_path))
        test(model_path, test_info)

    # kill_logits = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/unnorm/epoch_95_acc_0.9205_auc_0.9835_logits.txt"
    # # kill_logits = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    # threshold = generate_threshold(kill_logits, rate=0.02)
    # print("threshold" + str(threshold))
    # norm_logits = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/normal/epoch_95_acc_0.9205_auc_0.9835_logits.txt"
    # # norm_logits = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    # injudge_num, invit_num, injudge_ratio = get_injudge_rate(norm_logits, threshold)
    # print(injudge_ratio)



