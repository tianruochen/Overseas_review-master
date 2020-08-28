#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :modify_dataset.py
# @Author   :omega
# @Time     :2020/8/4 0:15

import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np

# pd.set_option("display.max_columns", None)

task = "unpron"
modify_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary", task)


def get_modify_data(mode='val'):
    modify_dir = os.path.join(modify_root_dir, mode)
    modify_files = os.listdir(modify_dir)
    modify_file_list = []
    for modify_file in modify_files:
        modify_file_list.append(os.path.join(modify_dir,modify_file))
    df = pd.DataFrame(columns=("img_labels", "prd_probas", "img_dirs", "modify_cla"))
    for csvfile in modify_file_list:
        temp_df = pd.read_csv(csvfile)
        df = df.append(temp_df,ignore_index=True)
    df = df.dropna(how='any')
    df = df[df["img_labels"] != df["modify_cla"]][["img_dirs", "img_labels", "modify_cla"]]
    modify_data_list = df.values.tolist()
    print(len(modify_data_list))

    modify_data_basepath = "modify_{}_data.csv".format(mode)
    modify_data_path = os.path.join(modify_dir,modify_data_basepath)
    if not os.path.exists(modify_data_path):
        df.to_csv(modify_data_path)
    return modify_data_list


def generate_dataset(modify_data_list, mode):
    new_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.exists(new_dataset_dir):
        os.mkdir(new_dataset_dir)
    if mode == "val":
        origin_dataset = '/data1/wangruihao/cocofun/mass1/val.txt'
        new_dataset = os.path.join(new_dataset_dir,"val.txt")
    else:
        origin_dataset = "/data1/wangruihao/cocofun/mass1/train.txt"
        new_dataset = new_dataset = os.path.join(new_dataset_dir, "train.txt")

    new_data_list = []
    skip_count = 0
    with open(origin_dataset, "r") as f:
        for line in f.readlines():
            img_path,label = line.strip().split("\t")

            skip = False
            for i, path in enumerate([modify_data[0] for modify_data in modify_data_list]):
                if path == img_path:
                    if int(modify_data_list[i][2]) < 0:
                        skip_count += 1
                        skip = True
                        break
                    label = modify_data_list[i][2]
                    break
            if skip:
                continue
            new_data_list.append([img_path, label])
    print("{} skip count: {}".format(mode, skip_count))

    with open(new_dataset, "w") as f:
        for path,new_label in tqdm(new_data_list):
            # print(path + "\t" + str(int(new_label)))
            print(path + "\t" + str(int(new_label)), file=f)
        print("{} data generate done: {}".format(mode, len(new_data_list)))

def get_outlier(logits_name, data_type="normal",thres=0.5):
    """
    :param logits_name: logits file name   .txt
    :param data_type:   normal dataset or unnorm dataset
    :return:
    """
    save_outlier_dir = os.path.join(os.path.dirname(os.getcwd()), "summary/outlier/unporn/")
    print(save_outlier_dir)
    if not os.path.exists(save_outlier_dir):
        os.makedirs(save_outlier_dir)
    logits_dir = os.path.join(os.path.dirname(os.getcwd()),"summary/logits/unporn/")
    if data_type == "normal":
        logits_dir = logits_dir + "cocofun_normal/"
        save_outlier_dir = save_outlier_dir + "cocofun_normal"
    else:
        logits_dir = logits_dir +"cocofun_unnorm/"
        save_outlier_dir = save_outlier_dir + "cocofun_unnorm"

    logits_path = os.path.join(logits_dir,logits_name)
    save_outlier_path = os.path.join(save_outlier_dir,logits_name.split('.')[0] + ".txt")
    print(save_outlier_path)

    with open(logits_path, 'r') as f:
        file_info_list = f.read().split("\n")
        invitation_map = {}
        invitation_list = []
        logits = []

        for line in file_info_list:
            if not line:
                continue
            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = np.array(json.loads(line[1])).min(axis=0)
            if invitation_name not in invitation_map:
                invitation_map[invitation_name] = logit[None, :]
            else:
                invitation_map[invitation_name] = np.concatenate([invitation_map[invitation_name], logit[None, :]],
                                                                 axis=0)
                # cmd + backspace 删除当前行
            logits.append(logit)

        logits = np.array(logits)
        print(logits[:5])
        print(f"total {len(invitation_map)} invitations")
        for invi_name, invitation_logits in invitation_map.items():
            if data_type == "normal":
                if invitation_logits.min(axis=0)[2] < thres:
                    invitation_list.append(invi_name)
            elif data_type == "unnorm":
                if invitation_logits.min(axis=0)[2] > thres:
                    invitation_list.append(invi_name)

        with open(save_outlier_path,"w") as f:
            for invitation in tqdm(invitation_list):
                f.write(invitation+"\n")
            f.flush()
        print("write done!")

        return invitation_list




if __name__ == "__main__":

    invi_list = get_outlier("best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt", data_type="unnorm",thres=0.3)
    print(invi_list[:10])
    # for mode in ["train", "val"]:
    #     modify_data_list = get_modify_data(mode)
    #     generate_dataset(modify_data_list, mode)