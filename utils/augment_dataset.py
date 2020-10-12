#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :augment_dataset.py
# @Time      :2020/9/9 下午4:00
# @Author    :Chang Qing
 
import os
import glob
import random
from PIL import Image



cla_label_dict = {
    "blood": 0,
    "religion": 1,
    "normal": 2,
    "group": 3
}

cocofun_normal_dir = "/data1/zhaoshiyu/cocofun_normal/"

def aug_coco_norm(data_path,rate=0.3):
    invitation_list = os.listdir(data_path)
    invitation_num = len(invitation_list)
    aug_num = int(invitation_num * rate)
    # random.choice(seq) 从队列中选取一个元素
    # random.choices(seq,k) 从队列中中有放回的选k个元素
    # random.sample(seq,k) 从队列中无放回的选k个元素
    img_choiced_list = []
    valid_invitation = random.sample(invitation_list, k=aug_num)
    print(valid_invitation[:20])
    for invitation in valid_invitation:
        invitation_path = os.path.join(cocofun_normal_dir,invitation)
        invitation_img_list = glob.glob(os.path.join(invitation_path, "*.*g"))
        if invitation_img_list:
            img_choiced = random.choice(invitation_img_list)
            try:
                Image.open(img_choiced)
            except:
                continue
            new_data_str = img_choiced + "\t" + str(2) + "\n"
            img_choiced_list.append(new_data_str)


    return img_choiced_list





def get_new_data(root_dir):
    aug_data_list = []
    for cla, label in cla_label_dict.items():
        cla_data_path = os.path.join(root_dir,cla)
        img_path_list = glob.glob(os.path.join(cla_data_path, "*.*g"))
        for img_path in img_path_list:
            try:
                Image.open(img_path)
                new_data_str = img_path + "\t" + str(label) + "\n"
                aug_data_list.append(new_data_str)
            except:
                continue
    return aug_data_list


def get_ori_data(data_path):
    ori_data_list = []
    if not os.path.exists(data_path):
        return ori_data_list
    with open(data_path) as f:
        for ori_data_str in f.readlines():
            ori_data_list.append(ori_data_str)
    return ori_data_list



if __name__ == "__main__":
    img_choiced_list = aug_coco_norm(cocofun_normal_dir)
    print(len(img_choiced_list))
    print(img_choiced_list[:5])
    aug_data_list = img_choiced_list
    # aug_data_path = "/home/changqing/data/unporn_data+"
    # aug_data_list = get_new_data(aug_data_path)
    # print(len(aug_data_list))

    # ori_data_path = "/home/changqing/workspaces/Overseas_review-master/data/modified_data/train.txt"
    ori_data_path = "/home/changqing/workspaces/Overseas_review-master/data/train.txt"
    ori_data_list = get_ori_data(ori_data_path)
    print(len(ori_data_list))

    new_data_list = aug_data_list
    new_data_list.extend(ori_data_list)
    new_data_path = "/home/changqing/workspaces/Overseas_review-master/data/train2.txt"

    with open(new_data_path,'w+') as f:
        f.writelines(new_data_list)
    # old_dataset = "/home/changqing/workspaces/Overseas_review-master/data/modified_data/train.txt"
    # with open(old_dataset) as f:
    #     old_data_list = f.readlines()
    #





