#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logits_diff.py
# @Time      :2020/9/15 下午7:13
# @Author    :ChangQing

import os



def generate_logits_diff(old_logits_file, new_logits_file):
    logits_diff_file = "/home/changqing/workspaces/Overseas_review-master/summary/logits_diff/best_with_0.9483.txt"
    with open(logits_diff_file,"w") as diff_file:
        with open(old_logits_file) as old_file:
            old_logits = old_file.readlines()
        with open(new_logits_file) as new_file:
            new_logits = new_file.readlines()
        assert len(new_logits) == len(old_logits), print("error")
        print(old_logits[:5])
        logits_diff = []
        for i in range(len(old_logits)):
            if int(old_logits[i].split("\t")[-1]) != int(new_logits[i].split("\t")[-1]):
                diff_logit = old_logits[i].split("\t")[0] + "\t" + old_logits[i].split("\t")[-1].strip() + "\t" + new_logits[i].split("\t")[-1].strip() + "\t" + old_logits[i].split("\t")[1] + "\t" + new_logits[i].split("\t")[1] + "\n"
                diff_file.write(diff_logit)

    print("Done")

def generate_diff_invitation(old_logits_file,new_logits_file):
    with open(old_logits_file) as old_file:
        old_logits = old_file.readlines()
    with open(new_logits_file) as new_file:
        new_logits = new_file.readlines()
    assert len(new_logits) == len(old_logits), print("error")
    print(old_logits[:5])
    invitation_diff = []
    for i in range(len(old_logits)):
        if int(old_logits[i].split("\t")[-1]) != int(new_logits[i].split("\t")[-1]):
            invitation_diff.append(old_logits[i].split("\t")[0].split("/")[-2])

    invitation_diff_set = set(invitation_diff)
    print(invitation_diff_set)


if __name__ == "__main__":
    old_logits_file = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    new_logits_file = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt"
    # generate_logits_diff(old_logits_file,new_logits_file)
    generate_diff_invitation(old_logits_file,new_logits_file)
