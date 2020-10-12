#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :gen_test.py.py
# @Time     :2020/9/30 上午11:06
# @Author   :Chang Qing
 
def generate_txt_by_name(name):
    if name == "normal":
        base_file = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
        txt_name = "normal_test.txt"
    elif name == "unnorm":
        base_file = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt"
        txt_name = "unnorm_test.txt"
    txt_path = "/home/changqing/workspaces/Overseas_review-master/data/" + txt_name
    with open(txt_path,"w") as dst:
        with open(base_file) as src:
            for line in src.readlines():
                info = line.split("\t")[0] + "\t" + "-1" + "\n"
                dst.write(info)


if __name__ == "__main__":
    cocofun_name = ["normal", "unnorm"]
    for name in cocofun_name:
        generate_txt_by_name(name)
    print("Generate Done!")


