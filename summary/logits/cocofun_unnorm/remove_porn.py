import json
import glob
import os

porn_dir = "/data/wangruihao/serious_data/porn"

porn_name_list = glob.glob("/data/wangruihao/serious_data/porn/*.mp4") + glob.glob("/data/wangruihao/serious_data/porn/*.jpg")
print(len(porn_name_list))
porn_name_list = [name.split('.')[0].split('/')[-1] for name in porn_name_list]
print(porn_name_list[:5])



new_logits_info = []

with open("unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt") as f:
    for logit in f.readlines():
        if logit.split("\t")[0].split('/')[-2] not in porn_name_list:
            new_logits_info.append(logit)

print(len(new_logits_info))
with open("unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt","w") as f:
    for logit in new_logits_info:
        f.write(logit)