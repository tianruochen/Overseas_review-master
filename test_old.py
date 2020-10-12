"""validate model on some validation datasets"""

import codecs
import os
import json
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import tqdm
import torch.nn as nn
from keras.models import load_model, Model
# import efficientnet.model as eff_model
from tqdm import tqdm
import imageio
from PIL import ImageFile, Image
# import webp
import sklearn
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from module.basemodel import Net
from utils.validation import generate_logits

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"


def trans_array(input_array):
    result = []
    for line in input_array:
        result.append(line)
    return np.array(result)



#########################################
# 评估数据集
val_unporn_dataset = {
    "cocofun_normal_path" : "/data1/zhaoshiyu/cocofun_normal",
    "cocofun_unnorm_path" : "/data/wangruihao/serious_data/kill_image",
    #### video #########
    "cocofun_disgust_path": "/data/wangruihao/serious_data/disgusting",
    "cocofun_sensitive_path": "/data/wangruihao/serious_data/sensitive"
}
########################################


def validate_pron():
    """pron 后接分类器的建模评估"""
    # file_path = "/data1/zhaoshiyu/porn_result_temp/porn0229_suffix_result.txt"
    # 后接分类器的训练数据路径  （EfficientNet的输出）
    suffix_file_path = "/data1/zhaoshiyu/porn_result_temp/porn_0620/softmax/porn0620_softmax_suffix_train_data_result.txt"
    root_dir = os.path.dirname(suffix_file_path)
    model_name = suffix_file_path.split('/')[-1].split('_suffix')[0]  # porn0620
    pickle_path = '/home/changqing/workspaces/Overseas_classification-master/EfficientNet_Simple/{}_lr.pickle'.format(
        model_name)

    porn_labels = ['common_porn', 'cartoon_porn']
    sexy_labels = ['cartoon_sexy', 'female_sexy']

    false_recall_rate = 0.018  # 误召率

    fr = codecs.open(suffix_file_path, 'r', 'utf-8').read().split('\n')
    print(fr[:5])
    # print(json.loads(fr[22393].split("\t")[2]))
    print("Trianing data : ", len(fr))  # 44720

    train_rate = 0.7
    porn_weight = 0.5
    sexy_weight = 0.2
    normal_weight = 0.5
    rand_seed = 11
    # raw_threshold = 0.339

    x_array = []
    y_array = []
    # i = 0
    sample_weight = []
    for line in fr:
        # i = i+1
        # print(i)       #22394
        tline = line.strip()
        if tline == "":
            continue
        # 类型，图片名称，训练数据[[11维的向量]..]
        items = tline.split('\t')
        temp_x = np.max(json.loads(items[2]), axis=0).tolist()
        label = items[0]
        x_array.append(temp_x)
        if label != 'normal':
            if label in porn_labels:
                sample_weight.append(porn_weight)
            elif label in sexy_labels:
                sample_weight.append(sexy_weight)
            else:
                continue
            y_array.append(1)
        else:
            y_array.append(0)
            sample_weight.append(normal_weight)

    total_array = np.array(list(zip(x_array, y_array, sample_weight)))
    print(total_array.size)
    np.random.seed(rand_seed)
    np.random.shuffle(total_array)

    split_point = int(len(total_array) * train_rate)
    train_array = total_array[: split_point]  # 30603
    test_array = total_array[split_point:]
    train_x = train_array[:, 0]
    train_x = trans_array(train_x)
    train_y = train_array[:, 1].astype(int)
    train_sample = train_array[:, 2]
    test_x = test_array[:, 0]
    test_x = trans_array(test_x)
    test_y = test_array[:, 1].astype(int)
    test_sample = test_array[:, 2]

    LR = LogisticRegression()
    LR.fit(train_x, train_y, sample_weight=train_sample)
    result = LR.score(test_x, test_y, sample_weight=test_sample)
    print(result)
    train_res = LR.score(train_x, train_y, sample_weight=train_sample)
    print(train_res)

    y_pred = LR.predict(test_x)
    y_proba = LR.predict_proba(test_x)[:, 1]

    test_len = len(y_pred)  # 13116

    raw_stat = [0, 0, 0]  # 预测的违规图片中porn的数量，预测的违规图片中sexy的数量,预测的违规图片数，
    new_stat = [0, 0, 0]  # 测试集中porn的数量，测试集中sexy的数量,测试集总量
    for i in range(test_len):
        temp_sample = test_sample[i]
        new_pred = y_pred[i]
        if new_pred > 0:
            new_stat[2] += 1  # 预测的违规数量
            if temp_sample == porn_weight:
                new_stat[0] += 1  # 预测的违规图片中porn的数量
            elif temp_sample == sexy_weight:
                new_stat[1] += 1  # 预测的违规图片中sexy的数量

        raw_stat[2] += 1
        if temp_sample == porn_weight:
            raw_stat[0] += 1
        elif temp_sample == sexy_weight:
            raw_stat[1] += 1

    print(raw_stat)
    print(new_stat)

    with open(pickle_path, 'wb') as f:
        # pickle.dump(LR, f, protocol=2)
        pickle.dump(LR, f)

    with open(pickle_path, 'rb') as f:
        clf2 = pickle.load(f)
        print(clf2.coef_)

    # /data1/zhaoshiyu/porn_result_temp/porn_0620/softmax/porn0620_softmax_im_eval_dataset_result.txt
    # model_name = 'porn0620'
    # root_dir = '/data1/zhaoshiyu/porn_result_temp/porn_0620/raw/'
    im_normal_file = "{}/{}_im_normal_dataset_result.txt".format(root_dir, model_name)
    im_sense_file = "{}/{}_im_eval_dataset_result.txt".format(root_dir, model_name)
    common_porn_file = "{}/{}_common_porn_0512_result.txt".format(root_dir, model_name)
    cartoon_porn_file = "{}/{}_cartoon_porn_0512_result.txt".format(root_dir, model_name)
    porn_hub_file = "{}/{}_porn_hub_dataset_result.txt".format(root_dir, model_name)

    normal_pred = []
    fnorm = codecs.open(im_normal_file, 'r', 'utf-8').read().split('\n')
    for line in fnorm:
        tline = line.strip()
        if tline == "":
            continue
        img_name, pred_str = tline.split('\t')
        pred = np.max(json.loads(pred_str), axis=0)
        normal_pred.append(pred)
    print('=' * 60)
    print("normal pred example:")
    print(normal_pred[:2])
    normal_pred = np.array(normal_pred)
    normal_scores = clf2.predict_proba(normal_pred)[:, 1].tolist()
    normal_scores.sort(reverse=True)
    print("sorted normal scores:")
    print(normal_scores[:50])
    threshold = normal_scores[int(np.round(len(normal_scores) * false_recall_rate))]
    print("threshold = {}".format(threshold))

    # im dataset
    sense_pred = []
    fsense = codecs.open(im_sense_file, 'r', 'utf-8').read().split('\n')
    for line in fsense:
        tline = line.strip()
        if tline == "":
            continue
        img_name, pred_str = tline.split('\t')
        pred = np.max(json.loads(pred_str), axis=0)
        sense_pred.append(pred)
    sense_pred = np.array(sense_pred)
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    recall_num = np.sum(sense_scores > threshold)
    recall_rate = recall_num / len(sense_scores)

    print("im recall_num = {}".format(recall_num))
    print("im recall_rate = {:.2f} %".format(recall_rate * 100))

    # common_porn dataset
    sense_pred = []
    fsense = codecs.open(common_porn_file, 'r', 'utf-8').read().split('\n')
    for line in fsense:
        tline = line.strip()
        if tline == "":
            continue
        img_name, pred_str = tline.split('\t')
        pred = np.max(json.loads(pred_str), axis=0)
        sense_pred.append(pred)
    sense_pred = np.array(sense_pred)
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    recall_num = np.sum(sense_scores > threshold)
    recall_rate = recall_num / len(sense_scores)

    print("common_porn recall_num = {}".format(recall_num))
    print("common_porn recall_rate = {:.2f} %".format(recall_rate * 100))

    # cartoon_porn dataset
    sense_pred = []
    fsense = codecs.open(cartoon_porn_file, 'r', 'utf-8').read().split('\n')
    for line in fsense:
        tline = line.strip()
        if tline == "":
            continue
        img_name, pred_str = tline.split('\t')
        pred = np.max(json.loads(pred_str), axis=0)
        sense_pred.append(pred)
    sense_pred = np.array(sense_pred)
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    recall_num = np.sum(sense_scores > threshold)
    recall_rate = recall_num / len(sense_scores)

    print("cartoon_porn recall_num = {}".format(recall_num))
    print("cartoon_porn recall_rate = {:.2f} %".format(recall_rate * 100))

    # pornhub dataset
    sense_pred = []
    fsense = codecs.open(porn_hub_file, 'r', 'utf-8').read().split('\n')
    for line in fsense:
        tline = line.strip()
        if not tline:
            continue
        img_name, pred_str = tline.split('\t')
        pred = np.max(json.loads(pred_str), axis=0)
        sense_pred.append(pred)
    sense_pred = np.array(sense_pred)
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    recall_num = np.sum(sense_scores > threshold)
    recall_rate = recall_num / len(sense_scores)

    print("pornhub recall_num = {}".format(recall_num))
    print("pornhub recall_rate = {:.2f} %".format(recall_rate * 100))

    # 色情模型的正常数据集  max_mode
    porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
    sense_pred = []
    fsense = codecs.open(porn_norm_file, 'r', 'utf-8').read().split('\n')
    print(len(fsense))  # 4999
    print(fsense[3027])
    # print(fsense[0].split("\t")[2])
    # fs_data = np.array(json.loads(fsense[0].split("\t")[2]))
    # print(fs_data)
    # soft = nn.Softmax()
    # pred_soft = soft(torch.from_numpy(np.array(json.loads(fsense[0].split("\t")[2]))))
    # print(pred_soft)
    for line in fsense:
        # print(i)
        tline = line.strip()
        if len(tline) == 0:
            continue
        img_name, data_type, pred_str = tline.split('\t')
        pred = json.loads(pred_str)

        if len(np.array(pred).shape) > 1:
            pred = np.max(json.loads(pred_str), axis=0)
            if len(pred) == 11:
                sense_pred.append(pred)
        elif len(pred) == 11:
            sense_pred.append(pred)
        else:
            continue

        i = i + 1
    print('=' * 60)
    print("海外 normal examples：")
    print(sense_pred[:2])
    sense_pred = np.array(sense_pred)
    print('numbers of test porn_normal : ', len(sense_pred))
    # softmax = nn.Softmax(dim=1)
    # sense_pred = softmax(torch.from_numpy(sense_pred)).numpy()
    print("after softmax examples：")
    print(sense_pred[:2])
    print(clf2.predict_proba(sense_pred)[:2000, :])
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    injudge_num = np.sum(sense_scores > threshold)
    injudge_rate = injudge_num / len(sense_scores)

    print("porn normal injudge_num = {}".format(injudge_num))
    print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))
    print('=' * 100)

    # ===============================================================
    # 色情模型的正常数据集   solo模式
    porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
    sense_pred = []
    fsense = codecs.open(porn_norm_file, 'r', 'utf-8').read().split('\n')
    print(len(fsense))  # 4999

    for line in fsense:
        tline = line.strip()
        if len(tline) == 0:
            continue
        img_name, data_type, pred_str = tline.split('\t')
        pred = json.loads(pred_str)
        pred = np.array(pred)

        if len(pred.shape) > 1:
            sense_pred.append(pred)
        elif len(pred) == 11:
            sense_pred.append(pred[None, :])
        elif len(pred) < 11:
            continue
    sense_pred = np.concatenate(sense_pred, axis=0)
    print('=' * 60)

    print('numbers of test porn_normal : ', (sense_pred.shape[0]))
    # softmax = nn.Softmax(dim=1)
    # sense_pred = softmax(torch.from_numpy(sense_pred))
    # print(clf2.predict_proba(sense_pred)[:2000, :])
    sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    injudge_num = np.sum(sense_scores > threshold)
    injudge_rate = injudge_num / len(sense_scores)

    print("porn normal injudge_num = {}".format(injudge_num))
    print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))

    # ===============================================================
    # 色情模型的正常数据集   帖子召回模式
    porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
    sense_pred = []
    fsense = codecs.open(porn_norm_file, 'r', 'utf-8').read().split('\n')
    print(len(fsense))  # 4999

    sense_count = 0
    injudge_count = 0
    invalid_sense = 0
    video_count = 0
    image_count = 0
    injudge_img = 0
    for i, line in enumerate(fsense):
        tline = line.strip()
        if len(tline) == 0:
            continue
        img_name, data_type, pred_str = tline.split('\t')
        pred = json.loads(pred_str)
        pred = np.array(pred)

        if len(pred.shape) > 1:
            video_count += 1

            sense_scores = clf2.predict_proba(pred)[:, 1]
            temp_sum = np.sum(sense_scores > threshold)
            if temp_sum:
                # print(sense_scores>threshold)
                injudge_img += temp_sum
                injudge_count += 1
                # print(line)
        elif len(pred) == 11:
            image_count += 1
            sense_scores = clf2.predict_proba(pred)
            if sense_scores > threshold:
                injudge_count += 1
        elif len(pred) < 11:
            # print(line)
            invalid_sense += 1
            continue
        sense_count += 1

    print('=' * 60)

    print("total sense: ", len(fsense))  # 4999
    print("video count: ", video_count)
    print("total injudge images:", injudge_img)
    print("image count: ", image_count)
    print("invalid sense count:", invalid_sense)

    # print('numbers of test porn_normal : ', (sense_pred.shape[0]))
    # softmax = nn.Softmax(dim=1)
    # sense_pred = softmax(torch.from_numpy(sense_pred))
    # print(clf2.predict_proba(sense_pred)[:2000, :])
    # sense_scores = clf2.predict_proba(sense_pred)[:, 1]

    injudge_rate = injudge_count / sense_count

    print("porn normal injudge_num = {}".format(injudge_count))
    print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))


    # im recall_num = 5060
    # im recall_rate = 98.44 %
    # common_porn recall_num = 1389
    # common_porn recall_rate = 97.27 %
    # cartoon_porn recall_num = 1411
    # cartoon_porn recall_rate = 95.14 %
    # pornhub recall_num = 8278
    # pornhub recall_rate = 95.83 %
    # porn normal injudge_rate = 4.60 %
    # porn normal injudge_rate = 6.25 %
    # porn normal injudge_rate = 21.80 %


def validate_on_normal_dataset():
    """pron 后接分类器的建模评估"""
    # file_path = "/data1/zhaoshiyu/porn_result_temp/porn0229_suffix_result.txt"
    # 后接分类器的训练数据路径  （EfficientNet的输出）
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt"
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_normal/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt"
    root_dir = os.path.dirname(suffix_file_path)
    model_name = suffix_file_path.split('/')[-1].split('.')[0]  # porn0620
    recall_mode = "invitation"

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with open(suffix_file_path, 'r') as f:
        file_info_list = f.read().split("\n")

        print("Trianing data : ", len(file_info_list))  # 44720
        print(file_info_list[:5])
        invitation_map = {}
        # img_score_list = []
        logits = []

        for line in file_info_list:
            if not line:
                continue

            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = None
            if recall_mode == "image":
                logit = np.array(json.loads(line[1])).min(axis=0)
            else:
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

        for threshold in thresholds:
            if recall_mode == "image":
                recall_logits = logits[logits[:, 2] > threshold]
                recall = recall_logits.shape[0] / logits.shape[0]
                print("under threshold {}, image recall: {}--{}/{}".format(threshold, recall, recall_logits.shape[0],
                                                                           logits.shape[0]))
            elif recall_mode == "invitation":
                recall_count = 0
                for _, invitation_logits in invitation_map.items():
                    if invitation_logits.min(axis=0)[2] > threshold:
                        recall_count += 1
                recall = recall_count / len(invitation_map.keys())
                print("under threshold {}, invitation recall: {}--{}/{}".format(threshold, recall, recall_count,
                                                                         len(invitation_map.keys())))

        # false_recall_rates = [0.0 + i * 0.02 for i in range(1, 20)]  # 误召率列表
        # print(false_recall_rates)
        # logits_score = np.sort(logits[:, 2])
        # print(logits_score[:5])
        # thresholds = [logits_score[int(i * logits.shape[0])] for i in false_recall_rates]
        #
        # print(thresholds)
        # old_threshold = [0.00010102284431923181, 0.0015352037735283375, 0.0033047317992895842, 0.015809779986739159,
        #                  0.051457248628139496, 0.12242656201124191, 0.21415027976036072, 0.33594521880149841,
        #                  0.45518290996551514, 0.58477669954299927, 0.6957353949546814, 0.78998690843582153,
        #                  0.85753726959228516, 0.90254783630371094, 0.93244779109954834, 0.95385324954986572,
        #                  0.96944737434387207, 0.97890317440032959, 0.98287522792816162]
        # new_threshold = [0.031195204704999924, 0.079025104641914368, 0.13231100142002106, 0.18247915804386139,
        #                  0.26748287677764893, 0.34285911917686462, 0.40871742367744446, 0.46329742670059204,
        #                  0.51341342926025391, 0.55436861515045166, 0.59521442651748657, 0.63195407390594482,
        #                  0.6621696949005127, 0.69205981492996216, 0.71700799465179443, 0.73953217267990112,
        #                  0.76203876733779907, 0.78064656257629395, 0.7974780797958374]


def validate_on_unnorm_dataset():
    injudge_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5218451534241008, 0.6, 0.7, 0.8, 0.9]

    thresholds = [0.0003747180162463337, 0.012226594612002373, 0.1024278774857521, 0.26573309302330017, 0.6236545443534851,0.7000325322151184, 0.9046652317047119, 0.9856488704681396, 0.9989489912986755, 0.9999810457229614]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/best_accuracy_4_class_b4_accuracy_adl_0_380.pth_logits.txt"

    thresholds = [0.05091641843318939, 0.18457616865634918, 0.330279678106308, 0.45077529549598694, 0.5733217597007751, 0.5984936356544495, 0.6890443563461304, 0.7873456478118896, 0.869635283946991, 0.9359068274497986]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unpron_cla_4_epoch_28_acc_0.9309_auc_0.9910.pth_logits.txt"

    thresholds = [8.558211266063154e-05, 0.004656499717384577, 0.03373356908559799, 0.18091733753681183, 0.5178699493408203, 0.5939021706581116, 0.8185585737228394, 0.9460698366165161, 0.9924302697181702, 0.9995730519294739]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"

    thresholds = [0.0002356839831918478, 0.007734728045761585, 0.06221406161785126, 0.22103212773799896, 0.5040110945701599, 0.592111349105835, 0.8422273397445679, 0.9716606736183167, 0.9972276091575623, 0.9999384880065918]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_1_acc_0.9458_auc_0.9926.pth_logits.txt"

    thresholds = [0.0003415944520384073, 0.009462394751608372, 0.07157686352729797, 0.2247319519519806, 0.5195491313934326,0.5960209369659424, 0.8404300212860107, 0.9723348617553711, 0.9967570900917053, 0.9999319314956665]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_2_acc_0.9458_auc_0.9926.pth_logits.txt"

    thresholds = [0.00036213966086506844, 0.010133186355233192, 0.07039403915405273, 0.19305504858493805, 0.5107679963111877, 0.5834735631942749, 0.8280558586120605, 0.9697278141975403, 0.9962018132209778, 0.9999088048934937]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_6_acc_0.9475_auc_0.9926.pth_logits.txt"
    #
    thresholds = [0.0002692318521440029, 0.0075437892228364944, 0.07093621790409088, 0.2460172325372696, 0.531326949596405, 0.5920621156692505, 0.845272421836853, 0.9725262522697449, 0.9975792765617371, 0.999945878982544]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_1_acc_0.9466_auc_0.9927.pth_logits.txt"
    #
    thresholds = [0.00024105433840304613, 0.007441721856594086, 0.06367805600166321, 0.20070111751556396, 0.49033495783805847, 0.5616140365600586, 0.8280057907104492, 0.9666893482208252, 0.9967689514160156, 0.9999200105667114]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_2_acc_0.9483_auc_0.9927.pth_logits.txt"

    thresholds = [8.558211266063154e-05, 0.004656499717384577, 0.03373356908559799, 0.18091733753681183, 0.5178699493408203, 0.5939021706581116, 0.8185585737228394, 0.9460698366165161, 0.9924302697181702, 0.9995730519294739]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_15_acc_0.9415_auc_0.9933.pth_logits.txt"

    thresholds = [0.0002474015927873552, 0.007550599053502083, 0.0652749165892601, 0.22173309326171875, 0.4914308190345764, 0.5730454325675964, 0.8327168226242065, 0.9704394936561584, 0.9972658157348633, 0.9999428987503052]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_3_acc_0.9483_auc_0.9927.pth_logits.txt"

    thresholds = [0.03970589116215706, 0.08720466494560242, 0.17782683670520782, 0.30580008029937744, 0.5273655652999878, 0.5680807828903198, 0.702741801738739, 0.8121428489685059, 0.8724690079689026, 0.9129164814949036]
    suffix_file_path = "/home/changqing/workspaces/Overseas_review-master/summary/logits/unporn/cocofun_unnorm/unporn_class_4_epoch_98_acc_0.9441_auc_0.9913.pth_logits.txt"

    root_dir = os.path.dirname(suffix_file_path)
    model_name = suffix_file_path.split('/')[-1].split('.')[0]  # porn0620
    recall_mode = "invitation"


    with open(suffix_file_path, 'r') as f:
        file_info_list = f.read().split("\n")

        print("Trianing data : ", len(file_info_list))
        print(file_info_list[:5])
        invitation_map = {}
        # img_score_list = []
        logits = []

        for line in file_info_list:
            if not line:
                continue

            line = line.split('\t')
            invitation_name = os.path.dirname(line[0])
            logit = None
            if recall_mode == "image":
                logit = np.array(json.loads(line[1])).min(axis=0)
            else:
                logit = np.array(json.loads(line[1])).min(axis=0)
            if invitation_name not in invitation_map:
                invitation_map[invitation_name] = logit[None, :]
            else:
                invitation_map[invitation_name] = np.concatenate([invitation_map[invitation_name], logit[None, :]],
                                                                 axis=0)
                # cmd + backspace 删除当前行
            logits.append(logit)

        logits = np.array(logits)
        print(suffix_file_path)
        for threshold in thresholds:
            if recall_mode == "image":
                recall_logits = logits[logits[:, 2] > threshold]
                recall = recall_logits.shape[0] / logits.shape[0]
                print("under threshold {}, image false recall: {}--{}/{}".format(threshold, recall, recall_logits.shape[0],
                                                                           logits.shape[0]))
            elif recall_mode == "invitation":
                recall_count = 0
                for _, invitation_logits in invitation_map.items():
                    if invitation_logits.min(axis=0)[2] > threshold:
                        recall_count += 1
                recall = recall_count / len(invitation_map.keys())
                print("injudge: {},  threshold {}, invitation outgassing rate recall: {}--{}/{}".format(
                    injudge_ratio[thresholds.index(threshold)],threshold, recall, recall_count,
                                                                         len(invitation_map.keys())))

        # false_recall_rates = [0.0 + i * 0.02 for i in range(1, 20)]  # 误召率列表
        # print(false_recall_rates)
        # logits_score = np.sort(logits[:, 2])
        # print(logits_score[:5])
        # thresholds = [logits_score[int(i * logits.shape[0])] for i in false_recall_rates]
        # print(thresholds)

def validate_unpron():
        # validate_on_normal_dataset()

        validate_on_unnorm_dataset()



#########################################
if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "unporn"
    mode = "val"

    if model_type == "unporn":
        validate_unpron()

    elif model_type == "porn":
        validate_pron()

