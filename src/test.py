import codecs
import os
import json
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn
from keras.models import load_model, Model
import efficientnet.model as eff_model
from tqdm import tqdm
import imageio
from PIL import ImageFile, Image
import webp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


def trans_array(input_array):
    result = []
    for line in input_array:
        result.append(line)
    return np.array(result)


class ModelsClassify(object):
    def __init__(self, model_path, network=None):
        if network == "B5":
            self.model = load_model(model_path,
                                    custom_objects={'ConvKernalInitializer': eff_model.ConvKernalInitializer})
            self.input_shape = (456, 456)

        elif network == "B4":
            self.model = load_model(model_path,
                                    custom_objects={'ConvKernalInitializer': eff_model.ConvKernalInitializer})
            self.input_shape = (380, 380)

        elif network == 'xcep':
            self.model = load_model(model_path)
            self.input_shape = (299, 299)

    def preprocess(self, image):
        image = image.resize(self.input_shape)
        image_array = np.asarray(image).astype(np.float32)
        image_array *= 1.0 / 255.0

        return image_array

    def predict_img_with_split_distribution(self, img_path, hw_rate=1.0, max_frame=5):
        if hw_rate >= 2.0:
            raise ValueError("invalid parameter of [hw_rate], it should in [1, 2)")
        try:
            im = Image.open(img_path)
            if im.format == "GIF":
                return self.predict_gif(img_path)
            elif im.format == "WEBP":
                return self.predict_webp(img_path)
            im = im.convert("RGB")
            img_w, img_h = im.size
        except:
            return None

        h_w = img_h / img_w
        img_list = []
        if h_w >= 2.0:
            split_len = int(img_w * hw_rate)
            h_div_w = img_h / split_len
            split_num = int(min(max_frame, np.ceil(h_div_w)))

            split_stride = int((img_h - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
                img_list.append(self.preprocess(t_img))

        elif h_w <= 0.5:
            split_len = int(img_h * hw_rate)
            h_div_w = img_w / split_len
            split_num = int(min(max_frame, np.ceil(h_div_w)))

            split_stride = int((img_w - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
                img_list.append(self.preprocess(t_img))
        else:
            uni_img = self.preprocess(im)
            img_list.append(uni_img)
        pred = self.model.predict(np.array(img_list))

        return pred

    def predict_webp(self, img_path, max_sample=5):
        im_list = webp.load_images(img_path, "RGB")
        num_im = len(im_list)
        img_array = []
        if num_im <= max_sample:
            for i in range(num_im):
                img_array.append(self.preprocess(im_list[i]))
        else:
            gif_stride = (num_im - 1) / (max_sample - 1)
            for i in range(max_sample):
                img_array.append(self.preprocess(im_list[int(np.round(i * gif_stride))]))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)
        return pred

    def predict_gif(self, img_path, max_sample=5):
        try:
            frames = np.array(imageio.mimread(img_path, memtest=False))
            frame_shape_axis = len(np.shape(frames))
            if frame_shape_axis < 4:
                frames = np.expand_dims(frames, axis=-1)
                frames = np.concatenate([frames, frames, frames], axis=-1)
            elif frame_shape_axis > 4:
                raise IOError("strange file {}".format(img_path))
            else:
                channel_num = np.shape(frames)[-1]
                if channel_num == 4:
                    frames = frames[:, :, :, :3]
                elif channel_num == 1:
                    frames = np.concatenate([frames, frames, frames], axis=-1)
                elif channel_num != 3:
                    raise IOError("strange file {}".format(img_path))
        except:
            return None
        num_frame = len(frames)
        img_array = []
        if num_frame <= max_sample:
            for i in range(num_frame):
                img_array.append(self.preprocess(Image.fromarray(frames[i])))

        else:
            gif_stride = (num_frame - 1) / (max_sample - 1)
            for i in range(max_sample):
                img_array.append(self.preprocess(Image.fromarray(frames[int(np.round(i * gif_stride))])))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        return pred


if __name__ == "__main__":

    # file_path = "/data1/zhaoshiyu/porn_result_temp/porn0229_suffix_result.txt"
    # 后接分类器的训练数据路径  （EfficientNet的输出）
    suffix_file_path = "/data1/zhaoshiyu/porn_result_temp/porn_0620/softmax/porn0620_softmax_suffix_train_data_result.txt"
    root_dir = os.path.dirname(suffix_file_path)
    model_name = suffix_file_path.split('/')[-1].split('_suffix')[0]     # porn0620
    pickle_path = '/home/changqing/workspaces/Overseas_classification-master/EfficientNet_Simple/{}_lr.pickle'.format(model_name)

    porn_labels = ['common_porn', 'cartoon_porn']
    sexy_labels = ['cartoon_sexy', 'female_sexy']

    false_recall_rate = 0.018      #误召率

    fr = codecs.open(suffix_file_path, 'r', 'utf-8').read().split('\n')
    print(fr[:5])
    # print(json.loads(fr[22393].split("\t")[2]))
    print("Trianing data : ",len(fr))        #44720

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
        #类型，图片名称，训练数据[[11维的向量]..]
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
    train_array = total_array[: split_point]       #30603
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

    test_len = len(y_pred)       # 13116

    raw_stat = [0, 0, 0]   # 预测的违规图片中porn的数量，预测的违规图片中sexy的数量,预测的违规图片数，
    new_stat = [0, 0, 0]   # 测试集中porn的数量，测试集中sexy的数量,测试集总量
    for i in range(test_len):
        temp_sample = test_sample[i]
        new_pred = y_pred[i]
        if new_pred > 0:
            new_stat[2] += 1        #预测的违规数量
            if temp_sample == porn_weight:
                new_stat[0] += 1        #预测的违规图片中porn的数量
            elif temp_sample == sexy_weight:
                new_stat[1] += 1        #预测的违规图片中sexy的数量

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
    print('='*60)
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
    fsense = codecs.open(porn_norm_file, 'r','utf-8').read().split('\n')
    print(len(fsense))   # 4999
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

        if len(np.array(pred).shape)>1:
            pred = np.max(json.loads(pred_str), axis=0)
            if len(pred)==11:
                sense_pred.append(pred)
        elif len(pred)==11:
            sense_pred.append(pred)
        else:
            continue

        i = i+1
    print('='*60)
    print("海外 normal examples：")
    print(sense_pred[:2])
    sense_pred = np.array(sense_pred)
    print('numbers of test porn_normal : ',len(sense_pred))
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
    print('='*100)


    #===============================================================
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
            sense_pred.append(pred[None,:])
        elif len(pred) < 11:
            continue
    sense_pred = np.concatenate(sense_pred,axis = 0)
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
    for i,line in enumerate(fsense):
        tline = line.strip()
        if len(tline) == 0:
            continue
        img_name, data_type, pred_str = tline.split('\t')
        pred = json.loads(pred_str)
        pred = np.array(pred)

        if len(pred.shape) > 1:
            video_count += 1

            sense_scores = clf2.predict_proba(pred)[:, 1]
            temp_sum = np.sum(sense_scores>threshold)
            if temp_sum:
                # print(sense_scores>threshold)
                injudge_img += temp_sum
                injudge_count +=1
                # print(line)
        elif len(pred) == 11:
            image_count += 1
            sense_scores = clf2.predict_proba(pred)
            if sense_scores>threshold:
                injudge_count += 1
        elif len(pred) < 11:
            # print(line)
            invalid_sense +=1
            continue
        sense_count += 1

    print('=' * 60)

    print("total sense: ",len(fsense))  # 4999
    print("video count: ",video_count)
    print("total injudge images:", injudge_img)
    print("image count: ",image_count)
    print("invalid sense count:",invalid_sense)

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