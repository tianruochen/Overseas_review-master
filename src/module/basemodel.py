import numpy as np
import torch.nn as nn
import cv2
from PIL import Image

from src.utils.util import get_tfms
from src.module.networks import *


class Net(object):

    def __init__(self, model_type, device,model_config, mode="train") -> None:
        super().__init__()
        self.model_type = model_type
        self.mode = mode
        self.device = device
        self.model_arch = model_config['model_arch']
        self.temp_model_path = model_config['temp_model']
        self.best_model_path = model_config['best_model']
        self.classes_num = model_config['classes_num']
        self.normal_axis = model_config['normal_axis']
        self.low_threshold = model_config['low_threshold']
        self.high_threshold = model_config['high_threshold']
        self.hw_rate = model_config['hw_rate']
        self.max_frames = model_config['max_frames']
        self.input_shape = model_config['input_shape']

        self.tfms = get_tfms(self.input_shape[0], self.input_shape[1], self.mode)

        self.model = self.build_model()

    def build_model(self):
        model =None
        # 构建网络结构
        if self.model_arch == "B4":
            model = Efficietnet_b4(self.classes_num)
            model = nn.DataParallel(model).to(self.device)
        # 加载网络参数
        if self.mode == "train":
            if self.temp_model_path:
                model.load_state_dict(torch.load(self.temp_model_path))
            model.train()
        elif self.mode == "val":
            assert self.best_model_path is not None
            model.load_state_dict(torch.load(self.best_model_path))
            model.eval()
        return model

    def get_model(self):
        return self.model

    def preprocess(self, image):
        image_array = self.tfms(image.resize((380, 380))).unsqueeze(0)
        return image_array

    def predict_img_api(self, im):
        self.model.eval()
        try:
            img_w, img_h = im.size
            img_w = float(img_w)
            img_h = float(img_h)
            h_w = img_h / img_w
        except:
            return -1, 0

        img_list = []
        if h_w > 2.0:
            split_len = int(img_w * self.hw_rate)
            h_div_w = img_h / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))

            split_stride = int((img_h - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
                img_list.append(self.preprocess(t_img))

        elif h_w < 0.5:
            split_len = int(img_h * self.hw_rate)
            h_div_w = img_w / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))

            split_stride = int((img_w - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
                img_list.append(self.preprocess(t_img))
        else:
            img_list.append(self.preprocess(im))


        with torch.no_grad():
            pred = self.model(torch.cat(img_list, 0).cuda())
            print(type(pred))
            print(pred.shape)
        risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis].cpu().numpy(), axis=1))
        # risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis]))
        return int(risk_rate > self.low_threshold) + int(risk_rate > self.high_threshold), pred.tolist()
        # return suf_ret, pred.tolist()

    def predict_webp_api(self, im_list):
        num_im = len(im_list)
        img_array = []
        if num_im <= self.max_frames:
            for i in range(num_im):
                img_array.append(self.preprocess(im_list[i]))
        else:
            gif_stride = (num_im - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(im_list[int(np.round(i * gif_stride))]))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()

    def predict_gif_api(self, frames):
        num_frame = len(frames)
        img_array = []
        if num_frame <= self.max_frames:
            for i in range(num_frame):
                img_array.append(self.preprocess(Image.fromarray(frames[i])))
        else:
            gif_stride = (num_frame - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(Image.fromarray(frames[int(np.round(i * gif_stride))])))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()

    def predict_im_list_api(self, im_list):
        img_list = []
        for im in im_list:
            uni_img = self.preprocess(im)
            img_list.append(uni_img)
        pred = self.model.predict(np.array(img_list))

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        print(suf_score)
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()