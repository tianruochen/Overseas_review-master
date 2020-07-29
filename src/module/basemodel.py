import numpy as np
import torch
import cv2
from PIL import Image

from src.module.networks import *


class BaseModel(object):

    def __init__(self,model_name, model_config) -> None:
        super().__init__()
        self.model_name = model_name
        self.network_type = model_config['network_type']
        self.model_path = model_config['model_path']
        self.classnum = model_config['class_num']
        self.normal_axis = model_config['normal_axis']
        self.low_threshold = model_config['low_threshold']
        self.high_threshold = model_config['high_threshold']
        self.hw_rate = model_config['hw_rate']
        self.max_frames = model_config['max_frames']
        self.input_shape = model_config['input_shape']

        self.model = self.buildNetwork()

    def buildNetwork(self):
        if self.network_type == "B4":
            model = Efficietnet_b4(self.classnum)
            self.model = torch.nn.DataParallel(model).cuda()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        return model

    def preprocess(self, image):
        image = image.resize(self.input_shape)
        image_array = np.asarray(image).astype(np.float32)
        image_array *= 1.0 / 255.0
        return image_array



    def predict_img_api(self, im):
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

        pred = self.model.predict(np.array(img_list))

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        # print(suf_score)
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_score,pred.tolist()
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