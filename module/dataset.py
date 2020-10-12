import torch
from PIL import Image

from torch.utils.data import Dataset
from utils.util import get_tfms
from utils.img_aug import image_data_augmentation
import cv2
import numpy as np

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


class AlignCollate(object):

    def __init__(self, mode, imgsize):
        self.mode = mode
        self.imgH, self.imgW = imgsize

        assert self.mode in ["train", "val"], print("mode should be one of train or val]")
        self.tfms = get_tfms(self.imgH, self.imgW, self.mode)


    def __call__(self, batch_imgs_info):
        imgs_data = []
        imgs_path = []
        imgs_label = []
        imgs_defficty = []

        for imginfo in batch_imgs_info:
            [image, label_, deffict_degree] = imginfo
            try:
                # PIL获得的图像是RGB格式的   通过img.size属性获得图片的（宽，高）
                # cv2获得的图像是BGR格式的   通过img.shpae属性获得图片的（高，宽）
                # 在经过tfms之前先将图片转换为ndarray
                # img = cv2.imread(image, flags=cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img,self.imgH, self.imgW)
                # if self.mode == "train":
                #     img = image_data_augmentation(img)
                # tfms 中的Resize等要求输入是PIL Image 不能是ndarray
                # img = self.tfms(Image.fromarray(img)).unsqueeze(0)
                img = self.tfms(Image.open(image).convert("RGB").resize((self.imgH,
                                                                         self.imgW))).unsqueeze(0)
                imgs_data.append(img)
                imgs_label.append(torch.tensor([int(label_)]))
                imgs_path.append(image)
                imgs_defficty.append(torch.tensor([deffict_degree]))
            except Exception as ex:
                # print(ex)
                # print(img)
                continue
        imgs_defficty_tensors = torch.cat(imgs_defficty, 0)
        imgs_data_tensors = torch.cat(imgs_data, 0)
        imgs_label_tensors = torch.cat(imgs_label, 0)
        return imgs_data_tensors, imgs_label_tensors, imgs_path, imgs_defficty_tensors


class _Dataset(Dataset):
    def __init__(self, datalist):
        images = []
        for l_append in datalist:
            l_append.append(0.0)  # 附加一个困难程度的指标
            images.append(l_append)
        # <class 'list'>: [['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/00177701.png', '0', 0.0],
        #       ['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/sex_toy_add_000786.png', '0', 0.0]]
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class DataFactory(object):

    def __init__(self, datalist, model_config, mode="train") -> None:
        super().__init__()
        # batchsize = model_config["batch_size"], mode = "train", inputsize = model_config["input_size"]
        self.datalist = datalist
        self.input_shape = model_config["input_shape"]
        self.batch_size = model_config["batch_size"]
        self.mode = mode
        self.AlignCollate = AlignCollate(self.mode, self.input_shape)

    def get_dataloader(self):
        #num_workers=max(int(self.batch_size / 4), 2)
        data_ = _Dataset(self.datalist)
        data_loader = torch.utils.data.DataLoader(
            data_, batch_size=self.batch_size,
            shuffle=True, num_workers=4,
            collate_fn=self.AlignCollate, pin_memory=False
        )
        return data_loader
