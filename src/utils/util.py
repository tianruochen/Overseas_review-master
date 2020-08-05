''' Some functions to deal with data '''

import numpy as np
import cv2

from torchvision import transforms

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def get_tfms(imgH, imgW, mode="train"):
    tfms = None
    if mode == "train":
        tfms = transforms.Compose([
            transforms.Resize((int(imgH * 1.1), int(imgW * 1.1))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((imgH, imgW)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])
    elif mode == "val":
        tfms = transforms.Compose([
            transforms.Resize((int(imgH), int(imgW))),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])
    return tfms


class Averager():
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, s, c):
        self.lose_sum += s
        self.n_count += c

    def reset(self):
        self.n_count = 0
        self.lose_sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = float(self.lose_sum) / float(self.n_count)
        return res


def getdata_from_dictory(path=None):
    datalist = []
    if path is not None:
        with open(path, "r") as f:
            for line in f.readlines():
                if len(line.strip().split("\t")) == 2:
                    datalist.append(line.strip().split("\t"))
    return datalist


def getMapDict(class_num):
    map_dict = {}
    if class_num == 11:
        class_name = ["sex0", "sex1", "sex2", "sex3", "sex4", "sex5", "sex6",
                      "sex7", "sex8", "sex9", "sex10"]
    elif class_num == 4:
        class_name = ["sex0", "sex1", "sex2", "sex3"]
    for id, name in enumerate(class_name):
        dict[id] = name

    return map_dict


def pil2bgr(self, im):
    im.thumbnail((512, 512))
    rgb_img = np.array(im)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return bgr_img
