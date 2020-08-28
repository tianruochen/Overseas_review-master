''' Some functions to deal with data '''

import numpy as np
import cv2
from module.datargument import *

from torchvision import transforms

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def get_tfms(imgH, imgW, mode="train"):
    tfms = None
    if mode == "train":
        tfms = transforms.Compose([
            transforms.Resize((int(imgH * 1.3), int(imgW * 1.3))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((imgH, imgW)),
            # RandomBlur(),
            # GaussianNoise(),
            # HSVArgument(),
            # RandomErasing(),
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


def pil2bgr(im):
    im.thumbnail((512, 512))
    rgb_img = np.array(im)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return bgr_img




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

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale

def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def random_blur(img, ksize=7):
    if random.randint(0,1):
        dst = cv2.GaussianBlur(img,(ksize,ksize),0)
    return dst


def gaussian_noise(img, mean=0, std = 127):
    noise = np.array(img)
    std = random.randint(5, std)
    cv2.randn(noise, mean, std)
    dst = img + noise
    return dst


def hsv_augment(img, sat=1.5, exp=1.5, hue=0.1):
    dhue = rand_uniform_strong(-hue, hue)
    dsat = rand_scale(sat)
    dexp = rand_scale(exp)
    hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
    hsv = cv2.split(hsv_src)
    hsv[1] *= dsat
    hsv[2] *= dexp
    hsv[0] += 179 * dhue
    hsv_src = cv2.merge(hsv)
    hsved_img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)
    return hsved_img


def cutout(img, n_holes=1, length = 70):
    h = img.shape[0]
    w = img.shape[1]
    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.

    # mask = torch.from_numpy(mask)
    # mask = mask.expand_as(img)
    img = img * mask

    return img


def random_erasing(img, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
    if random.randint(0,1):
        return img
    for attempt in range(100):
        area = img.shape[0] * img.shape[2]
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            if img.size()[0] == 3:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0] * 255
                img[1, x1:x1 + h, y1:y1 + w] = mean[1] * 255
                img[2, x1:x1 + h, y1:y1 + w] = mean[2] * 255
            else:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0] * 255
            return img
    return img

def image_data_augmentation(img):

    # 随机模糊
    img = random_blur(img)

    # 随机gaussian噪声
    img = gaussian_noise(img)

    # 色域变化
    img = hsv_augment(img)

    # cutout
    img = cutout(img)

    # random_erasing
    img = random_erasing(img)

    return img

