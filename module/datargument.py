#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :datargument.py
# @Author   :omega
# @Time     :2020/8/4 0:17

import math
import random
from PIL import Image,ImageFilter

import torch
import cv2
import numpy as np
from torchvision.transforms import *

"""
    Cutout是一种新的正则化方法。原理是在训练时随机把图片的一部分减掉，这样能提高模型的鲁棒性。
它的来源是计算机视觉任务中经常遇到的物体遮挡问题。通过cutout生成一些类似被遮挡的物体，不仅可以让模
型在遇到遮挡问题时表现更好，还能让模型在做决定时更多地考虑环境(context)。
"""


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=100):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


"""
    Random erasing[6]其实和cutout非常类似，也是一种模拟物体遮挡情况的数据增强方法。
区别在于，cutout是把图片中随机抽中的矩形区域的像素值置为0，相当于裁剪掉，random erasing
是用随机数或者数据集中像素的平均值替换原来的像素值。而且，cutout每次裁剪掉的区域大小是固定的，
Random erasing替换掉的区域大小是随机的。
"""


class RandomErasing(object):
    """
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0] * 255
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1] * 255
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2] * 255
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0] * 255
                return img
        return img


"""
    Mixup[10]是一种新的数据增强的方法。Mixup training，就是每次取出2张图片，然后将它们线性组合，得到新的图片，
以此来作为新的训练样本，进行网络的训练。

for (images, labels) in train_loader:
    l = np.random.beta(mixup_alpha, mixup_alpha)
    index = torch.randperm(images.size(0))
    images_a, images_b = images, images[index]
    labels_a, labels_b = labels, labels[index]

    mixed_images = l * images_a + (1 - l) * images_b
    outputs = model(mixed_images)
    loss = l * criterion(outputs, labels_a) + (1 - l) * criterion(outputs, labels_b)
    acc = l * accuracy(outputs, labels_a)[0] + (1 - l) * accuracy(outputs, labels_b)[0]
"""

class HSVArgument(object):
    def __init__(self,sat=1.5,exp=1.5,hue=0.1):

        self.dhue = self.rand_uniform_strong(-hue, hue)
        self.dsat = self.rand_scale(sat)
        self.dexp = self.rand_scale(exp)

    def __call__(self, image):

        hsv_src = cv2.cvtColor(np.array(image).astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
        hsv = cv2.split(hsv_src)
        hsv[1] *= self.dsat
        hsv[2] *= self.dexp
        hsv[0] += 179 * self.dhue
        hsv_src = cv2.merge(hsv)
        hsved_img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)
        return hsved_img

    def rand_scale(self,s):
        scale = self.rand_uniform_strong(1, s)
        if random.randint(0, 1) % 2:
            return scale
        return 1. / scale

    def rand_uniform_strong(self,min, max):
        if min > max:
            swap = min
            min = max
            max = swap
        return random.random() * (max - min) + min

class GaussianNoise(object):
    def __init__(self, mean=0,std=100):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img)
        # noise = np.array(img.shape)
        noise = img.copy()
        cv2.randn(noise, self.mean, self.std)
        img = img + noise
        img = np.clip(img, 0, 255)
        return Image.fromarray(img)


class RandomBlur(object):
    def __init__(self, ksize=7):
        self.ksize = ksize

    def __call__(self, img):
        if random.randint(0, 1):
            return img.filter(ImageFilter.GaussianBlur(radius=5))
            #cv2.GaussianBlur(np.array(img), (self.ksize, self.ksize), 0)
        return img




