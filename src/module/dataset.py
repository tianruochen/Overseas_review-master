import torch
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

class AlignCollate(object):

    def __init__(self, mode, imgsize):
        self.mode = mode
        self.imgH, self.imgW = imgsize

        assert self.mode in ["train", "val"], print("mode should be one of train or val]")

        if self.mode == "train":
            self.tfms = transforms.Compose([
                transforms.Resize(int(self.imgH*1.1, self.imgW*1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(self.imgH, self.imgW),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.mode == "val":
            self.tfms = transforms.Compose([
                transforms.Resize(int(self.imgH, self.imgW)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

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
                img = self.tfms(Image.open(image).convert("RGB").resize((self.imgH,
                                                                         self.imgW))).unsqueeze(0)
                imgs_data.append(img)
                imgs_label.append(torch.tensor([int(label_)]))
                imgs_path.append(image)
                imgs_defficty.append(torch.tensor([deffict_degree]))
            except Exception as ex:
                print(ex)
                continue
        imgs_defficty_tensors = torch.cat(imgs_defficty, 0)
        imgs_data_tensors = torch.cat(imgs_data, 0)
        imgs_label_tensors = torch.cat(imgs_label, 0)
        return imgs_data_tensors, imgs_label_tensors, imgs_path, imgs_defficty_tensors


class _Dataset(Dataset):
    def __init__(self, datalist):
        images = []
        for l_append in datalist:
            l_append.append(0.0)   #附加一个困难程度的指标
            images.append(l_append)
        #<class 'list'>: [['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/00177701.png', '0', 0.0],
        #       ['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/sex_toy_add_000786.png', '0', 0.0]]
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class DataFactory(object):

    def __init__(self, datalist, batchsize=32, mode="train", imgsize=(380, 380)) -> None:
        super().__init__()
        self.datalist = datalist
        self.imgsize = imgsize
        self.batch_data = batchsize
        self.mode = mode
        self.AlignCollate = AlignCollate(self.mode,self.imgsize)


    def getBatchData(self):
        data_ = _Dataset(self.datalist)
        self.batch_data = torch.utils.data.DataLoader(
            data_, batch_size=self.batch_data,
            shuffle=True, num_workers=max(int(self.batch_size/4), 2),
            collate_fn=self.AlignCollate, pin_memory=False
        )
