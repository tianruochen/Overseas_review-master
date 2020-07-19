import torch
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


class DatafromList(Dataset):

    def __init__(self, datalist, mode="train", imgsize=(380, 380)) -> None:
        super().__init__()
        self.datalist = datalist
        self.imgsize = imgsize
        if mode == "train":
            self.tfms = transforms.Compose(
                [transforms.Resize(int(self.imgsize[0] * 1.06)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomCrop(self.imgsize),
                 transforms.ToTensor(),
                 transforms.Normalize(means, stds)]
            )
        else:
            self.tfms = transforms.Compose(
                [transforms.Resize(self.imgsize),
                 transforms.ToTensor(),
                 transforms.Normalize(means, stds)]
            )

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, index: int) -> T_co:
        img = Image.open(self.datalist[index][0]).convert("RGB")
        data = self.tfms(img)
        label = self.datalist[index][1]
        return data, label
