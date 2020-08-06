"""Forward propagation and processes a test picture"""

import os
import json
import torch

from PIL import Image
from module.basemodel import Net
from utils.util import getdata_from_dictory, get_tfms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_logits(model_type, data_path, logits_path):
    """
    产生训练数据的logits（将所有数据前向传播通过网络的结果记录下来）
    :return:
    """


    # 加载配置参数
    model_config = json.load(open("./config/model_config.json"))[model_type]
    imgH, imgW = model_config["input_shape"]

    datalist = getdata_from_dictory(path=data_path)

    # 构建网络+将模型移入cuda+加载网络参数
    model = Net(model_type, device, model_config, mode="val").get_model()
    tfms = get_tfms(imgH, imgW, mode="val")
    with open(logits_path) as f:
        for img_path, img_label in datalist:
            img_tensor = tfms(Image.open(img_path).resize((imgH, imgW))).unsqueeze(0)
            img_logits = model(img_tensor).squeeze(0)
            print(img_path+'\t'+json.dumps(img_logits)+"\t"+json.dumps(img_label), file=f)



if __name__ == '__main__':

    # pytorch 卷积核的shape：[输出通道，输入通道，卷积核的高，卷积核的宽]
    # pytorch 卷积层的输出shape：[batch_size，输出通道，特征图高，特征图宽]
    # PIL获得的图像是RGB格式的   通过img.size属性获得图片的（宽，高）
    # cv2获得的图像是BGR格式的   通过img.shpae属性获得图片的（高，宽）
    # opencv 读取图片 BGR   BGR->rgb:  img = img[...,::-1]

    testimg = Image.open("testimg.jpeg")
    model_type = "unporn"
    mode = "val"

    model_config = json.load(open("config/model_config.json"))[model_type]
    classifier = Net(model_type, device, model_config, mode)

    risk, pred = classifier.predict_img_api(testimg)
    print(risk, pred)
