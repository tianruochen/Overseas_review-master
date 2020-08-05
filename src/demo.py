import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from src.module.basemodel import Net
# import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # testimg = cv2.imread("testimg.jpeg")
    testimg = Image.open("testimg.jpeg")
    model_type = "porn"
    mode = "val"
    # Porn_classification = PornModel('porn_cocofun')
    # risk,pred = Porn_classification.predict_img_api(testimg)

    # Unporn_classification = UnPorn_Model('unpron_cocofun')
    model_config = json.load(open("config/model_config.json"))[model_type]
    classifier = Net(model_type, device, model_config, mode)
    # net = classifier.getNetwork()
    # assert model_config["best_model"] is not None
    # net = torch.nn.DataParallel(net)
    # net.load_state_dict(torch.load(model_config["best_model"]))

    risk, pred = classifier.predict_img_api(testimg)
    print(risk, pred)
