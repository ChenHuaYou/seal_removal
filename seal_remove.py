# coding: utf-8
import os
import cv2
import sys
import time
import shutil
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from tqdm import tqdm
sys.path.append("..")
from models.experimental import *
from utils.general import non_max_suppression
from torchvision import transforms
from unet_model import UNet as Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输入图像处理
def preprocess_img(img, resize=960, maxsize=960, INPUT=(960, 960), color=(114, 114, 114)):
    h, w = img.shape[:2]
    ratio = resize / min(h, w)
    if ratio * max(h, w) > maxsize:
        ratio = maxsize / max(h, w)
    img_resize = cv2.resize(img, (int(ratio*w), int(ratio*h)))
    pw = INPUT[0] - img_resize.shape[1]
    ph = INPUT[1] - img_resize.shape[0]
    img = cv2.copyMakeBorder(img_resize, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, h, w


class SealRemove:
    def __init__(self):
        self.detect_model = attempt_load('./weights/best.pt', map_location='cpu')
        self.detect_model.float().eval().to(device)
        self.detect_classes = ['table', 'seal', 'handwriting']        
        self.Transform = transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]) 
        self.net = Net(3, 3)
        self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.load_state_dict(torch.load('./weights/300_0.0006.pth', map_location='cpu'))
        self.net.eval()
    
    def seal_remove(self, img):
        img_ratio, ratio, h, w = preprocess_img(img)
        img_ratio = cv2.cvtColor(img_ratio, cv2.COLOR_BGR2RGB)
        input = torch.from_numpy(img_ratio.transpose((2, 0 ,1))).float().div(255.0).unsqueeze(0).to(device)
        pred = self.detect_model(input, augment=False)[0] 
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0] 
        if pred is not None and len(pred) > 0:
            pred[:, :4] /= ratio
            pred[:, 0].clamp_(0, w)
            pred[:, 1].clamp_(0, h)
            pred[:, 2].clamp_(0, w)
            pred[:, 3].clamp_(0, h)
            pred = pred.detach().cpu().numpy()
            pred = pred[np.argsort(-pred[:,-2])]
            if pred is not None and len(pred) > 0:
                for i, bb in enumerate(pred):
                    if int(bb[-1]) != 1:
                        continue
                    label = self.detect_classes[int(bb[-1])] 
                    score = str(bb[-2])[:6]
                    x1, y1, x2, y2 = list(map(int, bb[:4]))
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if cx + 256 >= w:
                        cx = w - 256
                    if cx - 256 <= 0:
                        cx = 256
                    if cy + 256 >= h:
                        cy = h - 256
                    if cy - 256 <= 0:
                        cy = 256
                    crop_img = img[cy-256:cy+256, cx-256:cx+256,:]
                    input = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).convert('RGB')
                    input = self.Transform(input).unsqueeze(0).to(device)
                    output = self.net(input)
                    output = output.cpu().detach().numpy().squeeze()
                    output[0, ...] = ((output[0, ...]*0.5) + 0.5)*255
                    output[1, ...] = ((output[1, ...]*0.5) + 0.5)*255
                    output[2, ...] = ((output[2, ...]*0.5) + 0.5)*255
                    dst = np.uint8(np.transpose(output, (1, 2, 0)))  
                    img[cy-256:cy+256, cx-256:cx+256,:] = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        return img




if __name__ == "__main__":
    model = SealRemove()
    img = cv2.imread('./test.png')
    dst = model.seal_remove(img)
    cv2.imwrite('result.jpg', dst)
