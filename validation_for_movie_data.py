import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
import cv2
from data_augumentation import Compose,  Scale, Resize_frame

from autoencoder import AutoEncoder
from ocnn_model import OCNN
from train_ocnn import OCNNtrain
from ocnn_loss import nnscore, ocnn_loss


def main():
    file_path = './movie_data/ショベルモニタ_正常.MOV'
    cap = cv2.VideoCapture(file_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frame)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(device)

    #パラメータの設定
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    width = 960
    height = 512

    transform = transforms.Compose([
    Resize_frame(width, height),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    #-------------モデルのロード-------------------------------------------#
    ae_model_path = './weights/AEmodel.pth'
    ocnn_model_path = './weights/OCNNmodel.pth'

    #AutoEncoderのロード
    in_ch = 3
    f_out = 16
    ae_model = AutoEncoder(in_ch=in_ch, f_out=f_out)
    checkpoint_ae = torch.load(ae_model_path)
    ae_model.load_state_dict(checkpoint_ae['model_state_dict'])
    ae_model.to(device)
    ae_model.eval()
    encoder = ae_model.encoder

    #OC-NNのロード
    checkpoint_ocnn = torch.load(ocnn_model_path)
    r = checkpoint_ocnn['r_value']
    w1 = checkpoint_ocnn['w1']
    w2 = checkpoint_ocnn['w2']

    #_, frame = cap.read()
    result_list = []

    for i in range(1, total_frame):
        _, frame = cap.read()
        img = frame
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = encoder(img)
        score = nnscore(output, w1, w2)
        result_score = score.cpu().detach().numpy()
        result_score = result_score[0][0]
        result_list.append(result_score)
    
    print(result_list)
    result = pd.DataFrame({'result':result_list})
    result.to_csv('./val_result/ショベルモニタ_正常_result.csv')




if __name__ == '__main__':
    main()

