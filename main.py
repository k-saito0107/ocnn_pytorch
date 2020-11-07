import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import os
import os.path as osp
from PIL import Image
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv
import copy

from itertools import zip_longest
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

def main():
    path = os.getcwd()
    print(path)

    #画像pathの取得
    data_path = './data'
    train_path = osp.abspath(data_path+'/normal_img_data/')
    test_path = osp.abspath(data_path+'/abnormal_img_data/')

    train_path_img = glob(osp.join(train_path,'*.png'))
    test_path_img = glob(osp.join(test_path,'*.png'))

    #データセットの作成

    from make_dataset import Make_Dataset
    from data_augumentation import Compose,  Scale, Resize_frame
    from monitor_detection import Monitor_detection


    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    width = 960
    height = 512
    batch_size = 8

    train_transform = transforms.Compose([
        Monitor_detection(),
        Resize_frame(width, height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    test_transform = transforms.Compose([
        Monitor_detection(),
        Resize_frame(width, height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    train_dataset = Make_Dataset(img_path=train_path_img, img_transform=train_transform)
    test_dataset = Make_Dataset(img_path=test_path_img, img_transform=test_transform)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True, num_workers=2)

    #AutoEncoderのロード
    from autoencoder import AutoEncoder
    in_ch = 3
    f_out = 16

    model = AutoEncoder(in_ch=in_ch, f_out=f_out)

    #AutoEncoderの学習
    from train_ae import AEtrain
    num_epoch = 500
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    AEmodel = AEtrain(model=model, num_epochs=num_epoch,train_loader=train_loader, device=device)
    print('AutoEncoderの学習終了')

    #----------------------------------------------------------------------------------------------------
    #on-nn
    train_loader2=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True, num_workers=2)
    test_loader2=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True, num_workers=2)

    AEmodel = AEmodel.to(device)
    AEmodel.eval()

    encoder = AEmodel.encoder

    #normal encode
    normal_encode = []

    for normal_img in train_loader2:
        normal_img = normal_img.to(device)
        output = encoder(normal_img)
        
        output = output.cpu()
        output = output.detach().numpy()
        normal_encode.append(output)
    
    normal_encode = np.array(normal_encode)
    normal_encode = np.reshape(normal_encode, (normal_encode.shape[0], normal_encode.shape[2]))

    #anormaly encode
    anormaly_encode = []

    for anomaly_img in test_loader2:
        anomaly_img = anomaly_img.to(device)
        output = encoder(anomaly_img)

        output = output.cpu()
        output = output.detach().numpy()
        anormaly_encode.append(output)
        
    anormaly_encode = np.array(anormaly_encode)
    anormaly_encode = np.reshape(anormaly_encode, (anormaly_encode.shape[0], anormaly_encode.shape[2]))
    anormaly_encode = torch.FloatTensor(anormaly_encode/255.)
    anormaly_encode = anormaly_encode.to(device)


    #モデルのロード
    from ocnn_model import OCNN
    x_size = normal_encode.shape[1]
    print(x_size)
    h_size = 32
    y_size = 1

    OCNNmodel = OCNN(x_size=x_size, h_size=h_size, y_size=y_size)

    #lossの定義
    theta = np.random.normal(0, 1, h_size + h_size * x_size + 1)
    rvalue = np.random.normal(0, 1, (len(normal_encode), y_size))
    nu = 0.04

    from ocnn_loss import nnscore, ocnn_loss

    #oc-nnの学習
    from train_ocnn import OCNNtrain
    normal_encode = torch.FloatTensor(normal_encode/255.)
    train_loader_ocnn = torch.utils.data.DataLoader(normal_encode, batch_size=32, shuffle=True, num_workers=4, drop_last=True)


    num_epochs = 2000

    L_OCNNmodel, r, w1, w2 = OCNNtrain(model=OCNNmodel, num_epochs=num_epochs, train_loader=train_loader_ocnn,
                            device=device, nnscore=nnscore, ocnn_loss=ocnn_loss, theta=theta, nu=nu)
    
    normal_encode = normal_encode.to(device)
    #normal_encode = normal_encode.unsqueeze(0)
    train_score = nnscore(normal_encode, w1, w2)
    #print(train_score)
    #print(max(train_score))
    train_score = train_score.cpu().detach().numpy()-r
    print(r)
    print(train_score)
    print(max(train_score))
    print('------------------------------------------------------------------')



    test_score = nnscore(anormaly_encode, w1, w2)
    test_score = test_score.cpu().detach().numpy() - r
    print(test_score)
    print(max(test_score))
    #write decision Scores to CSV
    train_result_list = train_score.tolist()
    test_result_list = test_score.tolist()

    result_train = pd.DataFrame(
        {'train_result':train_result_list}
    )
    result_test = pd.DataFrame(
        {'test_result': test_result_list}
    )

    result_train.to_csv('./result/OCNN_result_train.csv')
    result_test.to_csv('./result/OCNN_result_test.csv')

    #plot score train
    flg = plt.figure()
    plt.plot()
    plt.title("One Class NN", fontsize="x-large", fontweight='bold')
    x_train = np.random.rand(len(train_score))
    plt.scatter(x_train, train_score, label='Normal', c='blue')
    x_test = np.random.rand(len(test_score))
    plt.scatter(x_test, test_score,label='Anomaly', c='red')
    flg.savefig('result_scatter.png')
    #%%
    ## Obtain the Metrics AUPRC, AUROC, P@10
    y_train = np.ones(train_score.shape[0])
    y_test = np.zeros(test_score.shape[0])
    y_true = np.concatenate((y_train, y_test))

    
    y_score = np.concatenate((train_score, test_score))
    #print(y_score)
    average_precision = average_precision_score(y_true, y_score)

    print('Average precision-recall score: {0:0.4f}'.format(average_precision))

    roc_score = roc_auc_score(y_true, y_score)

    print('ROC score: {0:0.4f}'.format(roc_score))

    def compute_precAtK(y_true, y_score, K = 10):

        if K is None:
            K = y_true.shape[0]

        # label top K largest predicted scores as + one's've

        idx = np.argsort(y_score)
        predLabel = np.zeros(y_true.shape)

        predLabel[idx[:K]] = 1

        prec = precision_score(y_true, predLabel)

        return prec

    prec_atk = compute_precAtK(y_true, y_score)

    print('Precision AtK: {0:0.4f}'.format(prec_atk))




if __name__ == "__main__":
    main()