import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os


def AEtrain(model, num_epochs,train_loader, device):

    model_path = './weights/AEmodel.pth'
    print(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logs = []

    for epoch in range(1, num_epochs+1):
        print(epoch)
        if epoch == 1:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                logs = checkpoint['logs']

                if epoch == num_epochs:
                    break
            
            model.to(device)
            
        running_loss = 0
        
        for img in train_loader:
            model.train()
            img = img.to(device)
            outputs = model(img)
            loss = criterion(img, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*img.size(0)
        
        train_loss = running_loss/len(train_loader)
        
        if epoch % 10 == 0 and epoch != 0:
            print('---------------------------------------------------------------')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'logs':logs
            },model_path)

            #ログを保存
            print('epoch : {}, train_loss : {}'.format(epoch, train_loss))
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('./result/AElog_out.csv')
            
        
    model.to('cpu')
    return model
