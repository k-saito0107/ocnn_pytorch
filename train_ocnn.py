import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy


def OCNNtrain(model, num_epochs,train_loader, device, nnscore, ocnn_loss, theta, nu):

    model_path = './weights/OCNNmodel.pth'
    print(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    logs = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    for epoch in range(1, num_epochs+1):
        if epoch == 1:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                logs = checkpoint['logs']
                r = checkpoint['r_value']
                w1 = checkpoint['w1']
                w2 = checkpoint['w2']
                if epoch == num_epochs:
                    break
            
            model.to(device)
            
        running_loss = 0

        for inputs in train_loader:
            model.train()
            inputs = inputs.to(device)

            w1, w2 = model(inputs)
            r = nnscore(inputs, w1, w2)
            loss = ocnn_loss(theta, inputs, nu, w1, w2, r)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)

        r = r.cpu().detach().numpy()
        r = np.percentile(r, q=100*nu)
        train_loss = running_loss/len(train_loader)
        
        print('Loss: {:.4f} '.format(train_loss))
        print('Epoch = %d, r = %f'%(epoch+1, r))
        '''
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'logs':logs,
                'r_value':r,
                'w1':w1,
                'w2':w2
            },model_path)
        '''
        if epoch % 10 == 0 and epoch != 0:
            print('---------------------------------------------------------------')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'logs':logs,
                'r_value':r,
                'w1':w1,
                'w2':w2
            },model_path)
            #ログを保存
            print('epoch : {}, train_loss : {}'.format(epoch, train_loss))
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('./result/OCNNlog_out.csv')
            
        
    model.to('cpu')
    return model, r, w1, w2