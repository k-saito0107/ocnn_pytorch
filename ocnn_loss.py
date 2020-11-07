import torch
import torchvision
import torch.nn.functional as F
import numpy as np

def nnscore(x, w, v):
    '''
    #print(x.shape,w.shape)
    r = torch.matmul(x,w)
    #print(r.shape, v.shape)
    z = torch.matmul(r, v)
    print(z.shape)
    '''
    return torch.matmul(torch.matmul(x, w), v)

def ocnn_loss(theta, x, nu, w1, w2, r):
    term1 = 0.5 * torch.sum(w1**2)
    term2 = 0.5 * torch.sum(w2**2)
    term3 = 1/nu * torch.mean(F.relu(r - nnscore(x, w1, w2)))
    term4 = -r
    
    return abs(term1 + term2 + term3 + term4)