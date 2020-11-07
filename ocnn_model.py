import torch.nn as nn
import torch.nn.functional as F

class OCNN(nn.Module):
    def __init__(self, x_size, h_size, y_size):
        super(OCNN, self).__init__()
        self.linear1 = nn.Linear(x_size, h_size)
        self.linear2 = nn.Linear(h_size, y_size)
    
    def forward(self, img):
        w1 = self.linear1(img)
        w2 = self.linear2(w1)

        return w1, w2