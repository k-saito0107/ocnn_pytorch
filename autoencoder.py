import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU()
        
       
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, f_out):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, f_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(f_out, f_out*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(f_out*2, f_out*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out*4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(f_out*4, f_out*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out*8),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(f_out*8, f_out*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out*16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(f_out*16, f_out*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f_out*32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        

    

        self.layer7 = nn.Sequential(
            nn.Linear(61440, 2048),
            nn.Dropout(0.4),
            nn.ELU(),
            nn.Linear(2048,32)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x=x.view(-1, 61440)
        x = self.layer7(x)

        return x



class Decoder(nn.Module):
    def __init__(self, in_ch, f_out):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(32, 2048),
            nn.Dropout(0.4),
            nn.ELU(),
            nn.Linear(2048, 61440)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(f_out*32, f_out*16, kernel_size=2, stride=2),
            nn.BatchNorm2d(f_out*16),
            nn.ELU()
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(f_out*16, f_out*8, kernel_size=2, stride=2),
            nn.BatchNorm2d(f_out*8),
            nn.ELU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(f_out*8, f_out*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(f_out*4),
            nn.ELU()
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(f_out*4, f_out*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(f_out*2),
            nn.ELU()
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(f_out*2, f_out, kernel_size=2, stride=2),
            nn.BatchNorm2d(f_out),
            nn.ELU()
        )

        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(f_out, in_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_ch),
            nn.ELU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), 512, 8,15)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_ch, f_out):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(in_ch, f_out)
        self.decoder = Decoder(in_ch, f_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

