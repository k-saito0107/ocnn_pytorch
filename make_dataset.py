import torch
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

class Make_Dataset(data.Dataset):
    def __init__(self, img_path, img_transform):
        self.img_path = img_path
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        img_file_path = self.img_path[index]
        img = Image.open(img_file_path)
        #img = cv2.imread(img_file_path)
        img = self.pil2cv(img)
        img = self.img_transform(img)
        return img
    
    def pil2cv(self, image):
        ''' PIL型 -> OpenCV型 '''
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

