import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import cv2


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        
        return img


class Resize_frame():
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        re_img = cv2.resize(img, (self.width, self.height))

        return re_img


class Resize():
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __call__(self, img):
        re_img = img.resize((self.width, self.height))

        return re_img



class Scale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):

        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        # 拡大倍率をランダムに設定
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[幅][高さ]
        scaled_h = int(height * scale)  # img.size=[幅][高さ]

        # 画像のリサイズ
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)


        # 画像を元の大きさに
        # 切り出し位置を求める
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            

        else:
            # input_sizeよりも短い辺はpaddingする

            img_original = img.copy()
            

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

        return img


'''
class RandomRotation():
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):

        # 回転角度を決める
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # 回転
        img = img.rotate(rotate_angle, Image.BILINEAR)

        return img

'''
'''
class RandomMirror(object):
    """50%の確率で左右反転させるクラス"""

    def __call__(self, img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
        return img

'''
