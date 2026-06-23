from imgaug import augmenters as iaa
import math
import cv2
from PIL import Image
from skimage import exposure
import random
from torchvision import transforms as tfs
import numpy as np
import torch.nn as nn
from functools import partial

patch_size = 48
pair = lambda x: x if isinstance(x, tuple) else (x, x)

def random_cropping(image, target_shape=patch_size, is_random=True):
    # print("shape:",image.shape)
    # image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    target_h, target_w = target_shape
    height, width = image.shape

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

    zeros = image[start_y:start_y + target_h, start_x:start_x + target_w]
    return zeros

def random_resize(img, probability=0.5, minRatio=0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio_h = random.uniform(minRatio, 1.0)
    ratio_w = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h * ratio_h)
    new_w = int(w * ratio_w)

    img = cv2.resize(img, (new_w, new_h))
    img = cv2.resize(img, (w, h))
    return img


def augumentor_train(vis, ir):

    augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

    vis = augment_img.augment_image(vis)
    ir = augment_img.augment_image(ir)

    if np.random.random() <= 0.001:
        fake = 1
        value = 0
        vis[:, :] = value  # full fill black
        ir[:, :] = value  # full fill black

    ir = random_resize(ir)
    vis = random_resize(vis)

    return vis, ir

def crop_image(image, num, size):   # image_dir 批量处理图像文件夹 size 裁剪后的尺寸

    image_num = []
    for counter in range(0 , num):
        h, w = image.shape
        h_no = h // size
        w_no = w // size

        for row in range(0, h_no):
            for col in range(0, w_no):
                cropped_img = image[size*row : size*(row+1), size*col : size*(col+1), : ]
                image_num.append(cropped_img)
    return image_num