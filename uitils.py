import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutil
import cv2
# import pytorch_msssim
from loss_ssim import *
import torchvision as tv

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).to(input.device)
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).to(input.device)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
     """
    return torch.clamp(value, min=min, max=max)

def randrot(img):
    mode = np.random.randint(0,4)
    return rot(img,mode)

def randfilp(img):
    mode = np.random.randint(0,3)
    return flip(img,mode)

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    """
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(B, C, W, H)
    out = clamp(out)
    return out


class Semantic_Loss(nn.Module):
     def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(Semantic_Loss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

     def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
    
def Int_Loss(fused_image, vis_image, inf_image, w1_vis):

    loss_Int: object = F.l1_loss(fused_image,inf_image) + w1_vis * F.l1_loss(fused_image, vis_image)
    return loss_Int


def gradinet_Loss(fused_image, vis_image, inf_image):
   
    gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_image)))

    return gradinet_loss

def SSIM_Loss(fuse_image, vis_image, inf_image):

    weight_A = weight_B = 0.5
    loss_out = weight_A * ssim(vis_image, fuse_image) + weight_B * ssim(inf_image, fuse_image) 
    return loss_out

def draw_features(width, height, x):
 
        img = x[0, 0, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  
        img = img.astype(np.uint8)  
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  
        img = img[:, :, ::-1]  
        plt.imshow(img)
 

def Tensor_to_img(feature_map):
    feature_map = feature_map.cpu().detach().numpy()
    im = feature_map[0, :, :, :]
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    return im

def draw_cnn(feature_map, model):
    im = Tensor_to_img(feature_map)
    plt.imshow(im[:, :, 0], cmap=model)


def disp_feature_image(im, model, x, y, title):
    plt.subplot(2, x, y)
    if model == 'feature':
        draw_cnn(im, 'viridis')
    elif model == 'img':
        draw_cnn(im, 'gray')
    else :
        draw_features(128, 128, im)
    plt.axis('off')
    plt.title(title, fontsize=12, y=-0.12)
    plt.subplots_adjust(top=1, bottom=0.07, left=0, right=1, hspace=0.12, wspace=0.02)
    plt.margins(0, 0)

def disp_image(im, model, x, y):
    plt.figure(figsize=(x, y))
    if model == 'feature':
        draw_cnn(im, 'viridis')
    elif model == 'img':
        draw_cnn(im, 'gray')
    else :
        draw_features(128, 128, im)
    plt.axis('off')
    plt.tight_layout()
 
    plt.margins(0, 0)
    plt.show()

def build_schedule(start, epoch, x):
    if epoch < x:

        return start * (0.1 ** (epoch // 20))  
    else:
        return 0.001 * (0.1 ** ((epoch - 100) // 10))

