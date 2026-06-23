import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os

class Conv3_Bn_LeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_LeakyRelu2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(self.refpadding(x))))
    
class Conv3_Tanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Tanh2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(self.refpadding(x)))
    
class Conv3_Bn_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_Relu2d, self).__init__()
        self.ref_padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.ref_padding(x))))
     
class Feature_reconstruction(nn.Module):
    def __init__(self, dim):
        super(Feature_reconstruction, self).__init__()
        self.conv3_1 = Conv3_Bn_LeakyRelu2d(dim, dim)
        self.conv3_2 = Conv3_Bn_LeakyRelu2d(dim, dim // 2)  
        self.conv3_3 = Conv3_Bn_LeakyRelu2d(dim // 2, dim // 4) 
        self.conv3_4 = Conv3_Bn_LeakyRelu2d(dim // 4, dim // 8) 
        self.conv3_5 = Conv3_Tanh2d(dim // 8, 1) 
        
    def forward(self,feature):
        conv3_1 = self.conv3_1(feature)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        conv3_5 = self.conv3_5(conv3_4)
        return conv3_5
    
