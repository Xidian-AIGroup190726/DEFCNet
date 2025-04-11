#111
#! /home/ai/anaconda3/envs/hzh/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ResNet import *
from cupath import *

class ResNet_1(nn.Module):
    def __init__(self, num_classes,input_size,choose):
        super(ResNet_1, self).__init__()
        self.start = ResNet_Start_single(input_size)
        self.end = ResNet_End_single(512, num_classes,choose)

        self.model1 = ResBlk(64, 64)
        self.down1 = downSample(64, 128)
        self.model2 = ResBlk(128, 128)


    def forward(self,data, labels=None):
        data = self.start(data)
        
        data = self.model1(data)
        data = self.down1(data)
        data = self.model2(data)

        out,feature,fea = self.end(data)

        return out,feature,fea

class ResNet_1_ms(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_1_ms, self).__init__()
        self.start = ResNet_Start_single(4)
        self.end = ResNet_End_single(512, num_classes,'ms4')

        self.model1 = ResBlk(64, 64)
        self.se1 = SEBlock(64)
        self.down1 = downSample(64, 128)
        self.model2 = ResBlk(128, 128)
        self.se2 = SEBlock(128)  # 在第二个残差块后加入 SE 模块


    def forward(self,data, labels=None):
        data = self.start(data)
        
        data = self.model1(data)
        data = self.se1(data)  # 加入通道注意力机制
        data = self.down1(data)
        data = self.model2(data)
        data = self.se2(data)  # 加入通道注意力机制

        out,feature,fea = self.end(data)

        return out,feature,fea

class ResNet_1_pan(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_1_pan, self).__init__()
        self.start = ResNet_Start_single(1)
        self.end = ResNet_End_single(512, num_classes,'pan')

        self.model1 = ResBlk_Spatial(64, 64)
        self.down1 = downSample(64, 128)
        self.model2 = ResBlk_Spatial(128, 128)


    def forward(self,data, labels=None):
        data = self.start(data)
        
        data = self.model1(data)
        data = self.down1(data)
        data = self.model2(data)

        out,feature,fea = self.end(data)

        return out,feature,fea

if __name__ == '__main__':
    model = ResNet_1_pan(11)
    data = torch.randn(64,1,64,64)
    out,feature,fea = model(data)
    print(out.shape)
    print(feature.shape)
    print(fea.shape)