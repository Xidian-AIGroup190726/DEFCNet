#! /home/ai/anaconda3/envs/hzh/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ResNet import *
from cupath import *
from torch.nn import functional as F

class ResNet_3(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_3, self).__init__()
        self.start = ResNet_Start()
        # self.end = ResNet_End_CAFM(1024, num_classes)
        self.end = ResNet_End_WCMF(512, num_classes)

        self.model1_ms = ResBlk(64, 64)
        self.down1_ms = downSample(64, 128)
        self.model2_ms = ResBlk(128, 128)

        self.model1_pan = ResBlk(64, 64)
        self.down1_pan = downSample(64, 128)
        self.model2_pan = ResBlk(128, 128)



    def forward(self, ms, pan, labels=None):
        ms, pan = self.start(ms, pan)

        ms = self.model1_ms(ms)
        ms = self.down1_ms(ms)
        ms = self.model2_ms(ms)

        pan = self.model1_pan(pan)
        pan = self.down1_pan(pan)
        pan = self.model2_pan(pan)
        #上采样
        ms = F.interpolate(ms, size=(32, 32), mode='bilinear', align_corners=False)
        #下采样(我不想使用)
        # pan = F.adaptive_avg_pool2d(pan, (8, 8))  # 或者使用AdaptiveMaxPool2d

        out,feature,fea = self.end(ms, pan)

        return out,feature,fea




if __name__ == '__main__':
    model = ResNet_3(11)
    ms = torch.randn(64,4,16,16)
    pan = torch.randn(64,1,64,64)
    out = model(ms, pan)
    print(out.shape)