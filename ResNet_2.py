#! /home/ai/anaconda3/envs/hzh/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ResNet import *
from cupath import *

class ResNet_2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_2, self).__init__()
        self.start = ResNet_Start()
        self.end1 = ResNet_End_1(2048, num_classes)
        self.end = ResNet_End(1024, num_classes)

        self.model1_ms = ResBlk(64, 64)
        self.down1_ms = downSample(64, 128)
        self.model2_ms = ResBlk(128, 128)

        self.model1_pan = ResBlk(64, 64)
        self.down1_pan = downSample(64, 128)
        self.model2_pan = ResBlk(128, 128)

        self.outlayer1_1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU(True)

        self.outlayer1_2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(True)

        self.outlayer2 = nn.Linear(512, num_classes)
        self.outlayer3_1 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU(True)
        self.outlayer3_2 = nn.Linear(512, num_classes)
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        self.pool2 = nn.AdaptiveMaxPool2d((2,2))


    def forward(self, ms, pan, labels=None):
        ms, pan = self.start(ms, pan)
        
        
        ms = self.model1_ms(ms)
        ms = self.down1_ms(ms)
        ms = self.model2_ms(ms)
        ms = self.pool1(ms)
        ms = torch.flatten(ms,start_dim=1)


        pan = self.model1_pan(pan)
        pan = self.down1_pan(pan)
        pan = self.model2_pan(pan)
        pan = self.pool2(pan) #[64,128,32,32]
        pan = torch.flatten(pan,start_dim=1) #[64,128,2,2]

        fea1 = torch.cat([ms,pan],1)
        fea2 = fea1

        out1 = self.outlayer1_1(fea1)
        out1 = self.relu1(out1)
        out2 = self.outlayer1_2(fea2)
        out2 = self.relu1(out2)
        feature = out2


        #方案一，拼接再下去
        #73.9
        out = torch.cat([out1,out2],1)
        out = self.outlayer3_1(out)
        out = self.relu3(out)
        out = self.outlayer3_2(out)
        
        

        #方案二，直接相加，下去
        #寄了
        #73.26
        # out = out1+out2
        # out = self.outlayer2(out)

        #方案三，单分支直接出去，标准老师学生模型
        #74.3
        # out = self.outlayer2(out1)


        return out, feature, fea2



# # 创建模型实例
# model = ResNet_2(num_classes=7)  # 假设有 7 个类别
# model.cuda()

# # 定义输入张量的形状
# input_ms = torch.randn(1, 4, 815, 845).cuda()  # MS 图像输入 (batch_size, channels, height, width)
# input_pan = torch.randn(1, 1, 3260, 3380).cuda()  # PAN 图像输入 (batch_size, channels, height, width)

# # 使用 torchinfo 来统计参数数量
# summary(model, input_data=[input_ms, input_pan], col_names=["input_size", "output_size", "num_params"])

if __name__ == '__main__':
    model = ResNet_2(11)
    ms = torch.randn(64,4,16,16)
    pan = torch.randn(64,1,64,64)
    out = model(ms, pan)
    print(out.shape)