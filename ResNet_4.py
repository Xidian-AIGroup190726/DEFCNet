# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ResNet import *
from cupath import *

class ResNet_4(nn.Module):
    def __init__(self, num_classes, model1_path, model2_path, model3_path):
        super(ResNet_4, self).__init__()
        self.start = ResNet_Start()
        self.model1 = ResBlk(64, 64)
        self.down1 = downSample(64, 128)
        self.model2 = ResBlk(128, 128)
        self.down2 = downSample(128, 256)
        self.model3 = ResBlk(256, 256)
        self.down3 = downSample(256, 512)
        self.model4 = ResBlk(512, 512)
        self.end = ResNet_End(4096, num_classes)

        self.model1.load_state_dict(torch.load(model1_path, weights_onlpan=True))
        self.model2.load_state_dict(torch.load(model2_path, weights_onlpan=True))
        self.model3.load_state_dict(torch.load(model3_path, weights_onlpan=True))

        # 将model1/model2/model3的参数设置为不更新
        model_list = [self.model1, self.model2, self.model3]
        for model in model_list:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, ms, pan):
        ms, pan = self.start(ms, pan)

        for x in [ms, pan]:
            x = self.model1(x)
            x = self.down1(x)
            x = self.model2(x)
            x = self.down2(x)
            x = self.model3(x)
            x = self.down3(x)
            x = self.model4(x)

        out = self.end(ms, pan)
        return out 

    def save(self):
        pass

