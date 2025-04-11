# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from wtconv import *

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),  # 恢复通道
            nn.Sigmoid()  # 激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 全连接层并调整形状
        return x * y  # 加权输出

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大值
        # 拼接均值和最大值
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积层生成空间注意力权重
        attention = self.conv(x_cat)
        attention = self.sigmoid(attention)  # 归一化到 [0, 1]
        return x * attention  # 加权输出

class ResBlk_SE(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk_SE, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.se_block = SEBlock(ch_out, 16)

        self.extra = nn.Sequential()
        if ch_out == ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se_block(out)
        out = x + out

        return out

class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out == ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = x + out

        return out

class ResBlk_Spatial(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk_Spatial, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(ch_in, ch_out)
        self.bn1 = nn.BatchNorm2d(ch_out)
        # self.conv2 = DepthwiseSeparableConv2d(ch_out, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out == ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = x + out

        return out


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    



class ResNet_Start(nn.Module):
    def __init__(self):
        super(ResNet_Start,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, ms, pan):
        ms = F.relu(self.conv1(ms))
        pan = F.relu(self.conv2(pan))
        return ms, pan

class ResNet_Start_single(nn.Module):
    def __init__(self,input_size):
        super(ResNet_Start_single,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self,data):
        data = F.relu(self.conv(data))
        return data

class downSample(nn.Module):
    def __init__(self, input, output):
        super(downSample, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu1 = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class ResNet_End(nn.Module):
    def __init__(self,output_feature,num_classes):
        super(ResNet_End,self).__init__()
        self.outlayer1 = nn.Linear(output_feature, int(output_feature / 2))
        self.relu = nn.ReLU(True)
        self.outlayer2 = nn.Linear(int(output_feature / 2), num_classes)
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        self.pool2 = nn.AdaptiveMaxPool2d((2,2))

    def forward(self, x, y, if_feature=False):
        x = self.pool1(x) #[64,128,32,32]
        x = torch.flatten(x,start_dim=1) #[64,128,2,2]
        #[64,512]

        y = self.pool2(y) #[64,128,32,32]
        y = torch.flatten(y,start_dim=1) #[64,128,2,2]
        #[64,512]
        out = torch.cat([x, y], 1)   # 通道拼接
        #[64,1024]
        fea = out
        #print(out.shape)
        if if_feature:
            return out

        out = self.outlayer1(out)
        out = self.relu(out)
        feature = out
        out = self.outlayer2(out)
        return out,feature,fea
    
class ResNet_End_single(nn.Module):
    def __init__(self,output_feature,num_classes,choose):
        super(ResNet_End_single,self).__init__()
        self.outlayer1 = nn.Linear(output_feature, int(output_feature / 2))
        self.relu = nn.ReLU(True)
        self.outlayer2 = nn.Linear(int(output_feature / 2), num_classes)
        if choose == 'ms4':
            self.pool = nn.AdaptiveAvgPool2d((2,2))
        elif choose == 'pan':
            self.pool = nn.AdaptiveMaxPool2d((2,2))

    def forward(self, x, if_feature=False):
        x = self.pool(x) #[64,128,32,32]
        out = torch.flatten(x,start_dim=1) #[64,128,2,2]
        
        #[64,1024]
        fea = out
        #print(out.shape)
        if if_feature:
            return out

        out = self.outlayer1(out)
        out = self.relu(out)
        feature = out
        out = self.outlayer2(out)
        return out,feature,fea

class ResNet_End_WCMF(nn.Module):
    def __init__(self,output_feature,num_classes):
        super(ResNet_End_WCMF,self).__init__()
        self.outlayer1 = nn.Linear(output_feature, int(output_feature / 2))
        self.relu = nn.ReLU(True)
        self.outlayer2 = nn.Linear(int(output_feature / 2), num_classes)
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        self.pool2 = nn.AdaptiveMaxPool2d((2,2))
        self.wcmf = WCMF(128)

    def forward(self, x, y, if_feature=False):
        x = self.wcmf(x,y)

        x = self.pool1(x) #[64,128,32,32]
        x = torch.flatten(x,start_dim=1) #[64,128,2,2]
        out = x
        fea = out
        #print(out.shape)
        if if_feature:
            return out

        out = self.outlayer1(out)
        out = self.relu(out)
        feature = out
        out = self.outlayer2(out)
        return out,feature,fea        



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024,512)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x






class ResNet_End_1(nn.Module):
    def __init__(self,output_feature,num_classes):
        super(ResNet_End_1,self).__init__()
        self.outlayer1 = nn.Linear(output_feature, int(output_feature / 2))
        self.relu = nn.ReLU(True)
        self.outlayer2 = nn.Linear(int(output_feature / 2), num_classes)
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        self.pool2 = nn.AdaptiveMaxPool2d((2,2))

        self.pool3 = nn.AdaptiveAvgPool2d((2,2))
        self.pool4 = nn.AdaptiveMaxPool2d((2,2))

    def forward(self, x, y,ms1,pan1):
        x = self.pool1(x)
        x = torch.flatten(x,start_dim=1)
        

        y = self.pool2(y)
        y = torch.flatten(y,start_dim=1)

        
        # 对 ms1 和 pan1 进行插值，使其与 x 和 y 的尺寸一致
        # ms1 = F.interpolate(ms1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        # pan1 = F.interpolate(pan1, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)
        ms1 = self.pool3(ms1)
        ms1 = torch.flatten(ms1, start_dim=1)  # 展平
        pan1 = self.pool4(pan1)
        pan1 = torch.flatten(pan1, start_dim=1)  # 展平

        out = torch.cat([x, y, ms1, pan1], 1)   # 通道拼接
        #print(out.shape)

        out = self.outlayer1(out)
        out = self.relu(out)
        out = self.outlayer2(out)
        return out


class WCMF(nn.Module):
    def __init__(self, channel=128):
        super(WCMF, self).__init__()

        self.conv_d1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_d2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def fusion(self, f1, f2, f_vec):
        # 这里的 f_vec 是形状 [64, 2, 32, 32]，即具有两个通道的特征图
        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)

        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self, ms, pan):
        ms = self.conv_d1(ms)  # 通过第一个卷积处理 ms
        pan = self.conv_d2(pan)  # 通过第二个卷积处理 pan

        f = torch.cat([ms, pan], dim=1)  # 将两个特征图拼接在一起
        f = self.conv_c1(f)  # 第一个卷积层
        f = self.conv_c2(f)  # 第二个卷积层

        # 调用融合操作
        Fo = self.fusion(ms, pan, f)
        return Fo

if __name__ == '__main__':
    ms = torch.randn(64,128,32,32)
    pan = torch.randn(64,128,32,32)

    wcmf = WCMF(128)
    output = wcmf(ms,pan)
    #output的预期输出也是[64,128,32,32]
    print(output.shape)

