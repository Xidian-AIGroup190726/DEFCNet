#! /home/ai/anaconda3/envs/hzh/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys
import torch.nn as nn
# from libtiff import TIFF
from tifffile import imread
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
from collections import Counter
import torch.optim as optim
from tqdm import tqdm
from ResNet import *
from train_model import *
from ResNet_1 import *
from ResNet_2 import *
from cupath import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import warnings
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from stage1 import *
warnings.simplefilter(action='ignore')
np.random.seed(42)  # 设置 NumPy 的随机种子



#核心参数
data_path = 'xian_big'

if data_path == 'xian_big':
    ms_num_classes = 22
    pan_num_classes = 23
elif data_path == 'huhehaote':
    ms_num_classes = 28
    pan_num_classes = 15
elif data_path == 'nanjing':
    ms_num_classes = 13
    pan_num_classes = 13
elif data_path == 'shanghai':
    ms_num_classes = 13
    pan_num_classes = 12
elif data_path == 'beijing':
    ms_num_classes = 19
    pan_num_classes = 14

lambda1 = 0.5

parser = argparse.ArgumentParser(description='long_tail')
parser.add_argument('--LR', type=float, nargs='+',default=[0,1e-4,1e-5,1e-5], help="学习率")
parser.add_argument('--train_amount', type=int, nargs='+', default=[0,2200,300000,1000,666], help='huhehaote')
parser.add_argument('--rate', type=float, default=1.1, help='huhehaote')
parser.add_argument('--EPOCH', type=int, nargs='+', default=[0,15,20,25], help="训练多少轮次")

parser.add_argument('--BATCH_SIZE', type=int, default=1, help="每次喂给的数据量")
parser.add_argument('--ms4_patch_size', type=int, default=16)
parser.add_argument('--pan_patch_size', type=int, default=64)


parser.add_argument('--ms4_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/ms4.tif')
parser.add_argument('--pan_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/pan.tif')
parser.add_argument('--train_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/train.npy')
parser.add_argument('--test_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/test.npy')
parser.add_argument('--model1_path', type=str, default=f'{CURRENT_DIR}/newwork4/model/model_part_1.pth')
parser.add_argument('--model2_path', type=str, default=f'{CURRENT_DIR}/newwork4/model/model_part_2.pth')
parser.add_argument('--model3_path', type=str, default=f'{CURRENT_DIR}/newwork4/model/model_part_3.pth')
parser.add_argument('--model_path', type=str, default=f'{CURRENT_DIR}/newwork4/model/')
global args
args = parser.parse_args()

#[11,1437,1483,729,469,2011,1115,583,774,773,156]   200
#[train_amount,1437,1483,729,2159,2011,1115,1014,774,773,738] 原版
     
# 读取图片
ms4_np = imread(args.ms4_path)
pan_np = imread(args.pan_path)

label_train = np.load(args.train_path)
label_train = label_train.astype(np.uint8) 
label_test = np.load(args.test_path)
label_test = label_test.astype(np.uint8) 

label_train = label_train - 1
label_test = label_test - 1

# 图像补零 (给图片加边框）
#导入参数
train_amount = args.train_amount
ms4_patch_size = args.ms4_patch_size
pan_patch_size = args.pan_patch_size
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(ms4_patch_size / 2 - 1), int(ms4_patch_size / 2),
                                                int(ms4_patch_size / 2 - 1), int(ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print(f'补零后的ms图的形状:', np.shape(ms4_np))
top_size, bottom_size, left_size, right_size = (int(pan_patch_size / 2 - 4), int(pan_patch_size / 2),
                                                int(pan_patch_size / 2 - 4), int(pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print(f'补零后的pan图的形状:', np.shape(pan_np))


label_element, element_count = np.unique(label_train, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('类标：', label_element.tolist()[:-1])
print('各类样本数：', element_count.tolist()[:-1])
Categories_Number = len(label_element) - 1  # 数据的类别数
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_train)  # 获取标签图的行、列

#[728, 1437, 1483, 729, 2159, 2011, 1115, 1014, 774, 773, 738] 原始

element_remain = element_count.tolist()[:-1] + [0] * 11
element_use = [0] * 100  
num_classes = Categories_Number

'''归一化图片'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy_train = np.array([[]] * num_classes * 20).tolist()   # [[],[],[],[],[],[],[]]  7个类别
ground_xy_test = np.array([[]] * num_classes * 20).tolist()  

#记录所有有标注像素点(训练集、测试集)
for row in range(label_row):  # 行
    for column in range(label_column):
        if label_train[row][column] != 255:
            ground_xy_train[int(label_train[row][column])].append([row, column])     # 记录属于每个类别的位置集合

for row in range(label_row):  # 行
    for column in range(label_column):
        if label_test[row][column] != 255:
            ground_xy_test[int(label_test[row][column])].append([row, column])     # 记录属于每个类别的位置集合

#类内打乱(训练集、测试集)
for categories in range(num_classes):
    ground_xy_train[categories] = np.array(ground_xy_train[categories])
    shuffle_array = np.arange(0, len(ground_xy_train[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy_train[categories] = ground_xy_train[categories][shuffle_array]

for categories in range(num_classes):
    ground_xy_test[categories] = np.array(ground_xy_test[categories])
    shuffle_array = np.arange(0, len(ground_xy_test[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy_test[categories] = ground_xy_test[categories][shuffle_array]

#测试集操作
ground_xy_t2 = []
label_t2 = []
for categories in range(num_classes):
    categories_number = len(ground_xy_test[categories])
    for i in range(categories_number):
        ground_xy_t2.append(ground_xy_test[categories][i])
    label_t2 = label_t2 + [categories for x in range(categories_number)]
        

ground_xy_t1 = []
label_t1 = []

for categories in range(num_classes):
    categories_number = len(ground_xy_train[categories])
    for i in range(categories_number):
        ground_xy_t1.append(ground_xy_train[categories][i])
    label_t1 = label_t1 + [categories for x in range(categories_number)]

ground_xy_t1,label_t1 = dataset_build(ground_xy_t1, label_t1)
ground_xy_t2,label_t2 = dataset_build(ground_xy_t2, label_t2)



# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
print(pan.shape)
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道
print(ms4.shape)

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


# 加载 teacher_ms 模型
# teacher_ms = ResNet_1(29,4,'ms4')  
teacher_ms = ResNet_1_ms(ms_num_classes) 
teacher_ms = teacher_ms.cuda()
teacher_ms.load_state_dict(torch.load(f'./model/{data_path}/teacher_ms.pth'))
teacher_ms.eval()  # 切换到评估模式，若要用于推理

# 加载 teacher_pan 模型
# teacher_pan = ResNet_1(16,1,'pan')  
teacher_pan = ResNet_1_pan(pan_num_classes)
teacher_pan = teacher_pan.cuda()
teacher_pan.load_state_dict(torch.load(f'./model/{data_path}/teacher_pan.pth'))
teacher_pan.eval()  # 切换到评估模式，若要用于推理

print("Stage2")
stu = ResNet_2(num_classes).cuda()
optimizer_stu = optim.Adam(stu.parameters(), lr=args.LR[3])
loss_mse1 = torch.nn.MSELoss(size_average=True, reduce=True)


train_dataset = MyData(ms4, pan, label_t1, ground_xy_t1, ms4_patch_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False, num_workers=0)
test_dataset = MyData(ms4, pan, label_t2, ground_xy_t2, ms4_patch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=0)

for epoch in range(args.EPOCH[3]):
    stu.train()  # 将模型设置为训练模式
    correct = 0.0
    total_loss = 0.0
    total = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False, 
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                    colour='cyan')
    
    for step, (ms, pan, labels, _) in enumerate(train_bar):
        ms, pan, labels = ms.cuda(), pan.cuda(), labels.cuda()  # 将数据迁移到GPU

        # 禁用教师模型的梯度计算，启用学生模型的梯度计算
        requires_grad(teacher_ms, False)
        requires_grad(teacher_pan, False)
        requires_grad(stu, True)

        

        # 获取模型输出
        outputs, fea_2, fea_3 = stu(ms, pan, labels)
        pred_train = outputs.max(1, keepdim=True)[1]  # 获取预测类别

        
        _, feature_ms, fea_h = teacher_ms(ms)
        _, feature_pan, fea_t = teacher_pan(pan)

        fea_target = torch.cat([fea_h,fea_t],1)
        mlp = MLP().cuda()
        fea_target = mlp(fea_target)
        # fea_target = torch.cat([feature_ms,feature_pan],1)


        # 计算标准分类损失（交叉熵损失）
        loss = F.cross_entropy(outputs, labels.long())
        # fea3 = F.softmax(fea_3, dim=1)
        # fea_traget = F.softmax(fea_target, dim=1)
        # loss1 = F.kl_div(fea_3, fea_target, reduction='batchmean')
        loss1 = loss_mse1(fea_2, fea_target)
        loss_all = loss + lambda1 * loss1
        optimizer_stu.zero_grad()  # 清空梯度
        loss_all.backward()
        optimizer_stu.step()

        # 更新训练准确率
        correct += pred_train.eq(labels.view_as(pred_train)).sum().item()  # 计算正确的预测数量
        total_loss += loss.item() * ms.size(0)  # 计算当前batch的总损失
        
        # 更新进度条
        train_bar.set_description(f"Epoch[{epoch}]")
        train_bar.set_postfix(train_loss=total_loss / (step + 1), train_acc=correct * 100.0 / (total + ms.size(0)))
        
        total += ms.size(0)  # 累计处理的样本数量

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = correct * 100.0 / len(train_loader.dataset)
    
    print(f"Epoch[{epoch}] Train Accuracy: {avg_acc:.3f}, Avg Loss: {avg_loss:.4f}")

class_mapping_all = {}
for i in range(num_classes):
    class_mapping_all[i] = i
average_correct_rate, class_correct_rate = test_model(stu, test_loader, num_classes, class_mapping_all, num_classes)

print("训练完成！")
exit(0)