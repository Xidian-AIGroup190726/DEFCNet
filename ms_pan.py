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
if_save = 0
IDIS = 1


if data_path == 'huhehaote':
    ms_threshold = 200
    pan_threshold = 200
elif data_path == 'nanjing':
    ms_threshold = 600
    pan_threshold = 600
elif data_path == 'xian_big':
    ms_threshold = 1000
    pan_threshold = 1000
elif data_path == 'shanghai':
    ms_threshold = 10000
    pan_threshold = 10000
elif data_path == 'beijing':
    ms_threshold = 10000
    pan_threshold = 10000

parser = argparse.ArgumentParser(description='long_tail')
parser.add_argument('--LR', type=float, nargs='+',default=[0,1e-4,1e-5,1e-5], help="学习率")
parser.add_argument('--train_amount', type=int, nargs='+', default=[0,2200,300000,1000,666], help='huhehaote')
parser.add_argument('--rate', type=float, default=1.1, help='huhehaote')
parser.add_argument('--EPOCH', type=int, nargs='+', default=[0,15,20,20], help="训练多少轮次")

parser.add_argument('--BATCH_SIZE', type=int, default=1, help="每次喂给的数据量")
parser.add_argument('--ms4_patch_size', type=int, default=16)
parser.add_argument('--pan_patch_size', type=int, default=64)


parser.add_argument('--ms4_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/ms4.tif')
parser.add_argument('--pan_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/pan.tif')
parser.add_argument('--train_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/train.npy')
parser.add_argument('--test_path', type=str, default=f'{CURRENT_DIR}/data/{data_path}/test.npy')
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

    




#画出训练集的分布
train_dataset = MyData(ms4, pan, label_t1, ground_xy_t1, ms4_patch_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=0)

# 提取所有ms数据和标签
all_ms_data = []
all_pan_data = []
all_labels = []
all_location = []
    


for data in train_loader:
    ms_data,pan_data, labels, locate_xy = data  # 修改解包方式

    ms_data_flatten = ms_data.view(ms_data.size(0), -1)  # 展平每个图像，但保留batch_size
    pan_data_flatten = pan_data.view(pan_data.size(0), -1)

    all_ms_data.append(ms_data_flatten.numpy())
    all_pan_data.append(pan_data_flatten.numpy())
    all_labels.append(labels.numpy())
    all_location.append(locate_xy.numpy())

all_ms_data = np.concatenate(all_ms_data, axis=0)
all_pan_data = np.concatenate(all_pan_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_location = np.array(all_location)

# 使用PCA将高维数据降维
pca_ms = PCA(n_components=3)
pca_pan = PCA(n_components=3)
ms4_2d = pca_ms.fit_transform(all_ms_data)
pan_2d = pca_pan.fit_transform(all_pan_data)

# def diversity_distinction(data_input):
#     mean = 0
#     all_classes = []
#     split_class = []
#     for i in range(num_classes):
#         data = data_input[all_labels == i]
#         num_points = data.shape[0]
#         #计算平均方差
#         diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
#         squared_distances = np.sum(diff ** 2, axis=2)
#         variance = np.sum(squared_distances) / (num_points * (num_points - 1))
#         all_classes.append(variance)
#         mean += variance
#         # print(f'类别{i}的方差:',variance)

#     mean = mean / num_classes
#     for i in range(num_classes):
#         if all_classes[i] > mean:
#             split_class.append(i)
#     return split_class

import numpy as np

def diversity_distinction(data_input, all_labels, num_classes):
    all_classes = []
    split_class = []
    mean_variance = 0
    
    for i in range(num_classes):
        data = data_input[all_labels == i]
        num_points = data.shape[0]

        if num_points < 2:
            all_classes.append(0)  # 样本过少，方差设为0
            continue

        # 计算均值
        mean_vector = np.mean(data, axis=0)
        
        # 计算每个样本与均值的平方差，并求均值
        variance = np.mean(np.sum((data - mean_vector) ** 2, axis=1))
        
        all_classes.append(variance)
        mean_variance += variance

    mean_variance /= num_classes

    # 挑选方差高于均值的类别
    for i in range(num_classes):
        if all_classes[i] > mean_variance:
            split_class.append(i)

    return split_class

if IDIS == 1:
    ms_split_class = diversity_distinction(ms4_2d, all_labels, num_classes)
    pan_split_class = diversity_distinction(pan_2d, all_labels, num_classes)
elif IDIS == 0:
    ms_split_class = []
    pan_split_class = []
print(f'ms_split:{ms_split_class}')
print(f'pan_split:{pan_split_class}')


def stage1(split_class,num_classes,data_2d,all_location,all_labels,threshold,choose):

    ground_xy_train,class_mapping,num_classes_split,ground_xy_points = single_kmeans(split_class,num_classes,data_2d,all_location,all_labels)

    train_loader_all,train_loader_h,train_loader_t,\
    test_loader_all,h_num_classes,t_num_classes,new_class_mapping,class_h,class_t = \
    rebuild_dataset_and_model(ground_xy_train,
                            class_mapping,num_classes_split,
                            threshold,
                            ms4,
                            pan,
                            ground_xy_t2,
                            label_t2,
                            ms4_patch_size,
                            choose,
                            ground_xy_points,
                            data_2d,
                            all_labels,
                            all_location)

    teacher = single_train(train_loader_all,train_loader_h,train_loader_t,\
        test_loader_all,num_classes_split,num_classes,h_num_classes,\
        t_num_classes,class_mapping,class_h,class_t,args.LR[1],args.EPOCH[1],args.EPOCH[2],choose)
    
    return teacher,class_mapping

teacher_ms,class_mapping_ms = stage1(ms_split_class,num_classes,ms4_2d,all_location,all_labels,threshold=ms_threshold,choose='ms4')
if if_save == 0:
    print("teacher_ms训练成功")
elif if_save == 1:
    os.makedirs(f'./model/{data_path}', exist_ok=True)
    torch.save(teacher_ms.state_dict(), f'./model/{data_path}/teacher_ms.pth')
    print("teacher_ms模型保存成功!")


teacher_pan,class_mapping_pan = stage1(pan_split_class,num_classes,pan_2d,all_location,all_labels,threshold=pan_threshold,choose='pan')
if if_save == 0:
    print("teacher_pan训练成功")
elif if_save == 1:
    os.makedirs(f'./model/{data_path}', exist_ok=True)
    torch.save(teacher_pan.state_dict(), f'./model/{data_path}/teacher_pan.pth')
    print("teacher_pan模型保存成功!")

exit(0)



