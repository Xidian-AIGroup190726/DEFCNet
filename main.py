#111
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
warnings.simplefilter(action='ignore')
np.random.seed(42)  # 设置 NumPy 的随机种子


#特征迁移、特征提取器、大号参差块(特征)
#修改train_model中的class_weights，使用一个函数来动态生成weights

#核心参数
parser = argparse.ArgumentParser(description='long_tail')
parser.add_argument('--LR', type=float, nargs='+',default=[0,1e-4,1e-5,1e-5], help="学习率")
parser.add_argument('--train_amount', type=int, nargs='+', default=[0,2200,300000,1000,666], help='huhehaote')
parser.add_argument('--rate', type=float, default=1.1, help='huhehaote')
parser.add_argument('--EPOCH', type=int, nargs='+', default=[0,15,20,15], help="训练多少轮次")

parser.add_argument('--BATCH_SIZE', type=int, default=1, help="每次喂给的数据量")
parser.add_argument('--ms4_patch_size', type=int, default=16)
parser.add_argument('--pan_patch_size', type=int, default=64)

data_path = 'huhehaote'
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
ground_xy_test = np.array([[]] * num_classes * 20).tolist()   # [[],[],[],[],[],[],[]]  7个类别

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
ground_xy_t2 = np.array(ground_xy_t2)
label_t2 = np.array(label_t2)

shuffle_array = np.arange(0, len(label_t2), 1)
np.random.shuffle(shuffle_array)
label_t2 = label_t2[shuffle_array]
ground_xy_t2 = ground_xy_t2[shuffle_array]
label_t2 = torch.from_numpy(label_t2).type(torch.LongTensor)
ground_xy_t2 = torch.from_numpy(ground_xy_t2).type(torch.LongTensor)



# 训练集起始分割操作
ground_xy_train_stage1 = np.array([[]] * num_classes * 20).tolist()
for categories in range(num_classes):
    if element_remain[categories] >= train_amount[1]:
        ground_xy_train_stage1[categories].extend(ground_xy_train[categories][:train_amount[1]])
        element_remain[categories] -= train_amount[1]
        element_use[categories] += train_amount[1]
    else:
        ground_xy_train_stage1[categories].extend(ground_xy_train[categories][:element_remain[categories]])
        element_use[categories] += element_remain[categories]
        element_remain[categories] = 0
        

ground_xy_t1 = []
label_t1 = []

for categories in range(num_classes):
    categories_number = len(ground_xy_train_stage1[categories])
    for i in range(categories_number):
        ground_xy_t1.append(ground_xy_train_stage1[categories][i])
    label_t1 = label_t1 + [categories for x in range(categories_number)]

#存档c
ground_xy_t1_check = ground_xy_t1
label_t1_check = label_t1

ground_xy_t1 = np.array(ground_xy_t1)
label_t1 = np.array(label_t1)
shuffle_array = np.arange(0, len(label_t1), 1)
np.random.shuffle(shuffle_array)
label_t1 = label_t1[shuffle_array]
ground_xy_t1 = ground_xy_t1[shuffle_array]
label_t1 = torch.from_numpy(label_t1).type(torch.LongTensor)
ground_xy_t1 = torch.from_numpy(ground_xy_t1).type(torch.LongTensor)



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

    
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):

        
        self.train_data1 = MS4
        self.cut_ms_size = cut_size
    
        self.train_data2 = Pan
        self.cut_pan_size = cut_size * 4
        
        
        self.train_labels = Label
        self.gt_xy = xy
        
        

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        
        ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                y_ms:y_ms + self.cut_ms_size]

    
        pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return ms, pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)



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


class_mapping = {}
ms_split = {}
count_num = 0
if_split = [0] * 60
save = np.array([[]]*60).tolist()
for i in range(num_classes):
    class_mapping[i] = i
# 初始检查是否有任何类的样本数量小于50, threshold是尾部类分界线
min_samples = 100
threshold = 800
#[2,5,7,8,10]
#[0,1,3,4,6,9]
# not_split = [0,1,2,3,4,5,6,7,8,9,10]
not_split = [0,1,3,4,6,9]




for class_id in range(num_classes): 
    if class_id in not_split:
        class_points_ms = ms4_2d[all_labels == class_id]
        class_location = all_location[all_labels == class_id]
        class_location = class_location.squeeze(axis=1)
        ms_new_bucket = np.array([[]]).tolist()
        for idx in range(len(class_points_ms)):
            x,y = class_location[idx]
            ms_new_bucket[0].append([x,y])

        count_num+= 1
        print(f'类{class_id}不拆分:{[len(x) for x in ms_new_bucket]}')    
                
        #保存分割结果
        ms_split[class_id] = ms_new_bucket
        continue
        
    class_points_ms = ms4_2d[all_labels == class_id]
    class_location = all_location[all_labels == class_id]
    class_location = class_location.squeeze(axis=1)

    k_range = range(2, 11)  # 轮廓系数计算需要 k >= 2
    
    # 对每个 k 值进行聚类
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels_kmeans = kmeans.fit_predict(class_points_ms)
        score = silhouette_score(class_points_ms, labels_kmeans) # 计算轮廓系数
        silhouette_scores.append(score)
    best_k_ms = k_range[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=best_k_ms, random_state=42)
    labels_kmeans_ms = kmeans.fit_predict(class_points_ms)

    kmeans_ms = {}
    for i in range(best_k_ms):
        kmeans_ms[i] = class_points_ms[labels_kmeans_ms == i]


    print(f"class_{class_id}:ms:{best_k_ms}")

    sub_classes = 0
    ms_new_bucket = np.array([[]] * best_k_ms).tolist()
    for i in range(len(labels_kmeans_ms)):
        idx = labels_kmeans_ms[i]
        x,y = class_location[i]
        ms_new_bucket[idx].append([x,y])

    kmeans_ms_center = []
    
    for i in range(best_k_ms):
        center_ms = kmeans_ms[i].mean(axis=0)
        kmeans_ms_center.append(center_ms.tolist())

    print(f'类{class_id}被分割为{best_k_ms}个新类!')
    new_bucket_non_empty = [bucket for bucket in ms_new_bucket if len(bucket) > 0]
    print(f'新类的数量分别为{[len(x) for x in new_bucket_non_empty]}')

    

exit(0)



# 处理新的类别
new_num_classes = 0
for class_id in range(num_classes):
    for idx in range(len(split[class_id])):
        ground_xy_train[new_num_classes] = np.array(split[class_id][idx])
        class_mapping[new_num_classes] = class_id
        new_num_classes += 1


print(f'总共分割出{new_num_classes}个新类!')
ground_xy_all = []
label_all = []
# 重新构建数据集
for class_id in range(new_num_classes):
    class_data = ground_xy_train[class_id]
    ground_xy_all.extend(class_data.tolist())
    label_all.extend([class_id for _ in range(len(class_data))])


#测试集操作
ground_xy_test_all = []
label_test_all = []
for class_id in range(new_num_classes):
    ground_xy_test_all.extend(ground_xy_test[class_id])
    label_test_all = label_test_all + [class_id for x in range(len(ground_xy_test[class_id]))]

def dataset_build(ground_xy, label):
    ground_xy = np.array(ground_xy)
    label = np.array(label)
    shuffle_array = np.arange(0, len(label), 1)
    np.random.shuffle(shuffle_array)
    label = label[shuffle_array]
    ground_xy = ground_xy[shuffle_array]
    label = torch.from_numpy(label).type(torch.LongTensor)
    ground_xy = torch.from_numpy(ground_xy).type(torch.LongTensor)
    return ground_xy, label

ground_xy_all, label_all = dataset_build(ground_xy_all, label_all)
ground_xy_test_all, label_test_all = dataset_build(ground_xy_test_all, label_test_all)


test_dataset = MyData(ms4, pan, label_t2, ground_xy_t2, ms4_patch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)
train_dataset_all = MyData(ms4, pan, label_all, ground_xy_all, ms4_patch_size)
train_loader_all = DataLoader(dataset=train_dataset_all, batch_size=256, shuffle=False, num_workers=0)

model = ResNet_2(new_num_classes).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.LR[1])



for epoch in range(1, args.EPOCH[1] + 1):
    avg_loss = train_model(model, train_loader_all, optimizer, epoch, new_num_classes)

average_correct_rate, class_correct_rate = test_model(model, test_loader, num_classes, class_mapping, Categories_Number)





exit(0)

ground_xy_all = []
ground_xy_h = []
ground_xy_t = []
label_all = []
label_h = []
label_t = []
class_h = []
class_t = []
element_use_h = []
element_use_t = []

h_num_classes = 0
t_num_classes = 0
new_class_mapping = {}
class_sizes = [(class_id, len(ground_xy_train[class_id])) for class_id in range(new_num_classes)]
class_sizes.sort(key=lambda x: x[1], reverse=True)

# 重新构建数据集
count = 0
for class_id, _ in class_sizes:
    new_class_mapping[count] = class_mapping[class_id]
    count += 1
    class_data = ground_xy_train[class_id]
    ground_xy_all.extend(class_data.tolist())
    label_all.extend([class_id for _ in range(len(class_data))])

    # 判断该类别是否为头部类别（类别数量大于 threshold）
    if len(class_data) > threshold:
        ground_xy_h.extend(class_data.tolist())
        label_h.extend([h_num_classes for _ in range(len(class_data))])
        h_num_classes += 1
        class_h.append(class_id)
        element_use_h.append([class_id, class_mapping[class_id], len(class_data)])
    else:  # 否则是尾部类别
        ground_xy_t.extend(class_data.tolist())
        label_t.extend([t_num_classes for _ in range(len(class_data))])
        t_num_classes += 1
        class_t.append(class_id)
        element_use_t.append([class_id, class_mapping[class_id], len(class_data)])


#测试集操作
ground_xy_test_all = []
ground_xy_test_h = []
ground_xy_test_t = []
label_test_all = []
label_test_h = []
label_test_t = []
for class_id in range(new_num_classes):
    ground_xy_test_all.extend(ground_xy_test[class_id])
    label_test_all = label_test_all + [class_id for x in range(len(ground_xy_test[class_id]))]
    if class_id in class_h:
        ground_xy_test_h.extend(ground_xy_test[class_id])
        label_test_h = label_test_h + [class_id for x in range(len(ground_xy_test[class_id]))]
    else:
        ground_xy_test_t.extend(ground_xy_test[class_id])
        label_test_t = label_test_t + [class_id for x in range(len(ground_xy_test[class_id]))]

def dataset_build(ground_xy, label):
    ground_xy = np.array(ground_xy)
    label = np.array(label)
    shuffle_array = np.arange(0, len(label), 1)
    np.random.shuffle(shuffle_array)
    label = label[shuffle_array]
    ground_xy = ground_xy[shuffle_array]
    label = torch.from_numpy(label).type(torch.LongTensor)
    ground_xy = torch.from_numpy(ground_xy).type(torch.LongTensor)
    return ground_xy, label

ground_xy_all, label_all = dataset_build(ground_xy_all, label_all)
ground_xy_h, label_h = dataset_build(ground_xy_h, label_h)
ground_xy_t, label_t = dataset_build(ground_xy_t, label_t)
ground_xy_test_all, label_test_all = dataset_build(ground_xy_test_all, label_test_all)
ground_xy_test_h, label_test_h = dataset_build(ground_xy_test_h, label_test_h)
ground_xy_test_t, label_test_t = dataset_build(ground_xy_test_t, label_test_t)

print("数据集构建完成!")
print(f'头部类数量: {h_num_classes}')
print(f'尾部类数量: {t_num_classes}')
print(element_use_h)
print(element_use_t)
print()    

test_dataset = MyData(ms4, pan, label_t2, ground_xy_t2, ms4_patch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)
# test_dataset_all = MyData(ms4, pan, label_test_all, ground_xy_test_all, ms4_patch_size)
# test_loader_all = DataLoader(dataset=test_dataset_all, batch_size=256, shuffle=False, num_workers=0)
# test_dataset_h = MyData(ms4, pan, label_test_h, ground_xy_test_h, ms4_patch_size)
# test_loader_h = DataLoader(dataset=test_dataset_h, batch_size=256, shuffle=False, num_workers=0)
# test_dataset_t = MyData(ms4, pan, label_test_t, ground_xy_test_t, ms4_patch_size)
# test_loader_t = DataLoader(dataset=test_dataset_t, batch_size=256, shuffle=False, num_workers=0)

train_dataset_all = MyData(ms4, pan, label_all, ground_xy_all, ms4_patch_size)
train_loader_all = DataLoader(dataset=train_dataset_all, batch_size=256, shuffle=False, num_workers=0)
train_dataset_h = MyData(ms4, pan, label_h, ground_xy_h, ms4_patch_size)
train_loader_h = DataLoader(dataset=train_dataset_h, batch_size=256, shuffle=False, num_workers=0)
train_dataset_t = MyData(ms4, pan, label_t, ground_xy_t, ms4_patch_size)
train_loader_t = DataLoader(dataset=train_dataset_t, batch_size=256, shuffle=False, num_workers=0)
model_h = ResNet_2(h_num_classes).cuda()
model_t = ResNet_2(t_num_classes).cuda()
optimizer_h = optim.Adam(model_h.parameters(), lr=args.LR[1])
optimizer_t = optim.Adam(model_t.parameters(), lr=args.LR[1])



for epoch in range(1, args.EPOCH[1] + 1):
    avg_loss = train_model(model_h, train_loader_h, optimizer_h, epoch, h_num_classes)

    # average_correct_rate, class_correct_rate = test_model_part(model_h, test_loader_h, h_num_classes, class_mapping, Categories_Number)
    # if average_correct_rate > best_acc_h:
    #     best_acc_h = average_correct_rate
    #     best_epoch_h = epoch
    if epoch == args.EPOCH[1]:
        torch.save(model_h.state_dict(),f'{CURRENT_DIR}/newwork11/model/teacher_h.pth')
        print("teacher_h保存成功")


for epoch in range(1, args.EPOCH[1] + 1):
    if len(train_dataset_t) == 0:
        break
    avg_loss = train_model(model_t, train_loader_t, optimizer_t, epoch, t_num_classes)

    # average_correct_rate, class_correct_rate = test_model_part(model_t, test_loader_t, t_num_classes, class_mapping, Categories_Number)
    # if average_correct_rate > best_acc_t:
    #     best_acc_t = average_correct_rate
    #     best_epoch_t = epoch
    if epoch == args.EPOCH[1]:
        torch.save(model_t.state_dict(),f'{CURRENT_DIR}/newwork11/model/teacher_t.pth')
        print("teacher_t保存成功")



teacher_h = model_h
teacher_t = model_t
teacher_h.eval()
teacher_t.eval()

def get_center(train_loader,num_classes,model):
    center_feature = np.zeros((num_classes, 512, 1, 1))
    num = np.zeros(num_classes)
    correct = np.zeros(num_classes)
    for step,(ms, pan, labels, _) in enumerate(train_loader):
        ms,pan,labels = ms.cuda(),pan.cuda(),labels.cuda()
        output,feature,_ = model(ms,pan)
        feature = feature.cpu().detach().numpy()
        pred_train = output.max(1, keepdim=True)[1].squeeze()
        
        for i in range(len(labels)):
            if labels[i] != pred_train[i]:
                continue
            correct[labels[i]] += 1
            center_feature[labels[i]] += feature[i].reshape(512,1,1)
            num[labels[i]] += 1
    for idx in range(num_classes):
        if num[idx] != 0:
            center_feature[idx] = center_feature[idx] / num[idx]
    return center_feature

center_feature_h = get_center(train_loader_h,h_num_classes,teacher_h)
center_feature_t = get_center(train_loader_t,t_num_classes,teacher_t)
center_feature = np.concatenate([center_feature_h, center_feature_t], axis=0)


stu = ResNet_2(new_num_classes).cuda()
optimizer_stu = optim.Adam(stu.parameters(), lr=args.LR[1])
loss_mse = torch.nn.MSELoss(size_average=True, reduce=True)
loss_fn = nn.CrossEntropyLoss()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


for epoch in range(args.EPOCH[2]):
    stu.train()  # 将模型设置为训练模式
    correct = 0.0
    total_loss = 0.0
    total = 0

    train_bar = tqdm(train_loader_all, desc=f'Epoch {epoch}', leave=False, 
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                     colour='cyan')
    
    for step, (ms, pan, labels, _) in enumerate(train_bar):
        ms, pan, labels = ms.cuda(), pan.cuda(), labels.cuda()  # 将数据迁移到GPU

        # 禁用教师模型的梯度计算，启用学生模型的梯度计算
        requires_grad(teacher_h, False)
        requires_grad(teacher_t, False)
        requires_grad(stu, True)

        

        # 获取模型输出
        outputs, features,fea_3 = stu(ms, pan, labels)
        pred_train = outputs.max(1, keepdim=True)[1]  # 获取预测类别

        cur_center_fea = np.zeros((len(ms),512,1,1))
        for idx in range(len(ms)):
            truth = labels[idx]
            cur_center_fea[idx] = center_feature[truth]
        cur_center_fea = torch.tensor(cur_center_fea).cuda()
        features = features.flatten()
        cur_center_fea = cur_center_fea.flatten()

        fea_target = [0] * len(ms)
        _, _, fea_h = teacher_h(ms, pan)
        fea_h = fea_h.cpu().detach().numpy()
        fea_h = fea_h.tolist()
        _, _, fea_t = teacher_t(ms, pan)
        fea_t = fea_t.cpu().detach().numpy()
        fea_t = fea_t.tolist()

        for k in range(len(labels)):
            if int(labels[k]) in class_h:
                fea_target[k] = fea_h[k]
            elif int(labels[k]) in class_t:
                fea_target[k] = fea_t[k]
        fea_target = torch.Tensor(fea_target).cuda()

        # 计算标准分类损失（交叉熵损失）
        loss = F.cross_entropy(outputs, labels.long())
        loss1 = loss_mse(features.float(), cur_center_fea.float())
        loss2 = loss_mse(fea_3, fea_target)
        loss_all = loss1 + loss + loss2
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


average_correct_rate, class_correct_rate = test_model(stu, test_loader, num_classes, class_mapping, Categories_Number)
print("训练完成！")
exit(0)

