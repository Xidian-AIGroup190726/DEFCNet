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
from scipy.spatial import KDTree
warnings.simplefilter(action='ignore')
np.random.seed(42)  # 设置 NumPy 的随机种子

def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

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

def dataset_build_triple(ground_xy, label, triple_pos_neg):
    ground_xy = np.array(ground_xy)
    label = np.array(label)
    triple_pos_neg = np.array(triple_pos_neg)
    shuffle_array = np.arange(0, len(label), 1)
    np.random.shuffle(shuffle_array)
    label = label[shuffle_array]
    ground_xy = ground_xy[shuffle_array]
    triple_pos_neg = triple_pos_neg[shuffle_array]
    label = torch.from_numpy(label).type(torch.LongTensor)
    ground_xy = torch.from_numpy(ground_xy).type(torch.LongTensor)
    triple_pos_neg = torch.from_numpy(triple_pos_neg).type(torch.LongTensor)
    return ground_xy, label, triple_pos_neg

def get_center(train_loader,num_classes,model):
    center_feature = np.zeros((num_classes, 256, 1, 1))
    num = np.zeros(num_classes)
    correct = np.zeros(num_classes)
    for step,(data, labels, _) in enumerate(train_loader):
        data,labels = data.cuda(),labels.cuda()
        output,feature,_ = model(data)
        feature = feature.cpu().detach().numpy()
        pred_train = output.max(1, keepdim=True)[1].squeeze()
        
        for i in range(len(labels)):
            if labels[i] != pred_train[i]:
                continue
            correct[labels[i]] += 1
            center_feature[labels[i]] += feature[i].reshape(256,1,1)
            num[labels[i]] += 1
    for idx in range(num_classes):
        if num[idx] != 0:
            center_feature[idx] = center_feature[idx] / num[idx]
    return center_feature


def single_kmeans(split_class,num_classes,data_2d,all_location,all_labels):
    class_mapping = {}
    split = {}
    split_points = {}
    count_num = 0
    for i in range(num_classes):
        class_mapping[i] = i


    for class_id in range(num_classes): 
        if class_id not in split_class:
            class_points = data_2d[all_labels == class_id]
            class_location = all_location[all_labels == class_id]
            class_location = class_location.squeeze(axis=1)
            new_bucket = np.array([[]]).tolist()
            new_bucket_points = np.array([[]]).tolist()
            for i in range(len(class_points)):
                x,y = class_location[i]
                a,b,c = class_points[i]
                new_bucket[0].append([x,y])
                new_bucket_points[0].append([a,b,c])

            count_num+= 1
            print(f'类{class_id}不拆分:{[len(x) for x in new_bucket]}')    
                    
            #保存分割结果
            split[class_id] = new_bucket
            split_points[class_id] = new_bucket_points
            continue
            
        class_points = data_2d[all_labels == class_id]
        class_location = all_location[all_labels == class_id]
        class_location = class_location.squeeze(axis=1)

        k_range = range(2, 11)  # 轮廓系数计算需要 k >= 2
        
        # 对每个 k 值进行聚类
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels_kmeans = kmeans.fit_predict(class_points)
            score = silhouette_score(class_points, labels_kmeans) # 计算轮廓系数
            silhouette_scores.append(score)
        best_k = k_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels_kmeans = kmeans.fit_predict(class_points)

        kmeans_ms = {}
        for i in range(best_k):
            kmeans_ms[i] = class_points[labels_kmeans == i]


        
        print(f"class_{class_id}:ms:{best_k}")

        new_bucket = np.array([[]] * best_k).tolist()
        new_bucket_points = np.array([[]] * best_k).tolist()
        for i in range(len(labels_kmeans)):
            idx = labels_kmeans[i]
            x,y = class_location[i]
            a,b,c = class_points[i]
            new_bucket[idx].append([x,y])
            new_bucket_points[idx].append([a,b,c])

        kmeans_center = []
        
            
        for i in range(best_k):
            center_ms = kmeans_ms[i].mean(axis=0)
            kmeans_center.append(center_ms.tolist())

        split[class_id] = new_bucket
        split_points[class_id] = new_bucket_points


        print(f'类{class_id}被分割为{best_k}个新类!')
        print(f'新类的数量分别为{[len(x) for x in new_bucket]}')

    # 处理新的类别
    new_num_classes = 0
    ground_xy_train = np.array([[]] * num_classes * 50).tolist()
    ground_xy_points = np.array([[]] * num_classes * 50).tolist()   
    for class_id in range(num_classes):
        for idx in range(len(split[class_id])):
            ground_xy_train[new_num_classes] = np.array(split[class_id][idx])
            ground_xy_points[new_num_classes] = np.array(split_points[class_id][idx])
            class_mapping[new_num_classes] = class_id
            new_num_classes += 1
    ground_xy_train = ground_xy_train[:new_num_classes]
    ground_xy_points = ground_xy_points[:new_num_classes]


    print(f'总共分割出{new_num_classes}个新类!')
    print([len(x) for x in ground_xy_train])

    return ground_xy_train,class_mapping,new_num_classes,ground_xy_points


def rebuild_dataset_and_model(ground_xy_train,class_mapping,new_num_classes,threshold,ms4,pan,\
                              ground_xy_t2,label_t2,ms4_patch_size, choose, \
                              ground_xy_points,data_2d,all_labels,all_location):
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


    ground_xy_all, label_all = dataset_build(ground_xy_all, label_all)
    ground_xy_h, label_h = dataset_build(ground_xy_h, label_h)
    ground_xy_t, label_t = dataset_build(ground_xy_t, label_t)

    print("数据集构建完成!")
    print(f'头部类数量: {h_num_classes}')
    print(f'尾部类数量: {t_num_classes}')
    print(element_use_h)
    print(element_use_t)
    print()    

    test_dataset_all = MyData_single(ms4, pan, label_t2, ground_xy_t2, ms4_patch_size, choose)
    test_loader_all = DataLoader(dataset=test_dataset_all, batch_size=256, shuffle=False, num_workers=0)

    train_dataset_all = MyData_single(ms4, pan, label_all, ground_xy_all, ms4_patch_size, choose)
    train_loader_all = DataLoader(dataset=train_dataset_all, batch_size=256, shuffle=False, num_workers=0)
    train_dataset_h = MyData_single(ms4, pan, label_h, ground_xy_h, ms4_patch_size, choose)
    train_loader_h = DataLoader(dataset=train_dataset_h, batch_size=256, shuffle=False, num_workers=0)
    train_dataset_t = MyData_single(ms4, pan, label_t, ground_xy_t, ms4_patch_size, choose)
    train_loader_t = DataLoader(dataset=train_dataset_t, batch_size=256, shuffle=False, num_workers=0)
    


    return train_loader_all,train_loader_h,train_loader_t,test_loader_all,\
            h_num_classes,t_num_classes,new_class_mapping,class_h,class_t


def single_train(train_loader_all,train_loader_h,train_loader_t,\
    test_loader_all,new_num_classes,num_classes,h_num_classes,\
    t_num_classes,class_mapping,class_h,class_t,lr1,epoch1,epoch2,choose):
    if choose == 'ms4':
        # model_h = ResNet_1(h_num_classes,4,'ms4').cuda()
        # model_t = ResNet_1(t_num_classes,4,'ms4').cuda()
        # stu = ResNet_1(new_num_classes,4,'ms4').cuda()
        model_h = ResNet_1_ms(h_num_classes).cuda()
        model_t = ResNet_1_ms(t_num_classes).cuda()
        stu = ResNet_1_ms(new_num_classes).cuda()
    elif choose == 'pan':
        # model_h = ResNet_1(h_num_classes,1,'pan').cuda()
        # model_t = ResNet_1(t_num_classes,1,'pan').cuda()
        # stu = ResNet_1(new_num_classes,1,'pan').cuda()
        model_h = ResNet_1_pan(h_num_classes).cuda()
        model_t = ResNet_1_pan(t_num_classes).cuda()
        stu = ResNet_1_pan(new_num_classes).cuda()
    optimizer_h = optim.Adam(model_h.parameters(), lr=lr1)
    optimizer_t = optim.Adam(model_t.parameters(), lr=lr1)

    for epoch in range(1, epoch1 + 1):
        avg_loss = train_model_single(model_h, train_loader_h, optimizer_h, epoch, h_num_classes)

        # average_correct_rate, class_correct_rate = test_model_part(model_h, test_loader_h, h_num_classes, class_mapping, Categories_Number)
        # if average_correct_rate > best_acc_h:
        #     best_acc_h = average_correct_rate
        #     best_epoch_h = epoch
        if epoch == epoch1:
            print("teacher_h训练成功")


    for epoch in range(1, epoch1 + 1):
        if len(train_loader_t) == 0:
            break
        avg_loss = train_model_single(model_t, train_loader_t, optimizer_t, epoch, t_num_classes)

        if epoch == epoch1:
            print("teacher_t训练成功")


    teacher_h = model_h
    teacher_t = model_t
    teacher_h.eval()
    teacher_t.eval()



    center_feature_h = get_center(train_loader_h,h_num_classes,teacher_h)
    center_feature_t = get_center(train_loader_t,t_num_classes,teacher_t)
    center_feature = np.concatenate([center_feature_h, center_feature_t], axis=0)


    
    optimizer_stu = optim.Adam(stu.parameters(), lr=lr1)
    loss_mse1 = torch.nn.MSELoss(size_average=True, reduce=True)
    loss_mse2 = torch.nn.MSELoss(size_average=True, reduce=True)

    
    for epoch in range(epoch2):
        stu.train()  # 将模型设置为训练模式
        correct = 0.0
        total_loss = 0.0
        total = 0

        train_bar = tqdm(train_loader_all, desc=f'Epoch {epoch}', leave=False, 
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                        colour='cyan')
        
        for step, (data, labels, _) in enumerate(train_bar):
            data, labels = data.cuda(), labels.cuda()# 将数据迁移到GPU

            # 禁用教师模型的梯度计算，启用学生模型的梯度计算
            requires_grad(teacher_h, False)
            requires_grad(teacher_t, False)
            requires_grad(stu, True)

            

            # 获取模型输出
            outputs, features, fea_3 = stu(data, labels)
            pred_train = outputs.max(1, keepdim=True)[1]  # 获取预测类别


            cur_center_fea = np.zeros((len(data),256,1,1))
            for idx in range(len(data)):
                truth = labels[idx]
                cur_center_fea[idx] = center_feature[truth]
            cur_center_fea = torch.tensor(cur_center_fea).cuda()
            features = features.flatten()
            cur_center_fea = cur_center_fea.flatten()

            fea_target = [0] * len(data)
            _, _, fea_h = teacher_h(data)
            fea_h = fea_h.cpu().detach().numpy()
            fea_h = fea_h.tolist()
            _, _, fea_t = teacher_t(data)
            fea_t = fea_t.cpu().detach().numpy()
            fea_t = fea_t.tolist()
            # print(len(fea_h))
            # print(len(fea_h[0]))
            for k in range(len(labels)):
                if int(labels[k]) in class_h:
                    fea_target[k] = fea_h[k]
                elif int(labels[k]) in class_t:
                    fea_target[k] = fea_t[k]
            fea_target = torch.Tensor(fea_target).cuda()

            # 计算标准分类损失（交叉熵损失）
            loss = F.cross_entropy(outputs, labels.long())
            loss1 = loss_mse1(features.float(), cur_center_fea.float())
            loss2 = loss_mse2(fea_3, fea_target)
            # loss3 = 
            # loss_all = loss + 1* (loss1 + loss2) + 1 * loss_triplet
            loss_all = loss + loss2
            optimizer_stu.zero_grad()  # 清空梯度
            loss_all.backward()
            optimizer_stu.step()

            # 更新训练准确率
            correct += pred_train.eq(labels.view_as(pred_train)).sum().item()  # 计算正确的预测数量
            total_loss += loss.item() * data.size(0)  # 计算当前batch的总损失
            
            # 更新进度条
            train_bar.set_description(f"Epoch[{epoch}]")
            train_bar.set_postfix(train_loss=total_loss / (step + 1), train_acc=correct * 100.0 / (total + data.size(0)))
            
            total += data.size(0)  # 累计处理的样本数量

        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader_all.dataset)
        avg_acc = correct * 100.0 / len(train_loader_all.dataset)
        
        print(f"Epoch[{epoch}] Train Accuracy: {avg_acc:.3f}, Avg Loss: {avg_loss:.4f}")


    average_correct_rate, class_correct_rate = test_model_single(stu, test_loader_all, new_num_classes, class_mapping, num_classes)
    return stu