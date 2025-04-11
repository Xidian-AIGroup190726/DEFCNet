# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
# from libtiff import TIFF
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import torch.optim as optim
from tqdm import tqdm

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

class MyData_single(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size,choose):

        
        self.train_data1 = MS4
        self.cut_ms_size = cut_size
    
        self.train_data2 = Pan
        self.cut_pan_size = cut_size * 4
        
        
        self.train_labels = Label
        self.gt_xy = xy
        self.choose = choose
        
        

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
        if self.choose == 'ms4':
            return ms, target, locate_xy
        elif self.choose == 'pan':
            return pan,target,locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData_triple(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size,choose, triple_pos_neg):

        
        self.train_data1 = MS4
        self.cut_ms_size = cut_size
    
        self.train_data2 = Pan
        self.cut_pan_size = cut_size * 4
        
        
        self.train_labels = Label
        self.gt_xy = xy
        self.choose = choose
        self.triple_pos_neg = triple_pos_neg
        
        

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pos_ms,y_pos_ms = self.triple_pos_neg[index][0]
        x_neg_ms,y_neg_ms = self.triple_pos_neg[index][1]
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        x_pos_pan = int(4 * x_pos_ms)
        y_pos_pan = int(4 * y_pos_ms)
        x_neg_pan = int(4 * x_neg_ms)
        y_neg_pan = int(4 * y_neg_ms)
        
        ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                y_ms:y_ms + self.cut_ms_size]
        pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        pos_ms = self.train_data1[:, x_pos_ms:x_pos_ms + self.cut_ms_size,
                y_pos_ms:y_pos_ms + self.cut_ms_size]
        pos_pan = self.train_data2[:, x_pos_pan:x_pos_pan + self.cut_pan_size,
                    y_pos_pan:y_pos_pan + self.cut_pan_size]
        neg_ms = self.train_data1[:, x_neg_ms:x_neg_ms + self.cut_ms_size,
                y_neg_ms:y_neg_ms + self.cut_ms_size]
        neg_pan = self.train_data2[:, x_neg_pan:x_neg_pan + self.cut_pan_size,
                    y_neg_pan:y_neg_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]
        target = self.train_labels[index]

        if self.choose == 'ms4':
            return ms, target, locate_xy, pos_ms, neg_ms
        elif self.choose == 'pan':
            return pan,target,locate_xy, pos_pan, neg_pan

    def __len__(self):
        return len(self.gt_xy)
    
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

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算欧氏距离
        distance_ap = torch.norm(anchor - positive, p=2, dim=1)  # 锚点和正样本的距离
        distance_an = torch.norm(anchor - negative, p=2, dim=1)  # 锚点和负样本的距离
        # 计算损失
        losses = torch.relu(distance_ap - distance_an + self.margin)
        return losses.mean()


def train_model(model, train_loader, optimizer, epoch, num_classes):
    model.train()
    correct = 0.0
    total_loss = 0.0
    total = 0
    # class_weights = torch.ones(num_classes).cuda()  # 默认每个类别的权重为1
    class_weights = [1.0] * 11
    class_weights = torch.ones(num_classes).cuda()  # 默认每个类别的权重为1

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False, 
                                                  bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                                                  colour='cyan')
    for step, (ms, pan, label, _) in enumerate(train_bar):  # 选择颜色
        ms, pan, label = ms.cuda(), pan.cuda(), label.cuda()

        optimizer.zero_grad()
        output,_,_ = model(ms, pan, label)
        total += output.size(0)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()

        

        loss = F.cross_entropy(output, label.long(), weight=class_weights)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_bar.set_description(f"Epoch[{epoch}]")
        train_bar.set_postfix(train_loss=loss.item(),train_acc=correct * 100.0 /total)

    avg_loss = total_loss / len(train_loader)
    print(f"Train Accuracy: {correct * 100.0 / len(train_loader.dataset):.3f}, Avg Loss: {avg_loss:.4f}")
    return avg_loss

def train_model_single(model, train_loader, optimizer, epoch, num_classes):
    model.train()
    correct = 0.0
    total_loss = 0.0
    total = 0
    # class_weights = torch.ones(num_classes).cuda()  # 默认每个类别的权重为1
    class_weights = [1.0] * 11
    class_weights = torch.ones(num_classes).cuda()  # 默认每个类别的权重为1

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False, 
                                                  bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                                                  colour='cyan')
    for step, (data, label, _) in enumerate(train_bar):  # 选择颜色
        data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        output,_,_ = model(data, label)
        total += output.size(0)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()

        

        loss = F.cross_entropy(output, label.long(), weight=class_weights)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_bar.set_description(f"Epoch[{epoch}]")
        train_bar.set_postfix(train_loss=loss.item(),train_acc=correct * 100.0 /total)

    avg_loss = total_loss / len(train_loader)
    print(f"Train Accuracy: {correct * 100.0 / len(train_loader.dataset):.3f}, Avg Loss: {avg_loss:.4f}")
    return avg_loss

def test_model(model, test_loader, num_classes,class_mapping, Categories_Number):
    model.eval()
    correct = 0.0
    total_samples = 0.0
    test_loss = 0.0
    total_1 = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes 
    test_matrix = np.zeros([Categories_Number, Categories_Number])
    my_target = 1
    target_predictions = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Test', leave=False,
                                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                                        colour='cyan')
        for ms, pan, target, _ in test_bar:
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()

            output,_,_ = model(ms, pan)
            total_1 += output.size(0)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            
            #映射
            pred = torch.tensor([class_mapping.get(p.item(), p.item()) for p in pred]).cuda()

            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            total_samples += target.size(0)  # 直接使用target.size(0)计算总样本数
            test_bar.set_postfix(test_loss=test_loss, test_acc=100.0 * correct / total_1)

            for i in range(num_classes):
                class_mask = (target == i)
                class_correct[i] += pred[class_mask].eq(i).sum().item()
                class_total[i] += class_mask.sum().item()

        test_loss /= len(test_loader.dataset)
        class_correct_rate = np.array(class_correct) / np.array(class_total)
        average_correct_rate = correct * 100.0 / total_samples

    b=np.sum(test_matrix,axis=0)
    accuracy=[]
    c = 0
    for i in range(0,Categories_Number):
        a=test_matrix[i][i]/b[i]
        accuracy.append(a)
        c+=test_matrix[i][i]
    
    print('OA: {0:f}'.format(c/np.sum(b,axis=0)))
    print('AA: {0:f}'.format(np.mean(accuracy)))
    print('KAPPA: {0:f}'.format(kappa(test_matrix)))
    print(f"Test Accuracy: {average_correct_rate:.3f}, Test Loss: {test_loss:.4f}")
    print("Class Correct:", class_correct)
    print("Class Total:", class_total)
    print("Class Correct Rate:", class_correct_rate)

    # print(f"\n--- Predicted Information for Target Class {my_target} ---")
    # for info in target_predictions:
    #     print(f"Original Target: {info['original_target']}, "
    #           f"Predicted Before Mapping: {info['predicted_before_mapping']}, "
    #           f"Predicted After Mapping: {info['predicted_after_mapping']}")

    return average_correct_rate, class_correct_rate

def test_model_single(model, test_loader, num_classes,class_mapping, Categories_Number):
    model.eval()
    correct = 0.0
    total_samples = 0.0
    test_loss = 0.0
    total_1 = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes 
    test_matrix = np.zeros([Categories_Number, Categories_Number])
    my_target = 1
    target_predictions = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Test', leave=False,
                                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                                        colour='cyan')
        for data, target, _ in test_bar:
            data, target = data.cuda(), target.cuda()

            output,_,_ = model(data)
            total_1 += output.size(0)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            
            #映射
            pred = torch.tensor([class_mapping.get(p.item(), p.item()) for p in pred]).cuda()

            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            total_samples += target.size(0)  # 直接使用target.size(0)计算总样本数
            test_bar.set_postfix(test_loss=test_loss, test_acc=100.0 * correct / total_1)

            for i in range(num_classes):
                class_mask = (target == i)
                class_correct[i] += pred[class_mask].eq(i).sum().item()
                class_total[i] += class_mask.sum().item()

        test_loss /= len(test_loader.dataset)
        class_correct_rate = np.array(class_correct) / np.array(class_total)
        average_correct_rate = correct * 100.0 / total_samples

    b=np.sum(test_matrix,axis=0)
    accuracy=[]
    c = 0
    for i in range(0,Categories_Number):
        a=test_matrix[i][i]/b[i]
        accuracy.append(a)
        c+=test_matrix[i][i]
    
    print('OA: {0:f}'.format(c/np.sum(b,axis=0)))
    print('AA: {0:f}'.format(np.mean(accuracy)))
    print('KAPPA: {0:f}'.format(kappa(test_matrix)))
    print(f"Test Accuracy: {average_correct_rate:.3f}, Test Loss: {test_loss:.4f}")
    print("Class Correct:", class_correct)
    print("Class Total:", class_total)
    print("Class Correct Rate:", class_correct_rate)

    # print(f"\n--- Predicted Information for Target Class {my_target} ---")
    # for info in target_predictions:
    #     print(f"Original Target: {info['original_target']}, "
    #           f"Predicted Before Mapping: {info['predicted_before_mapping']}, "
    #           f"Predicted After Mapping: {info['predicted_after_mapping']}")

    return average_correct_rate, class_correct_rate



def OA_AA_Kappa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)