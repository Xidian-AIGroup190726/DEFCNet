import pywt
import torch
import numpy as np

ms = torch.randn(256, 4, 16, 16)
pan = torch.randn(256, 1, 64, 64)

# 定义小波变换处理函数
def wavelet_transform(pan, wavelet='haar', level=2):
    output_list = []
    
    for i in range(pan.shape[0]):
        # 获取当前样本的pan图像（64x64）
        img = pan[i, 0, ...].numpy()
        
        # 执行二级小波分解
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        
        # 提取第二次分解后的系数：cA2, cH2, cV2, cD2
        cA2 = coeffs[0]
        cH2, cV2, cD2 = coeffs[1]
        
        # 堆叠为4个通道（4, 16, 16）
        subbands = np.stack([cA2, cH2, cV2, cD2], axis=0)
        output_list.append(torch.from_numpy(subbands).float())
    
    # 合并所有样本并恢复原始设备
    return torch.stack(output_list, dim=0).to(pan.device)



def swt_transform(ms, wavelet='haar'):
    ms = ms.cpu()
    output_list = []
    
    for i in range(ms.shape[0]):  # 遍历批次
        sample_output = []
        for c in range(ms.shape[1]):  # 遍历通道
            # 获取当前通道的 16x16 图像
            img = ms[i, c, ...].numpy()
            
            # 执行平稳小波变换（单层分解，不降采样）
            coeffs = pywt.swt2(img, wavelet, level=1, start_level=0)
            cA, (cH, cV, cD) = coeffs[0]  # 提取近似、水平、垂直、对角细节
            
            # 堆叠子带 (4, 16, 16)
            subbands = np.stack([cA, cH, cV, cD], axis=0)
            sample_output.append(torch.from_numpy(subbands).float())
        
        # 合并通道维度 (4原通道 * 4子带 = 16通道)
        sample_output = torch.cat(sample_output, dim=0)
        output_list.append(sample_output)
    
    # 合并批次并恢复设备
    return torch.stack(output_list, dim=0).to(ms.device)

# ms应用变换
# ms_transformed = swt_transform(ms)  # 输出形状 (256, 16, 16, 16)

# pan应用小波变换
pan_transformed = wavelet_transform(pan)  # 输出形状 (256, 4, 16, 16)

# print(ms_transformed.shape)
print(pan_transformed.shape)

#拼接ms和pan_transformed
ms_pan = torch.cat((ms, pan_transformed), dim=1)

print(ms_pan.shape)