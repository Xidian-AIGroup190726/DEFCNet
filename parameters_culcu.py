import torch
import torch.nn as nn
from ResNet_2 import ResNet_2

def conv_flops(layer, input_tensor):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = layer.out_channels
    kernel_height, kernel_width = layer.kernel_size
    stride = layer.stride[0]
    padding = layer.padding[0]

    # 计算输出尺寸
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    # 每个卷积操作的 FLOPs = 2 * output_size * kernel_size * in_channels * out_channels
    flops = 2 * out_height * out_width * kernel_height * kernel_width * in_channels * out_channels
    return flops

def downsample_flops(layer, input_tensor):
    # 假设下采样层是通过卷积来实现的
    return conv_flops(layer, input_tensor)

def count_flops(model, input_ms, input_pan):
    total_flops = 0
    
    def forward_hook(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            total_flops += conv_flops(module, input[0])
        elif isinstance(module, nn.Sequential):
            for submodule in module:
                if isinstance(submodule, nn.Conv2d):
                    total_flops += conv_flops(submodule, input[0])
    
    # 注册钩子
    hooks = []
    for layer in model.modules():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    # 计算 MS 和 Pan 图的 FLOPs
    model(input_ms, input_pan)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return total_flops

# 示例输入
input_ms = torch.randn(1, 4, 815, 845)  # MS 输入图像 (batch_size, channels, height, width)
input_pan = torch.randn(1, 1, 3260, 3380)  # Pan 输入图像 (batch_size, channels, height, width)

# 创建模型
model = ResNet_2(num_classes=7)  # 假设模型有7个类别

# 计算 FLOPs
flops = count_flops(model, input_ms, input_pan)
print(f"模型的浮点计算量 (FLOPs): {flops / 1e9} GFLOPs")
