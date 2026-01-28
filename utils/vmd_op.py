"""
修复后的VMD处理模块
"""

import torch
import torch.nn as nn
import numpy as np

class MultiModeVMD(nn.Module):
    """支持多模态的变分模态分解实现"""
    def __init__(self, num_modes=4, device='cuda'):
        super(MultiModeVMD, self).__init__()
        self.num_modes = num_modes
        self.device = device
        self.kernels_cache = {}
        self.requires_grad_(False)
        
    def _create_gaussian_kernel(self, size, sigma, channels=3):
        """创建高斯核 - 修复通道问题"""
        key = (size, sigma, channels)
        if key not in self.kernels_cache:
            # 创建2D高斯核
            x = torch.arange(size, dtype=torch.float32, device=self.device)
            x = x - size // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            
            # 创建2D核 [1, 1, size, size]
            kernel_2d = gauss.view(1, 1, size, 1) * gauss.view(1, 1, 1, size)
            
            # 扩展到3个通道 [3, 1, size, size]
            # 这样每个输入通道都使用相同的滤波器
            kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
            
            self.kernels_cache[key] = kernel_2d
        return self.kernels_cache[key]
    
    def forward(self, x):
        """执行VMD分解 - 支持多模态"""
        B, C, H, W = x.shape
        
        # 根据模态数量动态生成高斯核参数
        if self.num_modes == 4:
            kernel_params = [
                (15, 2.0, 7),  # (kernel_size, sigma, padding)
                (9, 1.0, 4),
                (5, 0.5, 2)
            ]
        elif self.num_modes == 5:
            kernel_params = [
                (17, 2.5, 8),
                (13, 1.8, 6),
                (9, 1.2, 4),
                (5, 0.6, 2)
            ]
        else:
            raise ValueError(f"不支持的模态数量: {self.num_modes}")
        
        # 获取不同尺度的高斯核
        kernels = []
        for size, sigma, padding in kernel_params:
            kernels.append((
                self._create_gaussian_kernel(size, sigma, channels=C),
                padding
            ))
        
        # 应用高斯滤波分离不同频率成分
        modes = []
        prev_result = None
        
        for i, (kernel, padding) in enumerate(kernels):
            # 使用分组卷积：每个通道独立处理
            # kernel形状: [C, 1, size, size]
            # groups=C 表示每个通道使用独立的滤波器
            current_result = torch.nn.functional.conv2d(
                x, kernel, padding=padding, groups=C
            )
            
            if prev_result is None:
                # 第一个模态是最低频
                modes.append(current_result)
            else:
                # 中间模态是相邻滤波结果的差
                modes.append(current_result - prev_result)
            
            prev_result = current_result
        
        # 最后一个模态是残差（最高频）
        if prev_result is not None:
            modes.append(x - prev_result)
        
        # 确保模态数量正确
        assert len(modes) == self.num_modes, f"模态数量错误: 期望{self.num_modes}, 实际{len(modes)}"
        
        # 将所有模态拼接在通道维度上
        # 结果形状: [B, num_modes*C, H, W]
        return torch.cat(modes, dim=1)

def batch_vmd_process(images, num_modes=4):
    """
    批量处理图像的VMD分解
    Args:
        images: [B, C, H, W] 张量
        num_modes: VMD模态数量
    Returns:
        vmd_modes: [B, num_modes*C, H, W] 张量
    """
    # 获取设备
    device = images.device
    
    # 创建VMD模型
    vmd_model = MultiModeVMD(num_modes=num_modes, device=device)
    
    # 设置为评估模式
    vmd_model.eval()
    
    # 应用VMD
    with torch.no_grad():
        vmd_modes = vmd_model(images)
    
    return vmd_modes

# 测试函数
def test_vmd():
    """测试VMD功能"""
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_image = torch.randn(batch_size, channels, height, width).cuda()
    
    # 应用VMD
    vmd_modes = batch_vmd_process(test_image, num_modes=4)
    
    print(f"输入形状: {test_image.shape}")
    print(f"VMD输出形状: {vmd_modes.shape}")
    print(f"模态数量: 4")
    print(f"每个模态通道数: {channels}")
    print(f"总通道数: {4 * channels} = {vmd_modes.shape[1]}")
    
    return vmd_modes

if __name__ == "__main__":
    test_vmd()
