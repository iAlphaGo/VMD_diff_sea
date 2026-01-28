"""
修复VMD集成模块 - 主要修复输入通道问题
"""

import torch
import torch.nn as nn
from utils.vmd_op import batch_vmd_process

class VMDEnhancedDenoiser(nn.Module):
    """VMD增强的去噪器 - 修复版"""
    def __init__(self, original_denoiser, vmd_mode='condition', num_modes=4):
        """
        Args:
            original_denoiser: 原始的Denoise_UNet
            vmd_mode: VMD集成模式
                - 'preprocess': 仅作为预处理（输入通道不变）
                - 'condition': 作为条件输入（扩展输入通道）
            num_modes: VMD模态数量
        """
        super(VMDEnhancedDenoiser, self).__init__()
        self.original_denoiser = original_denoiser
        self.vmd_mode = vmd_mode
        self.num_modes = num_modes
        
        # 获取原始输入输出通道
        original_in_channels = original_denoiser.image_proj.in_channels
        print(f"原始Denoise_UNet输入通道数: {original_in_channels}")
        
        if vmd_mode == 'condition':
            # 计算VMD增加的通道数
            vmd_channels = num_modes * 3  # 每个模态3个通道
            new_in_channels = original_in_channels + vmd_channels
            print(f"VMD条件模式 - 新输入通道数: {new_in_channels}")
            
            # 替换第一层卷积以接受更多输入通道
            self._replace_first_conv(new_in_channels)
    
    def _replace_first_conv(self, new_in_channels):
        """替换第一层卷积以适应更多输入通道"""
        original_conv = self.original_denoiser.image_proj
        
        # 获取原始权重和偏置
        old_weight = original_conv.weight.data
        old_bias = original_conv.bias.data
        out_channels = original_conv.out_channels
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ).to(old_weight.device)
        
        # 初始化新权重
        with torch.no_grad():
            # 复制原始权重到前original_in_channels个输入通道
            new_conv.weight[:, :old_weight.shape[1], :, :].copy_(old_weight)
            # 剩余通道初始化为0
            if new_in_channels > old_weight.shape[1]:
                new_conv.weight[:, old_weight.shape[1]:, :, :].zero_()
            new_conv.bias.copy_(old_bias)
        
        # 替换原来的卷积层
        self.original_denoiser.image_proj = new_conv
        
        print(f"已替换第一层卷积: {old_weight.shape} -> {new_conv.weight.shape}")
    
    def forward(self, x, t=None):
        """前向传播"""
        return self.original_denoiser(x, t)

class VMDEnhancedDocDiff(nn.Module):
    """VMD增强的DocDiff模型 - 修复版"""
    def __init__(self, original_docdiff, vmd_mode='condition', num_modes=4):
        super(VMDEnhancedDocDiff, self).__init__()
        self.original_docdiff = original_docdiff
        self.vmd_mode = vmd_mode
        self.num_modes = num_modes
        
        # 保存原始组件
        self.beta_predictor = original_docdiff.beta_predictor
        
        # 创建VMD增强的去噪器
        self.denoiser = VMDEnhancedDenoiser(
            original_docdiff.denoiser,
            vmd_mode=vmd_mode,
            num_modes=num_modes
        )
        
        print(f"VMDEnhancedDocDiff初始化完成")
        print(f"  - VMD模式: {vmd_mode}")
        print(f"  - 模态数: {num_modes}")
    
    def forward(self, x, condition, hist, depth, t, diffusion, vmd_modes=None):
        """
        Args:
            x: 真实图像 [B, 3, H, W]
            condition: 条件图像 [B, 3, H, W]
            hist: 直方图 [B, 3, H, W]
            depth: 深度图 [B, 3, H, W]
            t: 时间步 [B]
            diffusion: 扩散模型
            vmd_modes: VMD模态 [B, num_modes*3, H, W]
        """
        # 打印调试信息
        print(f"\n=== VMDEnhancedDocDiff前向传播 ===")
        print(f"输入形状:")
        print(f"  x: {x.shape}")
        print(f"  condition: {condition.shape}")
        print(f"  hist: {hist.shape}")
        print(f"  depth: {depth.shape}")
        print(f"  vmd_modes: {vmd_modes.shape if vmd_modes is not None else 'None'}")
        
        # 原始计算流程
        pred_beta = self.beta_predictor(condition)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        T_direct = torch.clamp((torch.exp(-pred_beta * depth_norm)), 0, 1)
        T_scatter = torch.clamp((1 - torch.exp(-pred_beta * depth_norm)), 0, 1)
        
        # 计算大气光
        from utils.utils import get_A
        atm_light = [get_A(item) for item in condition]
        atm_light = torch.stack(atm_light).to(x.device)
        
        # 计算场景辐射
        J = torch.clamp(((condition - T_scatter * atm_light) / (T_direct + 1e-8)), 0, 1)
        
        # 添加噪声
        noisy_image, noise_ref = diffusion.noisy_image(t, x)
        
        # 构建去噪器输入
        if self.vmd_mode == 'preprocess':
            # 预处理模式：只使用原始输入
            denoiser_input = torch.cat((noisy_image, condition, J, hist), dim=1)
            print(f"预处理模式 - denoiser输入形状: {denoiser_input.shape}")
        
        elif self.vmd_mode == 'condition' and vmd_modes is not None:
            # 条件输入模式：加入VMD模态
            # 原始输入: noisy_image(3) + condition(3) + J(3) + hist(3) = 12通道
            # VMD模态: num_modes * 3 = 12通道 (当num_modes=4时)
            base_input = torch.cat((noisy_image, condition, J, hist), dim=1)
            denoiser_input = torch.cat((base_input, vmd_modes), dim=1)
            print(f"条件输入模式:")
            print(f"  基础输入: {base_input.shape}")
            print(f"  VMD模态: {vmd_modes.shape}")
            print(f"  合并后: {denoiser_input.shape}")
        
        else:
            # 默认：只使用原始输入
            denoiser_input = torch.cat((noisy_image, condition, J, hist), dim=1)
            print(f"默认模式 - denoiser输入形状: {denoiser_input.shape}")
        
        # 去噪
        denoised_J = self.denoiser(denoiser_input, t)
        
        print(f"去噪输出形状: {denoised_J.shape}")
        print("=== 前向传播完成 ===\n")
        
        return J, noise_ref, denoised_J, T_direct, T_scatter