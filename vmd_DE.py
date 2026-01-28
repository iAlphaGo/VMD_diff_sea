import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
from typing import List, Tuple, Optional, Dict
import math
import os
import glob
from tqdm import tqdm
import shutil
import itertools
import random
from bayes_opt import BayesianOptimization
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

class MultiModeVMD(nn.Module):
    """支持多模态的变分模态分解实现"""
    def __init__(self, num_modes=5):
        super(MultiModeVMD, self).__init__()
        self.kernels_cache = {}
        self.num_modes = num_modes
        self.requires_grad_(False)

    def _create_gaussian_kernel(self, size, sigma, device):
        """创建高斯核"""
        key = (size, sigma, str(device))
        if key not in self.kernels_cache:
            x = torch.arange(size, dtype=torch.float32, device=device)
            x = x - size // 2
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            kernel = gauss.view(1, 1, size, 1) * gauss.view(1, 1, 1, size)
            self.kernels_cache[key] = kernel
        return self.kernels_cache[key]

    def forward(self, x):
        """执行VMD分解 - 支持多模态"""
        B, C, H, W = x.shape
        device = x.device
        
        # 根据模态数量动态生成高斯核参数
        if self.num_modes == 4:
            # 4模态分解
            kernel_params = [
                (15, 2.0, 7),  # (kernel_size, sigma, padding)
                (9, 1.0, 4),
                (5, 0.5, 2)
            ]
        elif self.num_modes == 5:
            # 5模态分解
            kernel_params = [
                (17, 2.5, 8),  # 更低频
                (13, 1.8, 6),
                (9, 1.2, 4),
                (5, 0.6, 2)
            ]
        elif self.num_modes == 6:
            # 6模态分解
            kernel_params = [
                (19, 3.0, 9),  # 更低频
                (15, 2.2, 7),
                (11, 1.5, 5),
                (7, 1.0, 3),
                (3, 0.4, 1)
            ]
        else:
            raise ValueError(f"不支持的模态数量: {self.num_modes}")
        
        # 获取不同尺度的高斯核
        kernels = []
        for size, sigma, padding in kernel_params:
            kernels.append((
                self._create_gaussian_kernel(size, sigma, device),
                padding
            ))
        
        # 批量处理所有通道
        x_flat = x.view(B*C, 1, H, W)
        
        # 应用高斯滤波分离不同频率成分
        modes = []
        prev_result = None
        
        for i, (kernel, padding) in enumerate(kernels):
            current_result = torch.nn.functional.conv2d(x_flat, kernel, padding=padding).view(B, C, H, W)
            
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
        
        return torch.cat(modes, dim=0)

class MultiModeConservativeVMD(nn.Module):
    """多模态保守策略的VMD增强实现"""
    def __init__(self, num_modes=5, color_strength=0.5, detail_enhance=1.2, noise_suppress=0.5):
        super(MultiModeConservativeVMD, self).__init__()
        self.requires_grad_(False)
        self.num_modes = num_modes
        self.color_strength = color_strength
        self.detail_enhance = detail_enhance
        self.noise_suppress = noise_suppress
        
    def adaptive_color_correction(self, low_freq_mode: torch.Tensor) -> torch.Tensor:
        """自适应颜色校正 - 适用于最低频模态"""
        B, C, H, W = low_freq_mode.shape
        corrected = low_freq_mode.clone()
        
        for b in range(B):
            # 计算各通道均值
            mean_r = low_freq_mode[b, 0].mean()
            mean_g = low_freq_mode[b, 1].mean() 
            mean_b = low_freq_mode[b, 2].mean()
            
            # 增强红色通道，抑制蓝色通道
            red_boost = 1.0 + self.color_strength * (1.0 - mean_r)
            blue_reduce = 1.0 - self.color_strength * (mean_b - 0.5)
            
            # 应用颜色校正
            corrected[b, 0] = low_freq_mode[b, 0] * red_boost
            corrected[b, 1] = low_freq_mode[b, 1] * 1.0
            corrected[b, 2] = low_freq_mode[b, 2] * blue_reduce
            
            # 限制在合理范围内
            corrected[b] = torch.clamp(corrected[b], 0, 1)
            
        return corrected
    
    def adaptive_detail_enhancement(self, mid_freq_mode: torch.Tensor, enhancement_factor=1.0) -> torch.Tensor:
        """自适应细节增强 - 适用于中频模态"""
        B, C, H, W = mid_freq_mode.shape
        enhanced = mid_freq_mode.clone()
        
        for b in range(B):
            for c in range(C):
                mode_slice = mid_freq_mode[b, c]
                local_var = torch.var(mode_slice)
                # 根据局部方差调整增强强度
                adaptive_strength = self.detail_enhance * (1.0 + local_var) * enhancement_factor
                enhanced[b, c] = mode_slice * adaptive_strength
                
        return enhanced
    
    def noise_suppression_with_threshold(self, high_freq_mode: torch.Tensor, suppression_factor=1.0) -> torch.Tensor:
        """带阈值的噪声抑制 - 适用于高频模态"""
        B, C, H, W = high_freq_mode.shape
        suppressed = high_freq_mode.clone()
        
        for b in range(B):
            for c in range(C):
                mode_slice = high_freq_mode[b, c]
                std_val = torch.std(mode_slice)
                threshold = std_val * 1.5
                
                mask = torch.abs(mode_slice) > threshold
                suppressed[b, c] = torch.where(
                    mask, 
                    mode_slice * self.noise_suppress * suppression_factor,
                    mode_slice * 0.1 * suppression_factor
                )
                
        return suppressed
    
    def forward(self, x):
        """多模态保守策略增强重构"""
        total_modes, C, H, W = x.shape
        B = total_modes // self.num_modes
            
        # 拆分模态
        modes = x.view(self.num_modes, B, C, H, W)
        
        # 根据模态数量应用不同的增强策略
        processed_modes = []
        
        for i in range(self.num_modes):
            if i == 0:
                # 第一个模态（最低频）- 颜色校正
                processed_mode = self.adaptive_color_correction(modes[i])
            elif i == self.num_modes - 1:
                # 最后一个模态（最高频）- 噪声抑制
                processed_mode = self.noise_suppression_with_threshold(modes[i])
            else:
                # 中间模态 - 细节增强，根据模态位置调整增强强度
                # 越靠近高频，增强强度越小
                enhancement_factor = 1.0 - (i / (self.num_modes - 1)) * 0.5
                processed_mode = self.adaptive_detail_enhancement(modes[i], enhancement_factor)
            
            processed_modes.append(processed_mode)
        
        # 重构图像
        reconstructed = torch.zeros_like(modes[0])
        for mode in processed_modes:
            reconstructed += mode
        
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        return reconstructed

def load_image(image_path, target_size=(256, 256)):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), image

def save_tensor_as_image(tensor, save_path):
    """将张量保存为图像文件"""
    # 确保张量在CPU上且维度正确
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 转换为numpy并保存
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """计算SSIM"""
    # 简化版SSIM计算
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, 1, 0)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, 1, 0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, 1, 0) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, 1, 0) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, 1, 0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def process_single_image(image_path, vmd_model, conservative_model, target_size=(256, 256)):
    """处理单张图像"""
    try:
        # 加载图像
        image_tensor, _ = load_image(image_path, target_size)
        
        # VMD分解
        decomposed = vmd_model(image_tensor)
        
        # 保守策略增强重构
        enhanced_reconstructed = conservative_model(decomposed)
        
        return enhanced_reconstructed, True, None
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None, False, str(e)

def evaluate_parameters(image_paths, reference_dir, color_strength, detail_enhance, noise_suppress, 
                       target_size, num_eval_images=50, num_modes=5):
    """评估参数组合的性能 - 支持多模态"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmd_model = MultiModeVMD(num_modes=num_modes).to(device)
    conservative_model = MultiModeConservativeVMD(
        num_modes=num_modes,
        color_strength=color_strength,
        detail_enhance=detail_enhance,
        noise_suppress=noise_suppress
    ).to(device)
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    # 检查图像路径和参考路径
    if not image_paths:
        print("错误: 没有找到图像文件")
        return -1000.0  # 返回一个很低的分数而不是 -inf
    
    # 使用指定数量的图像进行评估
    eval_images = image_paths[:num_eval_images]
    print(f"使用 {len(eval_images)} 张图像进行评估，模态数量: {num_modes}")
    
    for image_path in eval_images:
        try:
            # 处理图像
            enhanced_tensor, success, error_msg = process_single_image(
                image_path, vmd_model, conservative_model, target_size
            )
            
            if success and enhanced_tensor is not None:
                # 计算PSNR和SSIM（需要参考图像）
                filename = os.path.basename(image_path)
                reference_path = os.path.join(reference_dir, filename)
                
                if os.path.exists(reference_path):
                    reference_tensor, _ = load_image(reference_path, target_size)
                    psnr_val = calculate_psnr(enhanced_tensor, reference_tensor)
                    ssim_val = calculate_ssim(enhanced_tensor, reference_tensor)
                    
                    # 检查是否为有效值
                    if not np.isinf(psnr_val) and not np.isnan(psnr_val):
                        total_psnr += psnr_val
                        total_ssim += ssim_val
                        count += 1
                        # 每处理10张图像打印一次进度
                        if count % 10 == 0:
                            print(f"已处理 {count}/{len(eval_images)} 张图像, 当前PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                    else:
                        print(f"无效的度量值: {filename}, PSNR: {psnr_val}, SSIM: {ssim_val}")
                else:
                    print(f"参考图像不存在: {reference_path}")
            else:
                print(f"处理失败: {image_path}, 错误: {error_msg}")
                    
        except Exception as e:
            print(f"处理图像时出现异常: {image_path}, 错误: {e}")
            continue
    
    if count == 0:
        print("警告: 没有成功处理任何图像")
        return -1000.0  # 返回一个很低的分数而不是 -inf
    
    # 综合评分（可以调整权重）
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    composite_score = avg_psnr + 10 * avg_ssim  # SSIM权重更高
    
    print(f"评估完成: 处理 {count} 张图像, 平均 PSNR: {avg_psnr:.4f}, 平均 SSIM: {avg_ssim:.4f}, 综合分数: {composite_score:.4f}")
    
    return composite_score

def bayesian_optimization(input_dir, reference_dir, target_size=(256, 256), n_iter=20, num_eval_images=50, num_modes=5):
    """贝叶斯优化参数搜索 - 支持多模态"""
    
    # 首先检查图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    sample_images = []
    for ext in image_extensions:
        sample_images.extend(glob.glob(os.path.join(input_dir, ext)))
        sample_images.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not sample_images:
        print(f"错误: 在目录 {input_dir} 中没有找到图像文件")
        return None, None
    
    sample_images = sample_images[:num_eval_images]  # 使用指定数量的图像进行评估
    
    def black_box_function(color_strength, detail_enhance, noise_suppress):
        """黑盒目标函数"""
        # 将参数转换为合适的范围
        cs = max(0.1, min(1.0, color_strength))
        de = max(0.8, min(2.0, detail_enhance))
        ns = max(0.1, min(1.0, noise_suppress))
        
        score = evaluate_parameters(sample_images, reference_dir, cs, de, ns, target_size, num_eval_images, num_modes)
        
        # 确保分数不是无穷大
        if np.isinf(score) or np.isnan(score):
            return -1000.0
        
        return score
    
    # 定义参数边界
    pbounds = {
        'color_strength': (0.1, 1.0),
        'detail_enhance': (0.8, 2.0),
        'noise_suppress': (0.1, 1.0)
    }
    
    # 创建优化器
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    try:
        # 执行优化
        optimizer.maximize(
            init_points=5,  # 初始随机点
            n_iter=n_iter    # 优化迭代次数
        )
        
        # 获取最佳参数
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        print(f"\n贝叶斯优化完成!")
        print(f"模态数量: {num_modes}")
        print(f"最佳参数: {best_params}")
        print(f"最佳分数: {best_score:.4f}")
        
        return best_params, optimizer
    
    except Exception as e:
        print(f"贝叶斯优化失败: {e}")
        # 返回默认参数
        default_params = {'color_strength': 0.5, 'detail_enhance': 1.2, 'noise_suppress': 0.5}
        return default_params, None

def automated_parameter_tuning(train_input_dir, train_reference_dir, 
                             test_input_dir, test_reference_dir,
                             method='bayesian', target_size=(256, 256), num_eval_images=50, num_modes=5):
    """完整的自动化参数调优流程 - 支持多模态"""
    
    print("开始自动化参数调优...")
    print(f"使用 {num_eval_images} 张图像进行评估")
    print(f"模态数量: {num_modes}")
    
    # 检查目录是否存在
    if not os.path.exists(train_input_dir):
        print(f"错误: 训练输入目录不存在: {train_input_dir}")
        return None
    
    if not os.path.exists(train_reference_dir):
        print(f"错误: 训练参考目录不存在: {train_reference_dir}")
        return None
    
    if method == 'bayesian':
        best_params, optimizer = bayesian_optimization(
            train_input_dir, train_reference_dir, target_size, 
            num_eval_images=num_eval_images, num_modes=num_modes
        )
    else:
        # 这里可以添加其他优化方法，暂时只实现贝叶斯优化
        print(f"暂不支持 {method} 优化方法，使用贝叶斯优化")
        best_params, optimizer = bayesian_optimization(
            train_input_dir, train_reference_dir, target_size, 
            num_eval_images=num_eval_images, num_modes=num_modes
        )
    
    if best_params is None:
        print("参数优化失败，使用默认参数")
        best_params = {'color_strength': 0.5, 'detail_enhance': 1.2, 'noise_suppress': 0.5}
    
    # 在测试集上验证最佳参数
    print("\n在测试集上验证最佳参数...")
    
    # 获取测试图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_input_dir, ext)))
        test_images.extend(glob.glob(os.path.join(test_input_dir, ext.upper())))
    
    if test_images:
        # 使用更多的测试图像进行验证
        final_score = evaluate_parameters(
            test_images,  # 使用全部测试图像
            test_reference_dir,
            best_params['color_strength'] if isinstance(best_params, dict) else best_params[0],
            best_params['detail_enhance'] if isinstance(best_params, dict) else best_params[1],
            best_params['noise_suppress'] if isinstance(best_params, dict) else best_params[2],
            target_size,
            num_eval_images=len(test_images),  # 使用全部测试图像
            num_modes=num_modes
        )
        
        print(f"测试集最终分数: {final_score:.4f}")
    else:
        print("警告: 没有找到测试图像")
    
    return best_params

def process_dataset(input_dir, output_dir, target_size=(256, 256), 
                   color_strength=0.5, detail_enhance=1.2, noise_suppress=0.5,
                   backup_original=True, replace_original=True, num_modes=5):
    """
    处理整个数据集 - 支持多模态
    """
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建备份目录（如果需要）
    if backup_original:
        backup_dir = os.path.join(os.path.dirname(input_dir), "original_backup")
        os.makedirs(backup_dir, exist_ok=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmd_model = MultiModeVMD(num_modes=num_modes).to(device)
    conservative_model = MultiModeConservativeVMD(
        num_modes=num_modes,
        color_strength=color_strength,
        detail_enhance=detail_enhance,
        noise_suppress=noise_suppress
    ).to(device)
    
    # 查找所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_paths:
        print(f"警告: 在目录 {input_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_paths)} 张图像进行处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print(f"模态数量: {num_modes}")
    print(f"增强参数 - 颜色校正: {color_strength}, 细节增强: {detail_enhance}, 噪声抑制: {noise_suppress}")
    print("-" * 80)
    
    # 处理统计
    stats = {
        'processed': 0,
        'failed': 0,
        'errors': []
    }
    
    # 处理所有图像
    progress_bar = tqdm(image_paths, desc="处理图像")
    
    for image_path in progress_bar:
        try:
            filename = os.path.basename(image_path)
            progress_bar.set_postfix({'文件': filename})
            
            # 备份原始图像（如果需要）
            if backup_original:
                backup_path = os.path.join(backup_dir, filename)
                if not os.path.exists(backup_path):
                    shutil.copy2(image_path, backup_path)
            
            # 处理图像
            enhanced_tensor, success, error_msg = process_single_image(
                image_path, vmd_model, conservative_model, target_size
            )
            
            if success:
                # 确定输出路径
                if replace_original:
                    # 替换原始图像
                    output_path = image_path
                else:
                    # 保存到输出目录
                    output_path = os.path.join(output_dir, filename)
                
                # 保存增强后的图像
                save_tensor_as_image(enhanced_tensor, output_path)
                
                stats['processed'] += 1
                progress_bar.set_postfix({
                    '文件': filename, 
                    '状态': '完成', 
                    '成功': stats['processed'], 
                    '失败': stats['failed']
                })
            else:
                stats['failed'] += 1
                stats['errors'].append(f"{filename}: {error_msg}")
                progress_bar.set_postfix({
                    '文件': filename, 
                    '状态': '失败', 
                    '成功': stats['processed'], 
                    '失败': stats['failed']
                })
        
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{filename}: {str(e)}")
            progress_bar.set_postfix({
                '文件': filename, 
                '状态': '异常', 
                '成功': stats['processed'], 
                '失败': stats['failed']
            })
    
    # 输出处理统计
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"总图像数: {len(image_paths)}")
    print(f"成功处理: {stats['processed']}")
    print(f"处理失败: {stats['failed']}")
    print(f"成功率: {stats['processed']/len(image_paths)*100:.2f}%")
    print(f"模态数量: {num_modes}")
    
    if stats['errors']:
        print(f"\n错误详情 (前10个):")
        for i, error in enumerate(stats['errors'][:10]):
            print(f"  {i+1}. {error}")
        if len(stats['errors']) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")
    
    if backup_original:
        print(f"\n原始图像已备份到: {backup_dir}")
    
    if replace_original:
        print(f"增强图像已替换原始图像")
    else:
        print(f"增强图像已保存到: {output_dir}")

def process_lol_dataset(base_dir, mode='both', **kwargs):
    """
    处理LOL数据集
    """
    if mode in ['train', 'both']:
        train_input_dir = os.path.join(base_dir, "train", "input")
        train_output_dir = os.path.join(base_dir, "train", "enhanced")
        
        if os.path.exists(train_input_dir):
            print("开始处理训练集...")
            process_dataset(train_input_dir, train_output_dir, **kwargs)
        else:
            print(f"警告: 训练集目录不存在: {train_input_dir}")
    
    if mode in ['val', 'both']:
        val_input_dir = os.path.join(base_dir, "val", "input")
        val_output_dir = os.path.join(base_dir, "val", "enhanced")
        
        if os.path.exists(val_input_dir):
            print("\n开始处理验证集...")
            process_dataset(val_input_dir, val_output_dir, **kwargs)
        else:
            print(f"警告: 验证集目录不存在: {val_input_dir}")

if __name__ == "__main__":
    # 数据集路径 - 根据您的实际情况修改
    base_dataset_dir = "/root/autodl-tmp/SeaDiff-main/euvp"
    
    # 定义输入和参考图像路径
    train_input_dir = os.path.join(base_dataset_dir, "train", "input")
    train_reference_dir = os.path.join(base_dataset_dir, "train", "target")
    test_input_dir = os.path.join(base_dataset_dir, "val", "input")
    test_reference_dir = os.path.join(base_dataset_dir, "val", "target")
    
    # 检查目录是否存在
    if not os.path.exists(train_input_dir):
        print(f"错误: 训练输入目录不存在: {train_input_dir}")
        exit(1)
    
    if not os.path.exists(train_reference_dir):
        print(f"错误: 训练参考目录不存在: {train_reference_dir}")
        exit(1)
    
    # 选择优化方法
    optimization_method = 'bayesian'
    
    # 设置评估使用的图像数量
    num_eval_images = 100
    
    # 设置模态数量 (4, 5, 6)
    num_modes = 4  # 可以修改为4, 5, 6
    
    print("开始自动化参数调优...")
    print(f"数据集路径: {base_dataset_dir}")
    print(f"优化方法: {optimization_method}")
    print(f"评估图像数量: {num_eval_images}")
    print(f"模态数量: {num_modes}")
    print("=" * 80)
    
    try:
        # 自动化参数调优
        best_params = automated_parameter_tuning(
            train_input_dir, train_reference_dir,
            test_input_dir, test_reference_dir,
            method=optimization_method,
            num_eval_images=num_eval_images,
            num_modes=num_modes
        )
        
        if best_params is None:
            print("参数优化失败，使用默认参数")
            best_params = {'color_strength': 0.5, 'detail_enhance': 1.2, 'noise_suppress': 0.5}
        
        print(f"\n推荐的最佳参数: {best_params}")
        
        # 使用最佳参数处理整个数据集
        processing_params = {
            'target_size': (256, 256),
            'color_strength': best_params['color_strength'],
            'detail_enhance': best_params['detail_enhance'],
            'noise_suppress': best_params['noise_suppress'],
            'backup_original': True,
            'replace_original': True,
            'num_modes': num_modes
        }
        
        print(f"\n使用最佳参数处理完整数据集...")
        process_lol_dataset(base_dataset_dir, mode='both', **processing_params)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()