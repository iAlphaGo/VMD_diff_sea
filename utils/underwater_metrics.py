"""
水下图像增强评价指标计算
包括：PSNR, SSIM, MSE, UCIQE, UIQM
"""

import numpy as np
import torch
import cv2
from scipy import ndimage
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class UnderwaterMetrics:
    """水下图像质量评价指标计算类"""
    
    @staticmethod
    def calculate_psnr(enhanced, reference):
        """
        计算峰值信噪比 (PSNR)
        
        Args:
            enhanced: 增强后的图像 [0, 1]范围
            reference: 参考图像 [0, 1]范围
            
        Returns:
            PSNR值 (dB)
        """
        # 转换为numpy
        if isinstance(enhanced, torch.Tensor):
            enhanced = enhanced.detach().cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        
        # 确保在[0,1]范围内
        enhanced = np.clip(enhanced, 0, 1)
        reference = np.clip(reference, 0, 1)
        
        # 如果是3D张量，重新排列维度
        if enhanced.ndim == 3 and enhanced.shape[0] == 3:
            enhanced = np.transpose(enhanced, (1, 2, 0))
        if reference.ndim == 3 and reference.shape[0] == 3:
            reference = np.transpose(reference, (1, 2, 0))
        
        # 计算PSNR
        return peak_signal_noise_ratio(reference, enhanced, data_range=1.0)
    
    @staticmethod
    def calculate_ssim(enhanced, reference):
        """
        计算结构相似性指数 (SSIM)
        
        Args:
            enhanced: 增强后的图像 [0, 1]范围
            reference: 参考图像 [0, 1]范围
            
        Returns:
            SSIM值 (0-1之间)
        """
        # 转换为numpy
        if isinstance(enhanced, torch.Tensor):
            enhanced = enhanced.detach().cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        
        # 确保在[0,1]范围内
        enhanced = np.clip(enhanced, 0, 1)
        reference = np.clip(reference, 0, 1)
        
        # 如果是3D张量，重新排列维度
        if enhanced.ndim == 3 and enhanced.shape[0] == 3:
            enhanced = np.transpose(enhanced, (1, 2, 0))
        if reference.ndim == 3 and reference.shape[0] == 3:
            reference = np.transpose(reference, (1, 2, 0))
        
        # 如果是彩色图像，转换为灰度
        if enhanced.ndim == 3:
            enhanced_gray = 0.299 * enhanced[:,:,0] + 0.587 * enhanced[:,:,1] + 0.114 * enhanced[:,:,2]
            reference_gray = 0.299 * reference[:,:,0] + 0.587 * reference[:,:,1] + 0.114 * reference[:,:,2]
        else:
            enhanced_gray = enhanced
            reference_gray = reference
        
        # 计算SSIM
        return structural_similarity(
            reference_gray, enhanced_gray, 
            data_range=1.0, 
            gaussian_weights=True, 
            sigma=1.5, 
            use_sample_covariance=False
        )
    
    @staticmethod
    def calculate_mse(enhanced, reference):
        """
        计算均方误差 (MSE)
        
        Args:
            enhanced: 增强后的图像 [0, 1]范围
            reference: 参考图像 [0, 1]范围
            
        Returns:
            MSE值
        """
        # 转换为numpy
        if isinstance(enhanced, torch.Tensor):
            enhanced = enhanced.detach().cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        
        # 确保在[0,1]范围内
        enhanced = np.clip(enhanced, 0, 1)
        reference = np.clip(reference, 0, 1)
        
        return np.mean((enhanced - reference) ** 2)
    
    @staticmethod
    def calculate_uciqe(img):
        """
        计算水下彩色图像质量评价 (UCIQE) - 修正版本
        
        Args:
            img: 输入图像 [0, 1]范围, RGB格式
            
        Returns:
            UCIQE值 (正常范围0-1)
        """
        # 转换为numpy
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # 确保在[0,1]范围内
        img = np.clip(img, 0, 1)
        
        # 如果是3D张量，重新排列维度
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # 转换为[0,255]范围，uint8类型
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 将RGB转换为BGR（因为OpenCV使用BGR）
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        
        # 计算色度的标准差（归一化到0-1）
        delta = np.std(H) / 180.0  # 色度的标准差，H范围是0-180
        
        # 计算饱和度的平均值（归一化到0-1）
        mu = np.mean(S) / 255.0  # 饱和度的平均值，S范围是0-255
        
        # 计算亮度对比度
        n, m = V.shape
        number = math.floor(n * m / 100)  # 所需像素的个数
        Maxsum, Minsum = 0, 0
        V1, V2 = V / 255.0, V / 255.0  # 归一化到0-1
        
        # 计算前1%最亮像素的平均值
        for i in range(1, number + 1):
            Maxvalue = np.max(V1)
            x, y = np.where(V1 == Maxvalue)
            Maxsum = Maxsum + V1[x[0], y[0]]
            V1[x[0], y[0]] = 0
        top = Maxsum / number
        
        # 计算前1%最暗像素的平均值
        for i in range(1, number + 1):
            Minvalue = np.min(V2)
            X, Y = np.where(V2 == Minvalue)
            Minsum = Minsum + V2[X[0], Y[0]]
            V2[X[0], Y[0]] = 1
        bottom = Minsum / number
        
        conl = top - bottom  # 对比度
        
        # UCIQE计算公式（使用论文中的系数）
        uciqe = 0.468 * delta + 0.2745 * conl + 0.2576 * mu
        
        return uciqe
    
    @staticmethod
    def calculate_uiqm(img):
        """
        计算水下图像质量度量 (UIQM)
        UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
        
        Args:
            img: 输入图像 [0, 1]范围, RGB格式
            
        Returns:
            UIQM值
        """
        # 转换为numpy
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # 确保在[0,1]范围内
        img = np.clip(img, 0, 1)
        
        # 如果是3D张量，重新排列维度
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # 转换为[0,255]范围，uint8类型
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 将RGB转换为BGR（因为OpenCV使用BGR）
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        # 计算三个分量
        uicm = UnderwaterMetrics._calculate_uicm(img_bgr)
        uism = UnderwaterMetrics._calculate_uism(img_bgr)
        uiconm = UnderwaterMetrics._calculate_uiconm(img_bgr)
        
        # UIQM计算公式
        uiqm = 0.15 * uicm + 0.25 * uism + 0.6 * uiconm
        
        return uiqm
    
    @staticmethod
    def _calculate_uicm(img_bgr):
        """
        计算UICM (水下图像颜色度量)
        基于BGR图像
        
        Args:
            img_bgr: BGR格式的图像，[0,255]范围
            
        Returns:
            UICM值
        """
        b, g, r = cv2.split(img_bgr)
        
        # 计算RG和YB分量
        RG = r.astype(np.float32) - g.astype(np.float32)
        YB = (r.astype(np.float32) + g.astype(np.float32)) / 2 - b.astype(np.float32)
        
        m, n = r.shape  # 图像尺寸
        K = m * n
        
        # 参数设置
        alpha_L = 0.1
        alpha_R = 0.1
        T_alpha_L = math.ceil(alpha_L * K)  # 向上取整
        T_alpha_R = math.floor(alpha_R * K)  # 向下取整
        
        # 处理RG分量
        RG_list = RG.flatten()
        RG_list = np.sort(RG_list)
        
        # 去掉最大和最小的alpha%像素
        sum_RG = 0
        for i in range(T_alpha_L + 1, K - T_alpha_R):
            sum_RG += RG_list[i]
        
        U_RG = sum_RG / (K - T_alpha_R - T_alpha_L)
        
        # 计算RG的方差
        squ_RG = 0
        for i in range(K):
            squ_RG += np.square(RG_list[i] - U_RG)
        sigma2_RG = squ_RG / K
        
        # 处理YB分量
        YB_list = YB.flatten()
        YB_list = np.sort(YB_list)
        
        sum_YB = 0
        for i in range(T_alpha_L + 1, K - T_alpha_R):
            sum_YB += YB_list[i]
        
        U_YB = sum_YB / (K - T_alpha_R - T_alpha_L)
        
        # 计算YB的方差
        squ_YB = 0
        for i in range(K):
            squ_YB += np.square(YB_list[i] - U_YB)
        sigma2_YB = squ_YB / K
        
        # 计算UICM
        uicm = -0.0268 * np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586 * np.sqrt(sigma2_RG + sigma2_YB)
        
        return uicm
    
    @staticmethod
    def _calculate_uism(img_bgr):
        """
        计算UISM (水下图像锐度度量)
        基于BGR图像，使用EME方法
        
        Args:
            img_bgr: BGR格式的图像，[0,255]范围
            
        Returns:
            UISM值
        """
        b, g, r = cv2.split(img_bgr)
        
        # 计算每个通道的EME
        eme_b = UnderwaterMetrics._calculate_eme(b, 8)
        eme_g = UnderwaterMetrics._calculate_eme(g, 8)
        eme_r = UnderwaterMetrics._calculate_eme(r, 8)
        
        # 加权求和得到UISM
        uism = 0.299 * eme_r + 0.587 * eme_g + 0.114 * eme_b
        
        return uism
    
    @staticmethod
    def _calculate_eme(channel, L):
        """
        计算增强测量评估 (EME) 用于单个通道
        
        Args:
            channel: 单通道图像
            L: 块大小
            
        Returns:
            EME值
        """
        m, n = channel.shape
        number_m = math.floor(m / L)
        number_n = math.floor(n / L)
        
        m1 = 0
        E = 0
        
        for i in range(number_m):
            n1 = 0
            for t in range(number_n):
                A1 = channel[m1:m1+L, n1:n1+L]
                channel_min = np.min(A1)
                channel_max = np.max(A1)
                
                if channel_min > 0:
                    channel_ratio = channel_max / channel_min
                else:
                    channel_ratio = channel_max
                
                E += np.log(channel_ratio + 1e-5)  # 避免log(0)
                
                n1 += L
            m1 += L
        
        # 计算平均EME
        if number_m * number_n > 0:
            eme_sum = 2 * E / (number_m * number_n)
        else:
            eme_sum = 0
        
        return eme_sum
    
    @staticmethod
    def _calculate_uiconm(img_bgr):
        """
        计算UIConM (水下图像对比度度量)
        基于BGR图像，使用PLIP方法
        
        Args:
            img_bgr: BGR格式的图像，[0,255]范围
            
        Returns:
            UIConM值
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        m, n = gray.shape
        L = 8  # 块大小
        number_m = math.floor(m / L)
        number_n = math.floor(n / L)
        
        m1 = 0
        logAMEE = 0
        
        for i in range(number_m):
            n1 = 0
            for t in range(number_n):
                A1 = gray[m1:m1+L, n1:n1+L]
                gray_min = int(np.min(A1))
                gray_max = int(np.max(A1))
                
                # PLIP操作
                plip_add = gray_max + gray_min - gray_max * gray_min / 1026.0
                
                if 1026 - gray_min > 0:
                    plip_del = 1026 * (gray_max - gray_min) / (1026.0 - gray_min)
                    
                    if plip_del > 0 and plip_add > 0:
                        local_a = plip_del / plip_add
                        local_b = math.log(plip_del / plip_add)
                        phi = local_a * local_b
                        logAMEE += phi
                
                n1 += L
            m1 += L
        
        # 计算最终UIConM
        if number_m * number_n > 0:
            logAMEE = 1026 - 1026 * ((1 - logAMEE / 1026.0) ** (1.0 / (number_n * number_m)))
        else:
            logAMEE = 0
        
        return logAMEE
    
    @staticmethod
    def calculate_all_metrics(enhanced, reference):
        """
        计算所有评价指标
        
        Args:
            enhanced: 增强后的图像
            reference: 参考图像
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 全参考指标
        metrics['psnr'] = UnderwaterMetrics.calculate_psnr(enhanced, reference)
        metrics['ssim'] = UnderwaterMetrics.calculate_ssim(enhanced, reference)
        metrics['mse'] = UnderwaterMetrics.calculate_mse(enhanced, reference)
        
        # 无参考指标（只使用增强图像）
        metrics['uciqe'] = UnderwaterMetrics.calculate_uciqe(enhanced)
        metrics['uiqm'] = UnderwaterMetrics.calculate_uiqm(enhanced)
        
        return metrics
    
    @staticmethod
    def batch_calculate_metrics(enhanced_batch, reference_batch):
        """
        批量计算评价指标
        
        Args:
            enhanced_batch: 增强后的图像批次 [B, C, H, W]
            reference_batch: 参考图像批次 [B, C, H, W]
            
        Returns:
            平均指标字典
        """
        batch_size = enhanced_batch.shape[0]
        all_metrics = {
            'psnr': [], 'ssim': [], 'mse': [], 'uciqe': [], 'uiqm': []
        }
        
        for i in range(batch_size):
            enhanced = enhanced_batch[i]
            reference = reference_batch[i]
            
            try:
                metrics = UnderwaterMetrics.calculate_all_metrics(enhanced, reference)
                
                for key in metrics:
                    all_metrics[key].append(metrics[key])
            except Exception as e:
                print(f"计算第{i}个样本指标时出错: {e}")
                # 添加默认值
                for key in all_metrics:
                    all_metrics[key].append(0.0)
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics:
            if all_metrics[key]:  # 确保列表不为空
                avg_metrics[key] = np.mean(all_metrics[key])
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics