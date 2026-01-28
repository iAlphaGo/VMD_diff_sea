import math

import numpy as np
import torch
from torch import nn
from utils.utils import get_A


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1,
        is_noise: bool = True,
    ):
        super().__init__()
        self.is_noise = is_noise
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(x))
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels, n_channels, time_channels, is_noise=is_noise
        )
        self.dia1 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2)
        )
        self.dia2 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4)
        )
        self.dia3 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8)
        )
        self.dia4 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16)
        )
        self.res2 = ResidualBlock(
            n_channels, n_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.dia1(x)
        x = self.dia2(x)
        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Denoise_UNet(nn.Module):
    def __init__(
        self, input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise
    ):
        super().__init__()
        n_resolutions = len(ch_mults)
        
        # 修改：增加输入通道数以接受VMD模态
        # 原始输入通道：noisy_image(3) + condition(3) + J(3) + hist(3) = 12
        # 加上VMD模态：4个模态 × 3通道 = 12
        # 总输入通道：12 + 12 = 24
        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
            in_channels = n_channels * (ch_mults[i - 1] if i >= 1 else 1)
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise)
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = torch.tensor([0]).cuda()):
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(x))


class Beta_UNet(nn.Module):
    def __init__(self, input_channels, output_channels, n_channels, ch_mults, n_blocks):
        super().__init__()
        is_noise = False
        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)
        self.act = Swish()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = nn.Sequential(nn.Linear(128, 3), Swish(), nn.Linear(3, 3))

    def forward(self, x: torch.Tensor):
        t = None
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        x = torch.sigmoid(self.transform(self.pool(x).squeeze()))
        return x.unsqueeze(-1).unsqueeze(-1)


class DocDiff(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        n_channels,
        ch_mults,
        n_blocks,
    ):
        super(DocDiff, self).__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        
        # 修改：增加输入通道数以接受VMD模态
        # 原始输入通道：12
        # 加上VMD模态：4个模态 × 3通道 = 12
        # 总输入通道：24
        self.denoiser = Denoise_UNet(
            24, 3, n_channels, ch_mults, n_blocks, is_noise=True  # 修改：12改为24
        )

    def forward(self, x, condition, hist, depth, t, diffusion, vmd_modes=None):
        """
        参数说明：
        x: 真实图像 [B, 3, H, W]
        condition: 条件图像 [B, 3, H, W]
        hist: 直方图 [B, 3, H, W]
        depth: 深度图 [B, 3, H, W]
        t: 时间步 [B]
        diffusion: 扩散模型
        vmd_modes: VMD模态 [B, 12, H, W] (4个模态 × 3通道)
        """
        # 原始计算流程保持不变
        pred_beta = self.beta_predictor(condition)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        T_direct = torch.clamp((torch.exp(-pred_beta * depth)), 0, 1)
        T_scatter = torch.clamp((1 - torch.exp(-pred_beta * depth)), 0, 1)
        
        # 计算大气光
        from utils.utils import get_A
        atm_light = [get_A(item) for item in condition]
        atm_light = torch.stack(atm_light).to(x.device)
        
        # 计算场景辐射
        J = torch.clamp(((condition - T_scatter * atm_light) / (T_direct + 1e-8)), 0, 1)

        # 添加噪声
        noisy_image, noise_ref = diffusion.noisy_image(t, x)
        
        # 构建去噪器输入
        # 原始输入：noisy_image(3) + condition(3) + J(3) + hist(3) = 12通道
        # 如果提供VMD模态，则加入：vmd_modes(12) = 总24通道
        if vmd_modes is not None:
            # vmd_modes形状应为 [B, 12, H, W]
            denoiser_input = torch.cat((noisy_image, condition, J, hist, vmd_modes), dim=1)
        else:
            # 如果没有VMD模态，使用零填充
            B, _, H, W = condition.shape
            zero_vmd = torch.zeros(B, 12, H, W).to(condition.device)
            denoiser_input = torch.cat((noisy_image, condition, J, hist, zero_vmd), dim=1)
        
        # 去噪
        denoised_J = self.denoiser(denoiser_input, t)
        
        return J, noise_ref, denoised_J, T_direct, T_scatter


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)



if __name__ == "__main__":
    import argparse

    import torchsummary
    from schedule.diffusionSample import GaussianDiffusion
    from schedule.schedule import Schedule
    from src.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="../conf.yml", help="path to the config.yaml file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print("Config loaded")
    model = DocDiff(
        input_channels=config.CHANNEL_X + config.CHANNEL_Y,
        output_channels=config.CHANNEL_Y,
        n_channels=config.MODEL_CHANNELS,
        ch_mults=config.CHANNEL_MULT,
        n_blocks=config.NUM_RESBLOCKS,
    )
    schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
    diffusion = GaussianDiffusion(model, config.TIMESTEPS, schedule)
    model.eval()
    print(
        torchsummary.summary(
            model.init_predictor.cuda(), [(3, 128, 128)], batch_size=32
        )
    )
