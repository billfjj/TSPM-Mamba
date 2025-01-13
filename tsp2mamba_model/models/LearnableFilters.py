import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        # x → 1 16 512  这里按第二个维度也就是通道维度view进行分组，最后变成512 其实就是把每个组内的2个通道乘上宽高 2*W*H https://blog.csdn.net/weixin_44492824/article/details/124025689
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SDM(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False,  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        scale_factor = abs(1 - self.gn.gamma) + self.gn.beta
        normal_scale_factor = abs(1 - self.gn.gamma) + self.gn.beta / sum(abs(1 - self.gn.gamma) + self.gn.beta)
        re_weights = gn_x * normal_scale_factor
        new_input = self.sigomid(re_weights + x)
        info_mask = new_input >= self.gate_treshold
        x_s = info_mask * x
        return x_s


class FDM(nn.Module):
    # def __init__(self, dim: int = 256, h: int = 64, w: int = 64):
    def __init__(self, dim: int = 512, h: int = 128, w: int = 128):
        super().__init__()
        self.h = h
        self.w = w
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W, = x.shape  # 1 28 16 3
        # 第一个维度 (1): 表示批次大小，这里为1。
        # 第二个维度 (28): 表示输入张量的高度。
        # 第三个维度 (9): 表示变换后的宽度，对应输入宽度的一半加一（16//2 + 1）。
        # 第四个维度 (3): 表示通道数。
        # 这个输出表明在每个通道的每个位置上都有一个复数值，其中包含了傅里叶变换的实部和虚部。如果需要访问实部和虚部，可以使用索引 [..., :, :, 0] 和 [..., :, :, 1]。
        # print(x.shape)
        x = torch.fft.rfft2(x, dim=(1), norm='ortho')
        # 在 PyTorch 中，torch.view_as_complex 是一个函数，
        # 用于将两个实部和虚部的实数张量视为一个复数张量。这个函数在处理复数张量时非常有用，特别是在涉及频域操作（如傅里叶变换）时。
        weight = torch.view_as_complex(self.complex_weight)
        start_idx_h = torch.randint(0, self.h - H + 1, (1,))
        start_idx_w = torch.randint(0, self.w - W + 1, (1,))
        # print(start_idx_w)
        # print(start_idx_h)
        # print(x.shape)
        # print(weight.shape)
        selected_weight = weight[:, start_idx_h:start_idx_h + H, start_idx_w:start_idx_w + W]
        _x = x * selected_weight
        x = torch.fft.irfft2(_x, s=(C), dim=(1), norm='ortho')
        return x


class CDM(nn.Module):
    def __init__(self, oup_channels: int, alpha: float = 1 / 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.op_channel = oup_channels
        self.up_channel = up_channel = int(alpha * oup_channels)  # 计算上层通道数
        self.low_channel = low_channel = oup_channels - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(self.op_channel, up_channel, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(self.op_channel, low_channel, kernel_size=1, bias=False)  # 创建卷积层
        # 组卷积
        self.GWC1 = nn.Conv2d(up_channel, up_channel, kernel_size=group_kernel_size, stride=1,
                              padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        # 组卷积
        self.GWC2 = nn.Conv2d(low_channel, low_channel, kernel_size=group_kernel_size, stride=1,
                              padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        # 逐点卷积 其实就是kernel_size = 1的卷积
        self.PWC1 = nn.Conv2d(up_channel, up_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel, low_channel, kernel_size=1, bias=False)  # 创建卷积层
        self.GAP = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        x_sf_up = self.squeeze1(x)
        x_sf_down = self.squeeze2(x)

        y1 = self.PWC1(x_sf_up) + self.GWC2(x_sf_down)
        y2 = self.PWC2(x_sf_down) + self.PWC2(x_sf_down)
        x1 = self.GAP(y1)
        x2 = self.GAP(y2)
        x_new = F.softmax(torch.cat([x1, x2], dim=1), dim=1)
        out1, out2 = torch.split(x_new, x_new.size(1) // 2, dim=1)
        c1 = out1 * y1
        c2 = out2 * y2
        out = torch.cat([c1, c2], dim=1)
        return out


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class MDFD(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SDM = SDM(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.FDM = FDM(dim=int(op_channel / 2 + 1), h=64, w=64)  # 创建 CRU 层
        self.CDM = CDM(op_channel, alpha=alpha, group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        xs = self.SDM(x)
        xf = self.FDM(xs)
        xc = self.CDM(xf)
        return xc


class LearnableFilters(nn.Module):
    def __init__(self, dim, h, w):
        super(LearnableFilters, self).__init__()

        self.dim = dim
        self.h = h
        self.w = w
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化可学习的复权重
        complex_weight_low_tensor = torch.randn(dim, h, w, 2, dtype=torch.float32).to(self.device) * 0.02
        self.complex_weight_low = complex_weight_low_tensor.contiguous()
        complex_weight_high_tensor = torch.randn(dim, h, w, 2, dtype=torch.float32).to(self.device) * 0.02
        self.complex_weight_high = complex_weight_high_tensor.contiguous()

        complex_weight_band_tensor = torch.randn(dim, h, w, 2, dtype=torch.float32).to(self.device) * 0.02
        self.complex_weight_band = complex_weight_band_tensor.contiguous()

    def forward(self, x):
        _, _, h, w = x.shape
        ch, cw = h // 2, w // 2

        # 根据输入图像大小动态创建滤波器掩膜
        low_size = h // 4
        high_size = h // 4
        band_size = h // 2

        # 创建低通滤波器掩膜
        mask_low = torch.zeros_like(x, dtype=torch.complex64).to(self.device)
        mask_low[:, :, ch - low_size:ch + low_size + 1, cw - low_size:cw + low_size + 1] = 1

        # 创建高通滤波器掩膜
        mask_high = torch.ones_like(x, dtype=torch.complex64).to(self.device)
        mask_high[:, :, ch - high_size:ch + high_size + 1, cw - high_size:cw + high_size + 1] = 0

        # 创建带通滤波器掩膜
        mask_band = torch.zeros_like(x, dtype=torch.complex64).to(self.device)
        band_inner_size = band_size // 2
        mask_band[:, :, ch - band_size:ch - band_inner_size, cw - band_size:cw - band_inner_size] = 1
        mask_band[:, :, ch + band_inner_size + 1:ch + band_size + 1,
        cw + band_inner_size + 1:cw + band_size + 1] = 1

        # 对图像进行傅里叶变换
        fft_tensor = torch.fft.fft2(x)
        # 中心化处理
        fft_shift_tensor = torch.fft.fftshift(fft_tensor)

        # 将复权重转换为复数形式
        weight_low = torch.view_as_complex(self.complex_weight_low.contiguous())
        weight_high = torch.view_as_complex(self.complex_weight_high.contiguous())
        weight_band = torch.view_as_complex(self.complex_weight_band.contiguous())

        # 应用低通滤波器
        fft_shift_low = fft_shift_tensor * mask_low
        fft_low = torch.fft.ifftshift(fft_shift_low)
        image_low = torch.fft.ifft2(fft_low)
        # image_low = torch.abs(image_low)

        # 应用高通滤波器
        fft_shift_high = fft_shift_tensor * mask_high
        fft_high = torch.fft.ifftshift(fft_shift_high)
        image_high = torch.fft.ifft2(fft_high)
        # image_high = torch.abs(image_high)

        # 应用带通滤波器
        fft_shift_band = fft_shift_tensor * mask_band
        fft_band = torch.fft.ifftshift(fft_shift_band)
        image_band = torch.fft.ifft2(fft_band)
        # image_band = torch.abs(image_band)

        # 对频域图像应用不同的滤波器
        fft_shift_low = image_low * weight_low
        fft_shift_high = image_high * weight_high
        fft_shift_band = image_band * weight_band

        # 逆傅里叶变换并获取幅值
        fft_low = torch.fft.ifftshift(fft_shift_low)
        image_low = torch.fft.ifft2(fft_low)
        # image_low = torch.abs(image_low)

        fft_high = torch.fft.ifftshift(fft_shift_high)
        image_high = torch.fft.ifft2(fft_high)
        # image_high = torch.abs(image_high)

        fft_band = torch.fft.ifftshift(fft_shift_band)
        image_band = torch.fft.ifft2(fft_band)
        # image_band = torch.abs(image_band)

        fft_shift_tensor = torch.fft.ifftshift(fft_shift_tensor)
        fft_shift_tensor = torch.fft.ifft2(fft_shift_tensor)
        # fft_shift_tensor = torch.abs(fft_shift_tensor)

        return torch.abs(fft_shift_tensor + image_low + image_high + image_band)
        # return fft_shift_tensor + image_low + image_high + image_band


# if __name__ == '__main__':
#     # x = torch.randn(1, 384,16, 16)  # 创建随机输入张量
#     # model = FDM(193,16,16)  # 创建 ScConv 模型
#     # print(model(x).shape)  # 打印模型输出的形状
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(8, 96, 128, 128).to(device)
#     model = LearnableFilters(96, 128, 128)
#     print(model(x).shape)
