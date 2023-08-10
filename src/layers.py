import torch
import torch.nn as nn

EPSILON = 0.001

class MaxPool(nn.Module):
    
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 1, zero_pad: bool = False):
        super().__init__()
        self.is_zero_padded = zero_pad
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_zero_padded:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.is_zero_padded:
            x = x[:, :, 1:, 1:]
        return x

class SeparableConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dw_kernel_size: int, dw_stride: int, dw_padding: int):
        super().__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel_size,
                                          stride=dw_stride, padding=dw_padding, groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv2d(x)
        return self.pointwise_conv2d(x)

class BranchSeparables(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 stem_cell: bool = False, zero_pad: bool = False):
        super().__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.is_zero_padded = zero_pad
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels, kernel_size, stride, padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=EPSILON)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels, kernel_size, 1, padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_1(x)
        if self.is_zero_padded:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.is_zero_padded:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        return self.bn_sep_2(x)

class ReluConvBn(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.conv(x)
        return self.bn(x)

class Reduction(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2, count_include_pad=False),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        )
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.path_2_pool = nn.AvgPool2d(1, stride=2, count_include_pad=False)
        self.path_2_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x_path1 = self.path_1(x)
        x_path2 = self.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2_pool(x_path2)
        x_path2 = self.path_2_conv(x_path2)

        return self.final_path_bn(torch.cat([x_path1, x_path2], 1))