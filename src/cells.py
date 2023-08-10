import torch
import torch.nn as nn
from src.layers import MaxPool, BranchSeparables, ReluConvBn, Reduction

class CellBase(nn.Module):
    has_comb_iter_4_right: bool = False

    def cell_forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.has_comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        return  torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        

class CellStem0(CellBase):
    def __init__(self, in_channels_left: int, out_channels_left: int, in_channels_right: int, out_channels_right: int):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(in_channels_left, out_channels_left, kernel_size=5, stride=2, stem_cell=True)
        self.comb_iter_0_right = nn.Sequential(
            MaxPool(3, stride=2),
            nn.Conv2d(in_channels_left, out_channels_left, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_left, eps=0.001)
        )
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=7, stride=2)
        self.comb_iter_1_right = MaxPool(3, stride=2)
        self.comb_iter_2_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=5, stride=2)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3, stride=2)
        self.comb_iter_3_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=2)
        self.comb_iter_4_left = BranchSeparables(in_channels_right, out_channels_right, kernel_size=3, stride=2, stem_cell=True)
        self.has_comb_iter_4_right = True
        self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, kernel_size=1, stride=2)

    def forward(self, x_left: torch.Tensor) -> torch.Tensor:
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out

class Cell(CellBase):
    def __init__(self, in_channels_left: int, out_channels_left: int, in_channels_right: int, out_channels_right: int, 
                 is_reduction: bool = False, zero_pad: bool = False, match_prev_layer_dimensions: bool = False):
        super(Cell, self).__init__()

        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = Reduction(in_channels_left, out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left, out_channels_left, kernel_size=1)

        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left, out_channels_left, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=7, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_3_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left, out_channels_left, kernel_size=3, stride=stride, zero_pad=zero_pad)
        if is_reduction:
            self.has_comb_iter_4_right = True
        else:
            self.has_comb_iter_4_right = False
        self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, kernel_size=1, stride=stride)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out