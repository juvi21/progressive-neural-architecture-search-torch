import torch
import torch.nn as nn
from src.cells import CellStem0, Cell

class PNASNet(nn.Module):
    def __init__(self, num_classes: int = 1001):
        super(PNASNet, self).__init__()
        self.num_classes = num_classes
        self._initialize_layers()

    def _initialize_layers(self):
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(96, eps=0.001)
        )
        self.cell_stem_0 = CellStem0(96, 54, 96, 54)
        self.cell_stem_1 = Cell(96, 108, 270, 108, True, match_prev_layer_dimensions=True)
        self.cell_0 = Cell(270, 216, 540, 216, match_prev_layer_dimensions=True)
        self.cell_1 = Cell(540, 216, 1080, 216)
        self.cell_2 = Cell(1080, 216, 1080, 216)
        self.cell_3 = Cell(1080, 216, 1080, 216)
        self.cell_4 = Cell(1080, 432, 1080, 432, True, True)
        self.cell_5 = Cell(1080, 432, 2160, 432, match_prev_layer_dimensions=True)
        self.cell_6 = Cell(2160, 432, 2160, 432)
        self.cell_7 = Cell(2160, 432, 2160, 432)
        self.cell_8 = Cell(2160, 864, 2160, 864, True)
        self.cell_9 = Cell(2160, 864, 4320, 864, match_prev_layer_dimensions=True)
        self.cell_10 = Cell(4320, 864, 4320, 864)
        self.cell_11 = Cell(4320, 864, 4320, 864)
        
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, self.num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return x_cell_11

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.features(input)
        x = self.logits(x)
        return x