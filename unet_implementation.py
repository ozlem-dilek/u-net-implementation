import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()

        if mid_channels == None:
            mid_channels = out_channels

        self.doubleConv = nn.Sequential(
              nn.Conv2d(in_channels = in_channels, out_channels = mid_channels, kernel_size = 3, padding = 1, stride = 1),
              nn.BatchNorm2d(mid_channels),
              nn.ReLU(inplace = True),
              nn.Conv2d(in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = 1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace = True)
          )

    def forward(self, x):
          return self.doubleConv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels = None):
    super().__init__()

    self.doubleConv = DoubleConv(in_channels = in_channels, out_channels = out_channels, mid_channels = mid_channels)
    self.maxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)

  def forward(self, x):
    x = self.doubleConv(x)
    return self.maxPool(x)

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels = None):
    super().__init__()

    self.deConv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = in_channels//2, kernel_size = 2, stride = 2)
    self.doubleConv = DoubleConv(in_channels = in_channels, out_channels = out_channels)

  def forward(self, x, x_skip):
    x = self.deConv(x)
    x = torch.cat((x, x_skip), dim = 1)
    return self.doubleConv(x)