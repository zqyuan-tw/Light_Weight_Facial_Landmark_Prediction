import torch
import torch.nn as nn
import torchvision.ops as ops

class ELFace(nn.Module):
    """
    Light-Weight Distilled HRNet for Facial Landmark Detection
    """
    def __init__(self) -> None:
        super(ELFace, self).__init__()
        self.downsample = nn.Sequential(
            BN_Conv2d_Leaky(3, 6, 3, 2, 1),
            BN_Conv2d_Leaky(6, 12, 3, 2, 1),
            Unit(12, 24),
            Unit(24, 48)
        )
        self.unit3 = Unit(48, 72)
        self.unit4 = Unit(72, 96)
        self.fpn = ops.FeaturePyramidNetwork([48, 72, 96], 24)
        self.maxpool = nn.MaxPool2d(2)
        self.regressor = nn.Sequential(            
            nn.Linear(24 * 12 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, 68 * 2),
            nn.Tanh()
        )

    def forward(self, x):   # 3, 384, 384
        # print('elf1', x.shape)
        x = self.downsample(x)  # 48, 24, 24
        # print('elf2', x.shape)
        x1 = self.unit3(x)  # 72, 12, 12
        # print('elf3', x1.shape)
        x2 = self.unit4(x1) # 96, 6, 6
        # print('elf4', x2.shape)
        x = self.fpn({'a': x, 'b': x1, 'c': x2})['a']
        x = self.maxpool(x)
        # print('elf5', x.shape)
        x = x.view(x.size(0), -1)
        # print('elf6', x.shape)
        x = self.regressor(x)
        # print('elf7', x.shape)

        return x.view(x.size(0), 68, 2)

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Unit, self).__init__()
        self.conv1 = BN_Conv2d_Leaky(in_channels, in_channels // 2, 1, 1, 0)
        self.blocks = nn.Sequential(
            BN_Conv2d_Leaky(in_channels // 2, in_channels // 2, 3, 2, 1),
            BN_Conv2d_Leaky(in_channels // 2, in_channels, 1, 1, 0),
            Block(in_channels),
            Block(in_channels),
            Block(in_channels),
            Block(in_channels),
            BN_Conv2d_Leaky(in_channels, in_channels // 2, 1, 1, 0)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = BN_Conv2d_Leaky(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        # print('u0', x.shape)
        x = self.conv1(x)
        # print('u1', x.shape)
        out = self.blocks(x)
        # print('u2', out.shape)
        x = self.maxpool(x)
        # print('u3', x.shape)
        x = torch.cat((out, x), 1)
        # print('u4', x.shape)
        return self.conv2(x)

class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.blocks = nn.Sequential(
            BN_Conv2d_Leaky(in_channels, in_channels // 2, 1, 1, 0),
            BN_Conv2d_Leaky(in_channels // 2, in_channels // 2, 3, 1, 1),
            nn.Conv2d(in_channels // 2, in_channels, 1, 1, 0)
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        # print('b1', x.shape)
        out = self.blocks(x)
        # print('b2', out.shape)
        out += x
        # print('b3', out.shape)
        return self.leaky(out)

class BN_Conv2d_Leaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BN_Conv2d_Leaky, self).__init__()
        self.bn_conv_leaky = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.bn_conv_leaky(x)


if __name__ == "__main__":
    from torchinfo import summary
    m = ELFace()
    print(summary(m, input_size=(2, 3, 384, 384)))
    # (2, 3, 384, 384)