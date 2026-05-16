import torch
import torch.nn as nn
import torch.nn.functional as F

class UConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = UConv(in_ch+skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder1 = UConv(in_ch, 64)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.encoder2 = UConv(64, 128)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.encoder3 = UConv(128, 256)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.encoder4 = UConv(256, 512)
        self.pooling4 = nn.MaxPool2d(2, stride=2)
        self.bottleneck = UConv(512, 1024)
        self.upblock1 = UpBlock(1024, 512, 512)
        self.upblock2 = UpBlock(512, 256, 256)
        self.upblock3 = UpBlock(256, 128, 128)
        self.upblock4 = UpBlock(128, 64, 64)
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.pooling1(x1)
        x2 = self.encoder2(x)
        x = self.pooling2(x2)
        x3 = self.encoder3(x)
        x = self.pooling3(x3)
        x4 = self.encoder4(x)
        x = self.pooling4(x4)
        x = self.bottleneck(x)
        x = self.upblock1(x, x4)
        x = self.upblock2(x, x3)
        x = self.upblock3(x, x2)
        x = self.upblock4(x, x1)
        x = self.out_conv(x)
        return x

