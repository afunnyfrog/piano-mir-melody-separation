import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Padding=1 確保輸出的長寬不變 (Same Padding)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AudioUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(AudioUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder (Downscaling) ---
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = DoubleConv(16, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        
        # MaxPool
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bot = DoubleConv(128, 256)

        # --- Decoder (Upscaling) ---
        # 使用 Transpose Conv 放大
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 256 是因為 concat 之後通道變兩倍
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)

        # --- Output Layer ---
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, 1, 1024, 256)
        
        # Encoder
        x1 = self.inc(x)            # -> (16, 1024, 256)
        p1 = self.pool(x1)          # -> (16, 512, 128)
        
        x2 = self.down1(p1)         # -> (32, 512, 128)
        p2 = self.pool(x2)          # -> (32, 256, 64)
        
        x3 = self.down2(p2)         # -> (64, 256, 64)
        p3 = self.pool(x3)          # -> (64, 128, 32)
        
        x4 = self.down3(p3)         # -> (128, 128, 32)
        p4 = self.pool(x4)          # -> (128, 64, 16)
        
        # Bottleneck
        x5 = self.bot(p4)           # -> (256, 64, 16)
        
        # Decoder
        # 1. Upsample
        u1 = self.up1(x5)           # -> (128, 128, 32)
        # 2. Concat (Skip Connection)
        u1 = torch.cat([u1, x4], dim=1) 
        # 3. Conv
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv4(u4)

        logits = self.outc(u4)      # -> (2, 1024, 256)
        return logits