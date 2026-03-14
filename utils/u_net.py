import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(AudioUNet, self).__init__()

        # --- Encoder (下採樣) ---
        # 輸入現在是 [Batch, 1, 1025, Time]
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # --- Decoder (上採樣) ---
        self.dec5 = self._make_dec_layer(256 + 256, 128)
        self.dec4 = self._make_dec_layer(128 + 128, 64)
        self.dec3 = self._make_dec_layer(64 + 64, 32)
        self.dec2 = self._make_dec_layer(32 + 32, 16)

        self.final = nn.Conv2d(16 + 16, n_classes, kernel_size=1)
        self.relu = nn.ReLU() # 頻譜能量為正數，不再使用 Tanh

    def _make_dec_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [Batch, 1, Freq, Time]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)

        # Decoder + Skip Connections
        # 使用 bilinear 插值對齊 2D 尺寸
        d5 = F.interpolate(b, size=(e5.shape[2], e5.shape[3]), mode='bilinear', align_corners=False)
        d5 = self.dec5(torch.cat([d5, e5], dim=1))

        d4 = F.interpolate(d5, size=(e4.shape[2], e4.shape[3]), mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False)
        out = self.final(torch.cat([d1, e1], dim=1))

        return self.relu(out)