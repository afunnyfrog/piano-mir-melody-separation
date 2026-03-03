import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class AudioUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_fft=2048, hop_length=512):
        super(AudioUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer('window', torch.hann_window(n_fft))

        # --- Encoder (Downscaling) ---
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = DoubleConv(16, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128, dropout_rate=0.5)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bot = DoubleConv(128, 256, dropout_rate=0.5)

        # --- Decoder (Upscaling) ---
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)

        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, waveform, midi_hints):
        # 1. GPU STFT 轉換
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=self.window, center=True, return_complex=True)
        magnitude = torch.abs(stft)
        log_spec = torch.log(magnitude + 1e-6)
        
        # 2. 裁切與維度對齊 (1024 bins)
        x_spec = log_spec[:, :1024, :].unsqueeze(1) # (B, 1, 1024, T)
        
        if x_spec.shape[3] != midi_hints.shape[3]:
            min_time = min(x_spec.shape[3], midi_hints.shape[3])
            x_spec = x_spec[:, :, :, :min_time]
            midi_hints = midi_hints[:, :, :, :min_time]
            
        x = torch.cat([x_spec, midi_hints], dim=1) # (B, 3, 1024, T)

        # 3. ✨ [關鍵修正]：自動對齊 16 的倍數 (Padding)
        # U-Net 有 4 層下採樣，所以寬度與高度必須是 16 的倍數
        # 這裡的 Height=1024 已經是 16 倍數，主要處理 Time (Width)
        B, C, H, T = x.shape
        pad_t = (16 - (T % 16)) % 16
        if pad_t > 0:
            x = F.pad(x, (0, pad_t)) # 僅在右側補齊時間幀

        # --- Encoder ---
        x1 = self.inc(x)            
        p1 = self.pool(x1)          
        x2 = self.down1(p1)         
        p2 = self.pool(x2)          
        x3 = self.down2(p2)         
        p3 = self.pool(x3)          
        x4 = self.down3(p3)         
        p4 = self.pool(x4)          
        
        # --- Bottleneck ---
        x5 = self.bot(p4)           
        
        # --- Decoder ---
        u1 = self.up1(x5)           
        u1 = torch.cat([u1, x4], dim=1) 
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
        
        logits = self.outc(u4) 
        masks = torch.sigmoid(logits)
        
        # 4. ✨ [還原]：切掉 Padding 的部分
        if pad_t > 0:
            masks = masks[:, :, :, :T]
        
        return masks
