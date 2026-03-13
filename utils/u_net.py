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
    def __init__(self, n_channels=1, n_classes=1, n_fft=2048, hop_length=512):
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

    def forward(self, waveform, midi_hints=None, return_audio=False):
        # 1. GPU STFT 轉換
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=self.window, center=True, return_complex=True)
        magnitude = torch.abs(stft)
        log_spec = torch.log(magnitude + 1e-6)
        
        # 2. 裁切與維度對齊 (1024 bins)
        x_raw = log_spec[:, :1024, :].unsqueeze(1) # (B, 1, 1024, T)
        
        # ✨ [關鍵修正]：對輸入也進行正規化，確保輸入範圍在 0~1 之間
        # 必須與 turn_STFT_dataset.py 中的正規化公式一致
        x = (x_raw + 10.0) / 10.0
        
        # 3. ✨ [自動對齊]：自動對齊 16 的倍數 (Padding)
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
        # 譜映射 (Spectral Mapping) 邏輯：使用 ReLU 輸出正向能量強度
        # ✨ [穩定性修正]：限制輸出範圍，防止 exp 溢位導致 NaN
        mapped_spec = torch.clamp(F.relu(logits), min=0.0, max=2.0) 
        
        # 4. ✨ [還原]：切掉 Padding 的部分
        if pad_t > 0:
            mapped_spec = mapped_spec[:, :, :, :T]
            
        # 5. ✨ [SI-SDR 支援]：在映射模式下，直接使用生成的頻譜與原始相位結合
        if return_audio:
            # 取得原始相位 (Phase)
            phase = torch.angle(stft)
            
            # --- 逆正規化：將 0~1 的輸出轉回線性 Magnitude ---
            # 公式必須與 Dataset/Predict 一致: exp((val * 10) - 10)
            log_mapped = (mapped_spec * 10.0) - 10.0
            mag_mapped = torch.exp(log_mapped)
            
            # 將映射出的頻譜補齊 bin 數
            mapped_1025 = F.pad(mag_mapped.squeeze(1), (0, 0, 0, 1))
            
            # 譜映射還原：將生成的 Magnitude 與原始 Phase 結合
            recon_complex = torch.polar(mapped_1025, phase)
            
            pred_audio = torch.istft(recon_complex, n_fft=self.n_fft, hop_length=self.hop_length, 
                                     window=self.window, center=True, length=waveform.shape[-1])
            return mapped_spec, pred_audio.unsqueeze(1)
        
        return mapped_spec
