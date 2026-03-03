import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSeparationLoss(nn.Module):
    """
    專為音源分離設計的複合 Loss 函數
    結合時域、頻域、感知域的多重約束
    """
    def __init__(self, 
                 alpha_l1=1.0,        # L1 Loss 權重
                 alpha_spectral=2.0,  # Multi-Scale Spectral Loss 權重
                 alpha_sisdr=0.5,     # SI-SDR Loss 權重
                 melody_weight=1.5,   # 旋律通道權重
                 accomp_weight=1.0):  # 伴奏通道權重
        super().__init__()
        
        self.alpha_l1 = alpha_l1
        self.alpha_spectral = alpha_spectral
        self.alpha_sisdr = alpha_sisdr
        self.melody_weight = melody_weight
        self.accomp_weight = accomp_weight
        
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target, pred_audio=None, target_audio=None):
        """
        Args:
            pred: [B, C, F, T] 預測的頻譜圖 (C=1: 僅伴奏, C=2: 旋律+伴奏)
            target: [B, 4, F, T] 目標頻譜圖 (0=Melody, 1=Accomp)
            pred_audio: [B, C, L] 預測的時域音訊 (可選)
            target_audio: [B, 2, L] 目標時域音訊 (可選)
        """
        
        num_channels = pred.shape[1]
        
        if num_channels == 1:
            # --- 僅伴奏模式 ---
            pred_accomp = pred[:, 0, :, :]
            target_accomp = target[:, 1, :, :] # 索引 1 是伴奏
            
            # 1. L1 Loss
            l1_loss = self.l1_loss(pred_accomp, target_accomp) * self.accomp_weight
            
            # 2. Spectral Loss
            spectral_loss = self.multi_scale_spectral_loss(pred_accomp, target_accomp) * self.accomp_weight
            
            # 3. SI-SDR Loss
            if pred_audio is not None and target_audio is not None:
                sisdr_loss = self.si_sdr_loss(pred_audio[:, 0], target_audio[:, 1]) * self.accomp_weight
            else:
                sisdr_loss = torch.tensor(0.0, device=pred.device)
            
            loss_dict = {
                'total': 0.0,
                'l1_accomp': l1_loss.item(),
                'spectral_accomp': spectral_loss.item(),
                'si_sdr_accomp': sisdr_loss.item()
            }
        else:
            # --- 旋律 + 伴奏模式 (原本邏輯) ---
            pred_melody = pred[:, 0, :, :]
            pred_accomp = pred[:, 1, :, :]
            target_melody = target[:, 0, :, :]
            target_accomp = target[:, 1, :, :]
            
            l1_melody = self.l1_loss(pred_melody, target_melody)
            l1_accomp = self.l1_loss(pred_accomp, target_accomp)
            l1_loss = self.melody_weight * l1_melody + self.accomp_weight * l1_accomp
            
            spectral_loss_melody = self.multi_scale_spectral_loss(pred_melody, target_melody)
            spectral_loss_accomp = self.multi_scale_spectral_loss(pred_accomp, target_accomp)
            spectral_loss = (self.melody_weight * spectral_loss_melody + 
                            self.accomp_weight * spectral_loss_accomp)
            
            if pred_audio is not None and target_audio is not None:
                sisdr_melody = self.si_sdr_loss(pred_audio[:, 0], target_audio[:, 0])
                sisdr_accomp = self.si_sdr_loss(pred_audio[:, 1], target_audio[:, 1])
                sisdr_loss = self.melody_weight * sisdr_melody + self.accomp_weight * sisdr_accomp
            else:
                sisdr_loss = torch.tensor(0.0, device=pred.device)
                
            loss_dict = {
                'total': 0.0,
                'l1_melody': l1_melody.item(),
                'l1_accomp': l1_accomp.item(),
                'spectral': spectral_loss.item(),
                'si_sdr': sisdr_loss.item() if isinstance(sisdr_loss, torch.Tensor) else 0.0
            }
        
        # 總 Loss
        total_loss = (self.alpha_l1 * l1_loss + 
                     self.alpha_spectral * spectral_loss + 
                     self.alpha_sisdr * sisdr_loss)
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def multi_scale_spectral_loss(self, pred, target):
        """
        Multi-Scale Spectral Loss
        在不同頻率解析度下計算 L1 距離,捕捉不同尺度的頻譜特徵
        
        靈感來源: MelGAN, HiFi-GAN 等語音合成模型
        """
        loss = 0.0
        
        # 原始尺度
        loss += F.l1_loss(pred, target)
        
        # 下採樣 2x (關注低頻)
        pred_2x = F.avg_pool2d(pred.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        target_2x = F.avg_pool2d(target.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss += F.l1_loss(pred_2x, target_2x)
        
        # 下採樣 4x (關注更低頻)
        pred_4x = F.avg_pool2d(pred.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
        target_4x = F.avg_pool2d(target.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
        loss += F.l1_loss(pred_4x, target_4x)
        
        return loss / 3.0  # 平均三個尺度
    
    def si_sdr_loss(self, pred, target, eps=1e-8):
        """
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss
        
        音源分離領域的黃金標準指標,對音量不敏感
        
        公式:
        SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        其中 s_target = <pred, target> / ||target||^2 * target
             e_noise = pred - s_target
        
        參考論文: "SDR - Half-baked or Well Done?" (ICASSP 2019)
        """
        # 確保輸入是 2D [B, L]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        # Zero-mean normalization (重要!)
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        # 計算投影係數 alpha = <pred, target> / ||target||^2
        dot_product = (pred * target).sum(dim=1, keepdim=True)
        target_energy = (target ** 2).sum(dim=1, keepdim=True) + eps
        alpha = dot_product / target_energy
        
        # 投影信號 s_target
        s_target = alpha * target
        
        # 殘差信號 e_noise
        e_noise = pred - s_target
        
        # 計算 SI-SDR (單位: dB)
        signal_power = (s_target ** 2).sum(dim=1) + eps
        noise_power = (e_noise ** 2).sum(dim=1) + eps
        si_sdr = 10 * torch.log10(signal_power / noise_power)
        
        # 返回負值作為 Loss (要最大化 SI-SDR)
        return -si_sdr.mean()


class SpectrogramReconstructor(nn.Module):
    """
    輔助類別: 將預測的頻譜圖轉回時域音訊
    用於計算 SI-SDR Loss
    """
    def __init__(self, n_fft=2048, hop_length=512, win_length=2048):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def forward(self, log_spec):
        """
        Args:
            log_spec: [B, 2, F, T] Log magnitude spectrogram
        
        Returns:
            audio: [B, 2, L] 時域音訊
        """
        # 將 log 轉回線性
        mag_spec = torch.exp(log_spec)
        
        # Griffin-Lim 相位重建 (簡化版本)
        # 注意: 實際應用建議預測複數頻譜或使用 Vocoder
        batch_size = mag_spec.shape[0]
        audio_list = []
        
        for b in range(batch_size):
            channel_audios = []
            for c in range(2):  # 旋律 + 伴奏
                spec = mag_spec[b, c]  # [F, T]
                # 這裡需要使用 Griffin-Lim 或其他相位重建演算法
                # 為簡化起見,這裡返回 None (在實際使用時需要實作)
                audio = None  # 實際應該呼叫 griffin_lim(spec)
                channel_audios.append(audio)
            audio_list.append(torch.stack(channel_audios))
        
        return None  # 實際應返回 torch.stack(audio_list)


# ========================================
# 使用範例與整合到訓練循環
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("音源分離 Loss 函數使用範例")
    print("=" * 60)
    
    # 初始化 Loss 函數
    criterion = AudioSeparationLoss(
        alpha_l1=1.0,
        alpha_spectral=2.0,
        alpha_sisdr=0.5,  # 如果沒有時域音訊可設為 0
        melody_weight=1.5,
        accomp_weight=1.0
    )
    
    # 模擬資料
    batch_size = 4
    freq_bins = 1025  # n_fft//2 + 1
    time_frames = 100
    
    pred = torch.randn(batch_size, 2, freq_bins, time_frames)
    target = torch.randn(batch_size, 2, freq_bins, time_frames)
    
    # 計算 Loss
    total_loss, loss_dict = criterion(pred, target)
    
    print(f"\n總 Loss: {total_loss.item():.4f}")
    print("\n各項 Loss 詳細數值:")
    for key, value in loss_dict.items():
        print(f"  {key:12s}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("整合到訓練循環的程式碼片段:")
    print("=" * 60)
    