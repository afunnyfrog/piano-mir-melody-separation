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
            pred: [B, 1, F, T] 預測的伴奏頻譜圖
            target: [B, 4, F, T] 目標頻譜圖 (1 是伴奏)
            pred_audio: [B, 1, L] 預測的伴奏時域音訊
            target_audio: [B, 2, L] 目標音訊 (1 是伴奏)
        """
        
        # --- 僅針對伴奏 (Accompaniment) 進行計算 ---
        pred_acc = pred[:, 0, :, :]
        target_acc = target[:, 1, :, :] # 索引 1 是伴奏頻譜
        
        # 1. L1 Loss
        l1_loss = self.l1_loss(pred_acc, target_acc) * self.accomp_weight
        
        # 2. Spectral Loss
        spec_loss = self.multi_scale_spectral_loss(pred_acc, target_acc) * self.accomp_weight
        
        # 3. SI-SDR Loss
        if pred_audio is not None and target_audio is not None:
            # 預測值索引 0 vs 目標值索引 1 (伴奏音訊)
            sisdr_loss = self.si_sdr_loss(pred_audio[:, 0], target_audio[:, 1]) * self.accomp_weight
        else:
            sisdr_loss = torch.tensor(0.0, device=pred.device)
        
        # 總 Loss 計算
        total_loss = (self.alpha_l1 * l1_loss + 
                     self.alpha_spectral * spec_loss + 
                     self.alpha_sisdr * sisdr_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'spec': spec_loss.item(),
            'sisdr': sisdr_loss.item()
        }
        
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
        更穩定的 SI-SDR Loss 實作
        """
        # 防止輸入含有 NaN 或 Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any() or \
           torch.isnan(target).any() or torch.isinf(target).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if pred.dim() == 1: pred = pred.unsqueeze(0)
        if target.dim() == 1: target = target.unsqueeze(0)
        
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        dot_product = (pred * target).sum(dim=1, keepdim=True)
        target_energy = (target ** 2).sum(dim=1, keepdim=True) + eps
        alpha = dot_product / target_energy
        
        s_target = alpha * target
        e_noise = pred - s_target
        
        signal_power = (s_target ** 2).sum(dim=1)
        noise_power = (e_noise ** 2).sum(dim=1)
        
        # ✨ [穩定性修正]：對功率進行 clamp，防止 log10(0) 或 log10(inf)
        # 1e-10 約對應 -100dB, 1e10 約對應 100dB
        sig_p = torch.clamp(signal_power, min=eps, max=1e10)
        noi_p = torch.clamp(noise_power, min=eps, max=1e10)
        
        snr = 10 * (torch.log10(sig_p) - torch.log10(noi_p))
        return -torch.clamp(snr, min=-50, max=50).mean()


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
    