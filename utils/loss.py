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
                 alpha_similarity=2.0, # 旋律/伴奏互斥相似度權重 (處罰伴奏中的旋律殘留)
                 melody_weight=1.5,   # 旋律通道權重
                 accomp_weight=1.0):  # 伴奏通道權重
        super().__init__()
        
        self.alpha_l1 = alpha_l1
        self.alpha_spectral = alpha_spectral
        self.alpha_sisdr = alpha_sisdr
        self.alpha_similarity = alpha_similarity
        self.melody_weight = melody_weight
        self.accomp_weight = accomp_weight
        
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target, pred_audio=None, target_audio=None, pred_other=None):
        """
        Args:
            pred: [B, 1, F, T] 預測的旋律頻譜圖 (Melody)
            target: [B, 2, F, T] 目標頻譜圖 (0: 旋律, 1: 伴奏)
            pred_audio: [B, 1, L] 預測的旋律時域音訊
            target_audio: [B, 2, L] 目標音訊 (0: 旋律, 1: 伴奏)
            pred_other: [B, 1, F, T] 預測的伴奏頻譜圖 (被扣除的部分)
        """
        
        # --- 通道提取 ---
        pred_mel = pred[:, 0, :, :]
        target_mel = target[:, 0, :, :] 
        target_acc = target[:, 1, :, :] 
        
        # 1. 旋律基礎擬合 Loss
        l1_loss_mel = self.l1_loss(pred_mel, target_mel) * self.melody_weight
        spec_loss_mel = self.multi_scale_spectral_loss(pred_mel, target_mel) * self.melody_weight
        
        # 2. 伴奏參考約束 (確保扣掉的東西確實像伴奏)
        if pred_other is not None:
            l1_loss_acc = self.l1_loss(pred_other[:, 0], target_acc) * self.accomp_weight
        else:
            l1_loss_acc = torch.tensor(0.0, device=pred.device)
            
        # 3. 伴奏-旋律相似度懲罰 (Similarity Penalty)
        # 我們希望預測的旋律 (pred_mel) 與真實的伴奏 (target_acc) 越不相似越好
        p_mel_flat = pred_mel.reshape(pred_mel.size(0), -1)
        t_acc_flat = target_acc.reshape(target_acc.size(0), -1)
        
        # 使用餘弦相似度，只處罰正相關 (即旋律裡有伴奏殘留)
        sim = F.cosine_similarity(p_mel_flat, t_acc_flat, dim=1).mean()
        sim_loss = torch.clamp(sim, min=0.0) * self.alpha_similarity

        # 4. SI-SDR Loss (針對旋律)
        if pred_audio is not None and target_audio is not None:
            sisdr_loss = self.si_sdr_loss(pred_audio[:, 0], target_audio[:, 0]) * self.melody_weight
        else:
            sisdr_loss = torch.tensor(0.0, device=pred.device)
        
        # 總 Loss 計算
        total_loss = (self.alpha_l1 * l1_loss_mel + 
                     self.alpha_spectral * spec_loss_mel + 
                     self.alpha_sisdr * sisdr_loss +
                     l1_loss_acc +
                     sim_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'l1_mel': l1_loss_mel.item(),
            'acc_ref': l1_loss_acc.item(),
            'sim_penalty': sim_loss.item(),
            'sisdr': sisdr_loss.item()
        }
        
        return total_loss, loss_dict
    
    def multi_scale_spectral_loss(self, pred, target):
        loss = 0.0
        loss += F.l1_loss(pred, target)
        
        pred_2x = F.avg_pool2d(pred.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        target_2x = F.avg_pool2d(target.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss += F.l1_loss(pred_2x, target_2x)
        
        pred_4x = F.avg_pool2d(pred.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
        target_4x = F.avg_pool2d(target.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
        loss += F.l1_loss(pred_4x, target_4x)
        
        return loss / 3.0
    
    def si_sdr_loss(self, pred, target, eps=1e-8):
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
        
        sig_p = torch.clamp(signal_power, min=eps, max=1e10)
        noi_p = torch.clamp(noise_power, min=eps, max=1e10)
        
        snr = 10 * (torch.log10(sig_p) - torch.log10(noi_p))
        return -torch.clamp(snr, min=-50, max=50).mean()
