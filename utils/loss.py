import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSeparationLoss(nn.Module):
    def __init__(self, alpha_leakage=40.0, alpha_wav=100.0, alpha_attack=50.0, melody_weight=40.0):
        super(AudioSeparationLoss, self).__init__()
        self.alpha_leakage = alpha_leakage
        self.alpha_wav = alpha_wav
        self.alpha_attack = alpha_attack # ✨ 新增：Attack 過濾權重
        self.melody_weight = melody_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_mags, target_mags, pred_wav_mel=None, target_wav_mel=None, mag_mix=None, target_wav_acc=None):
        device = pred_mags.device
        # 取得混音頻譜 (B, F, T)
        m_mix = mag_mix.squeeze(1) if mag_mix.dim() == 4 else mag_mix
        eps = 1e-8

        # --- ✨ 關鍵定義：強制遮罩化 (只能從現有聲音切割) ---
        raw_pred_mel = pred_mags[:, 0, :, :]
        mask_mel = torch.clamp(raw_pred_mel / (m_mix + eps), 0.0, 1.0)
        pm_mel = mask_mel * m_mix # 重新定義後的預測頻譜

        tm_mel = target_mags[:, 0, :, :]
        tm_acc = target_mags[:, 1, :, :]

        # 1. 頻譜 L1 損失
        freq_bins = pm_mel.shape[1]
        freq_weight = torch.linspace(1.0, 5.0, steps=freq_bins, device=device).view(1, -1, 1)
        loss_mel = torch.mean(freq_weight * torch.abs(pm_mel - tm_mel))

        # 2. 波形一致性
        loss_wav = self.l1_loss(pred_wav_mel, target_wav_mel) if pred_wav_mel is not None else torch.tensor(0.0).to(device)

        # 3. ✨ Attack 互斥損失 (捕捉按鍵敲擊)
        loss_attack = torch.tensor(0.0).to(device)
        if pred_wav_mel is not None and target_wav_acc is not None:
            # 計算二階差分 (捕捉加速度突變)
            accel_pred_mel = torch.abs(pred_wav_mel[:, 2:] - 2*pred_wav_mel[:, 1:-1] + pred_wav_mel[:, :-2])
            accel_target_acc = torch.abs(target_wav_acc[:, 2:] - 2*target_wav_acc[:, 1:-1] + target_wav_acc[:, :-2])
            # 當伴奏有強烈 Attack 時，旋律軌若也有變動則處罰
            loss_attack = torch.mean(accel_pred_mel * accel_target_acc)

        # 4. 溢出懲罰 (Leakage)
        loss_leakage = torch.mean(pm_mel * tm_acc)

        # 5. 創造力懲罰 (處罰模型「想加東西」的意圖)
        loss_creation = torch.mean(torch.relu(raw_pred_mel - m_mix))

        # 總和計算
        final_loss = (loss_mel * self.melody_weight) + \
                     (loss_wav * self.alpha_wav) + \
                     (loss_leakage * self.alpha_leakage) + \
                     (loss_attack * self.alpha_attack) + \
                     (loss_creation * 500.0)

        return final_loss, {
            'total': final_loss.item(),
            'wav': loss_wav.item(),
            'leak': loss_leakage.item(),
            'atk': loss_attack.item(),
            'create': loss_creation.item()
        }