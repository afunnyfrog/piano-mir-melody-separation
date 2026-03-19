import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

# 引入自定義模組
from utils.dataset import WeightedAudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# --- 固定參數設定 ---
DATA_DIR = r"C:\Users\cebit\Desktop\專題生成\classified_dataset"
BASE_CHECKPOINT_DIR = "./optuna_studies"
SEGMENT_SECONDS = 2.0
SAMPLES_PER_EPOCH = 200
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
RUN_EPOCHS = 150 # 依照用戶需求跑滿 150 個 Epochs

def objective(trial):
    # --- 1. 定義超參數搜尋空間 ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    alpha_leakage = trial.suggest_float("alpha_leakage", 5.0, 20.0)
    alpha_wav = trial.suggest_float("alpha_wav", 50.0, 200.0)
    alpha_attack = trial.suggest_float("alpha_attack", 40.0, 120.0)
    melody_weight = trial.suggest_float("melody_weight", 20.0, 60.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 建立該次 Trial 的專屬目錄
    trial_dir = os.path.join(BASE_CHECKPOINT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # --- 2. 初始化模型與 Loss ---
    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    criterion = AudioSeparationLoss(
        alpha_leakage=alpha_leakage,
        alpha_wav=alpha_wav,
        alpha_attack=alpha_attack,
        melody_weight=melody_weight
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    window = torch.hann_window(WIN_LENGTH).to(device)

    # --- 3. 資料載入 ---
    train_dataset = WeightedAudioDataset(root_dir=DATA_DIR, split="train", segment_seconds=SEGMENT_SECONDS,
                                         samples_per_epoch=SAMPLES_PER_EPOCH)
    val_dataset = WeightedAudioDataset(root_dir=DATA_DIR, split="val", segment_seconds=SEGMENT_SECONDS,
                                       samples_per_epoch=50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float('inf')

    # --- 4. 訓練迴圈 ---
    for epoch in range(RUN_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{RUN_EPOCHS}", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            stft_mix = torch.stft(data.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                  window=window, return_complex=True)
            mag_mix = torch.abs(stft_mix).unsqueeze(1)
            phase_mix = torch.angle(stft_mix)

            stft_mel = torch.stft(target[:, 0, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
            stft_acc = torch.stft(target[:, 1, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
            mag_targets = torch.cat([torch.abs(stft_mel).unsqueeze(1), torch.abs(stft_acc).unsqueeze(1)], dim=1)

            optimizer.zero_grad()
            pred_mags = model(mag_mix)

            m_mix_flat = mag_mix.squeeze(1)
            mask = torch.clamp(pred_mags[:, 0, :, :] / (m_mix_flat + 1e-8), 0.0, 1.0)
            masked_stft = (mask * m_mix_flat) * torch.exp(1j * phase_mix)
            pred_wav_mel = torch.istft(masked_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                       window=window, length=data.shape[-1])

            loss, loss_dict = criterion(pred_mags, mag_targets, pred_wav_mel, target[:, 0, :], mag_mix, target[:, 1, :])
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Wav": f"{loss_dict['wav']:.4f}",
                "Atk": f"{loss_dict['atk']:.5f}",
                "Leak": f"{loss_dict['leak']:.4f}"
            })

        # --- 5. 驗證階段 ---
        model.eval()
        val_loss_accum = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validating Trial {trial.number} Epoch {epoch + 1}", leave=False)
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                stft_mix_v = torch.stft(data.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
                mag_mix_v = torch.abs(stft_mix_v).unsqueeze(1)
                phase_mix_v = torch.angle(stft_mix_v)
                mag_targets_v = torch.cat([
                    torch.abs(torch.stft(target[:, 0, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)).unsqueeze(1),
                    torch.abs(torch.stft(target[:, 1, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)).unsqueeze(1)
                ], dim=1)

                pred_mags_v = model(mag_mix_v)
                m_mix_v_flat = mag_mix_v.squeeze(1)
                mask_v = torch.clamp(pred_mags_v[:, 0, :, :] / (m_mix_v_flat + 1e-8), 0.0, 1.0)
                pred_wav_v = torch.istft((mask_v * m_mix_v_flat) * torch.exp(1j * phase_mix_v),
                                         n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, length=data.shape[-1])

                v_loss, v_loss_dict = criterion(pred_mags_v, mag_targets_v, pred_wav_v, target[:, 0, :], mag_mix_v, target[:, 1, :])
                val_loss_accum += v_loss.item()
                val_pbar.set_postfix({"v_Loss": f"{v_loss.item():.4f}"})

        avg_val_loss = val_loss_accum / len(val_loader)
        print(f"[Trial {trial.number} Epoch {epoch+1}] Train: {train_loss_accum/len(train_loader):.4f} | Val: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pth"))

        # 向 Optuna 回報進度，並檢查是否需要剪枝 (Pruning)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    # 建立 Study，目標是最小化 (minimize) 驗證損失
    # 使用 SQLite 資料庫來保存結果，這樣即使中斷也可以續傳
    study_name = "melody_separation_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 開始優化，n_trials 可以根據您的資源調整
    print("🚀 開始貝氏優化...")
    study.optimize(objective, n_trials=50)

    print("\n\n" + "="*30)
    print("🏆 優化完成！")
    print(f"最佳驗證損失: {study.best_value:.4f}")
    print("最佳參數設定:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*30)
