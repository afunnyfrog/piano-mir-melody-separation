import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm

# 引入自定義模組
from utils.dataset import WeightedAudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# --- 參數設定 ---
DATA_DIR = r"C:\Users\cebit\Desktop\專題生成\classified_dataset"
CHECKPOINT_DIR = "./checkpoints_dynamic_8s"
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
SEGMENT_SECONDS = 2.0
SAMPLES_PER_EPOCH = 600
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
RESUME_FROM = os.path.join(CHECKPOINT_DIR, "best_model.pth")
RUN_EPOCHS = 50



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    start_epoch, best_val_loss = 0, float('inf')

    # 1. 斷點續訓
    if os.path.exists(RESUME_FROM):
        checkpoint = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # 2. 初始化 Loss (確保參數名與你最新的 loss.py 一致)
    criterion = AudioSeparationLoss(
        alpha_leakage=2.0,
        alpha_wav=12.0,
        alpha_attack=6.0,
        melody_weight=10.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    window = torch.hann_window(WIN_LENGTH).to(device)

    # 3. 資料載入
    train_dataset = WeightedAudioDataset(root_dir=DATA_DIR, split="train", segment_seconds=SEGMENT_SECONDS,
                                         samples_per_epoch=SAMPLES_PER_EPOCH)
    val_dataset = WeightedAudioDataset(root_dir=DATA_DIR, split="val", segment_seconds=SEGMENT_SECONDS,
                                       samples_per_epoch=50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 4. 訓練迴圈
    end_epoch = start_epoch + RUN_EPOCHS
    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_loss_accum = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # --- A. 時域 -> 頻域轉換 ---
            stft_mix = torch.stft(data.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                  window=window, return_complex=True)
            mag_mix = torch.abs(stft_mix).unsqueeze(1)
            phase_mix = torch.angle(stft_mix)

            # 目標頻譜
            stft_mel = torch.stft(target[:, 0, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
            stft_acc = torch.stft(target[:, 1, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
            mag_targets = torch.cat([torch.abs(stft_mel).unsqueeze(1), torch.abs(stft_acc).unsqueeze(1)], dim=1)

            # --- B. 模型推論 ---
            optimizer.zero_grad()
            pred_mags = model(mag_mix)

            # --- C. ✨ 關鍵修正：套用遮罩邏輯還原波形 ---
            # 必須先拿到 pred_mags 才能算 mask
            m_mix_flat = mag_mix.squeeze(1)
            mask = torch.clamp(pred_mags[:, 0, :, :] / (m_mix_flat + 1e-8), 0.0, 1.0)

            # 使用定義中的「強制遮罩頻譜」來還原波形，這樣 Wav Loss 才會精準
            masked_stft = (mask * m_mix_flat) * torch.exp(1j * phase_mix)
            pred_wav_mel = torch.istft(masked_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                                       window=window, length=data.shape[-1])

            # --- D. 計算聯合 Loss (傳入 6 個參數) ---
            loss, loss_dict = criterion(
                pred_mags,
                mag_targets,
                pred_wav_mel,
                target[:, 0, :],
                mag_mix,
                target[:, 1, :]  # 傳入伴奏軌用於 Attack 過濾
            )

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            # --- E. 進度條更新 (確保 Key 名稱與 loss.py 對齊) ---
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Wav": f"{loss_dict['wav']:.4f}",
                "Atk": f"{loss_dict['atk']:.5f}",
                "Leak": f"{loss_dict['leak']:.4f}"
            })

        # --- 5. 驗證階段 (同步修正邏輯) ---
        model.eval()
        val_loss_accum = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", leave=False)
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)

                stft_mix_v = torch.stft(data.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
                mag_mix_v = torch.abs(stft_mix_v).unsqueeze(1)
                phase_mix_v = torch.angle(stft_mix_v)

                mag_targets_v = torch.cat([
                    torch.abs(
                        torch.stft(target[:, 0, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)).unsqueeze(
                        1),
                    torch.abs(
                        torch.stft(target[:, 1, :], n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)).unsqueeze(
                        1)
                ], dim=1)

                pred_mags_v = model(mag_mix_v)

                # 驗證時也套用遮罩還原，維持一致性
                m_mix_v_flat = mag_mix_v.squeeze(1)
                mask_v = torch.clamp(pred_mags_v[:, 0, :, :] / (m_mix_v_flat + 1e-8), 0.0, 1.0)
                pred_wav_v = torch.istft((mask_v * m_mix_v_flat) * torch.exp(1j * phase_mix_v),
                                         n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, length=data.shape[-1])

                v_loss, v_loss_dict = criterion(pred_mags_v, mag_targets_v, pred_wav_v, target[:, 0, :], mag_mix_v,
                                                target[:, 1, :])
                val_loss_accum += v_loss.item()
                val_pbar.set_postfix({
                    "v_Loss": f"{v_loss.item():.4f}",
                    "v_Atk": f"{v_loss_dict['atk']:.5f}",
                    "v_Create": f"{v_loss_dict['create']:.5f}"
                })

        avg_val_loss = val_loss_accum / len(val_loader)
        duration = time.time() - start_time
        print(
            f"Epoch {epoch + 1} | Time: {duration:.1f}s | Train: {train_loss_accum / len(train_loader):.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'best_val_loss': best_val_loss},
                       os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("🏆 已更新最佳模型。")

    print(f"🎉 訓練完成。")


if __name__ == "__main__":
    main()