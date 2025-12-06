import torch
import librosa
import numpy as np
import soundfile as sf
import os
import argparse
from utils.u_net import AudioUNet

# --- 設定 ---
# 這裡填寫你訓練好的權重檔案路徑
MODEL_PATH = "./checkpoints/model_epoch_50.pth" 
# 設定你要測試的歌曲路徑
INPUT_AUDIO = r"C:\Users\richa\Documents\專題數據\maestro-v3.0.0\maestro-v3.0.0\2017\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.wav"
# 輸出的資料夾
OUTPUT_DIR = "./results"

SAMPLE_RATE = 44100

def main():
    # 1. 準備裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 載入模型架構
    print("正在載入模型...")
    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    
    # 3. 載入權重 (Load Weights)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"成功載入權重: {MODEL_PATH}")
    else:
        print(f"找不到權重檔: {MODEL_PATH}")
        return
    
    model.eval() # 設定為推論模式 (這很重要！會關閉 Dropout 和 BN 的訓練行為)

    # 4. 讀取音訊
    print(f"正在讀取音訊: {INPUT_AUDIO} ...")
    # 為了避免記憶體爆掉，建議先用 30 秒片段測試，若顯存夠大可讀整首
    # duration=30 代表只讀前 30 秒，想讀整首就把 duration 拿掉
    y, sr = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE, duration=30.0) 
    
    # 5. 轉頻譜圖 (STFT)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.exp(1.j * np.angle(stft)) # 保存相位資訊 (重建聲音需要)
    
    # Log Scale & Normalize
    log_spec = np.log(magnitude + 1e-6)
    
    # 紀錄原始的最大最小值，以便還原
    min_val = log_spec.min()
    max_val = log_spec.max()
    norm_spec = (log_spec - min_val) / (max_val - min_val + 1e-6)

    # --- 關鍵：維度處理 ---
    # 訓練時我們丟掉了第 1025 點，這裡也要丟掉
    input_spec = norm_spec[:1024, :]
    
    # 檢查時間軸長度，必須是 16 的倍數 (因為 U-Net 下採樣 4 次)
    time_steps = input_spec.shape[1]
    pad_len = 0
    if time_steps % 16 != 0:
        pad_len = 16 - (time_steps % 16)
        input_spec = np.pad(input_spec, ((0,0), (0, pad_len)))
    
    # 轉成 Tensor: (1, 1, 1024, Time)
    input_tensor = torch.tensor(input_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 6. 推論 (Inference)
    print("正在進行分離運算...")
    with torch.no_grad(): # 不計算梯度，省記憶體
        pred_masks = model(input_tensor)
        
    # pred shape: (1, 2, 1024, Time) -> 2 個 Channel (Melody, Accomp)
    pred_masks = pred_masks.cpu().numpy()[0] # 取出第一筆 batch -> (2, 1024, Time)

    # 7. 後處理 (還原成聲音)
    print("正在應用遮罩還原音訊...")
    # --- [進階技巧]：互斥鎖定 (Sum Constraint) ---
    # 強迫 mask_mel + mask_acc = 1
    # 這能讓分離更乾淨，避免兩個都搶著要同一個聲音    
    mask_mel = pred_masks[0]
    mask_acc = pred_masks[1]
    print("-" * 30)
    print(f"Melody Mask -> Mean: {mask_mel.mean():.4f}, Max: {mask_mel.max():.4f}, Min: {mask_mel.min():.4f}")
    print(f"Accomp Mask -> Mean: {mask_acc.mean():.4f}, Max: {mask_acc.max():.4f}, Min: {mask_acc.min():.4f}")
    print("-" * 30)
    
    final_masks = [mask_mel, mask_acc]
    
    # 分別處理 Melody (Channel 0) 和 Accomp (Channel 1)
    for i, name in enumerate(['melody', 'accompaniment']):
        # 取出頻譜
        mask = final_masks[i]
        
        # 去除剛剛補的 Padding
        if pad_len > 0:
            mask = mask[:, :-pad_len]
            
        
        # 補回第 1025 個頻率點 (用 0 補，或是複製第 1024 點)
        # 我們需要補一行讓它變回 1025
        mask = np.vstack([mask, np.zeros((1, mask.shape[1]))])
        # D. [核心步驟] 應用遮罩
        # 分離後的能量 = 原始能量 (Magnitude) * 遮罩 (Mask)
        # 這樣做音質最好，因為我們直接操作原始訊號的能量
        sep_magnitude = magnitude * mask
        
        # 結合原始相位 (使用 Phase Reconstruction)
        # 這是一種簡單的做法，假設分離後的相位跟原曲一樣
        y_recon = librosa.istft(sep_magnitude * phase, hop_length=512)
        
        # --- [自動音量最大化] ---
        # 1. 找出目前的音量最大值
        max_amp = np.max(np.abs(y_recon))
    
        # 2. 如果聲音太小，就放大它
        if max_amp > 0:
            # 將最大值拉到 0.9 (保留一點點空間避免破音 clipping)
            scale_factor = 0.9 / max_amp
            y_recon = y_recon * scale_factor
            print(f"  -> 已自動放大音量 (放大倍率: {scale_factor:.2f}x)")
        
        # 存檔
        save_name = os.path.join(OUTPUT_DIR, f"result_{name}.wav")
        sf.write(save_name, y_recon, SAMPLE_RATE)
        print(f"已儲存: {save_name}")

    print("完成！請至 results 資料夾試聽。🎧")

if __name__ == "__main__":
    main()