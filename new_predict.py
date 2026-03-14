import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa  # ✨ 關鍵：確保這行存在

# 確保 Python 能找到 utils 資料夾
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.u_net import AudioUNet
    print("✅ 成功載入模型定義")
except ModuleNotFoundError:
    print("❌ 仍找不到 utils.u_net，請檢查資料夾結構是否正確")

# --- 設定 ---
MODEL_PATH = r"C:\Users\cebit\Downloads\piano-mir-melody-separation-main\piano-mir-melody-separation-main\checkpoints_dynamic_8s\best_model.pth"
INPUT_AUDIO = r"C:\Users\cebit\Desktop\專題生成\classified_dataset\balanced\mix_audio_flac\Classical_Classical Era_Muzio Clementi_Sonatina-1-2_mixed.flac"
OUTPUT_DIR = "./results"
SAMPLE_RATE = 44100
SEGMENT_LEN = 44100 * 4  # 必須與訓練時的 8 秒一致


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. 載入模型 (1D 架構)
    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 自動相容新舊存檔格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. 讀取音訊 (不使用 STFT)
    y, _ = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE)

    # 3. 切割成一段一段 8 秒進行推論 (避免記憶體爆掉)
    # 這裡我們用最簡單的填充方式確保長度
    total_samples = len(y)
    pad_len = SEGMENT_LEN - (total_samples % SEGMENT_LEN)
    y_padded = np.pad(y, (0, pad_len))

    mel_chunks = []
    acc_chunks = []

    print("正在分離...")
    with torch.no_grad():
        for i in range(0, len(y_padded), SEGMENT_LEN):
            chunk = y_padded[i:i + SEGMENT_LEN]
            # 轉換為 (1, 1, 352800)
            input_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # 推論
            output = model(input_tensor)  # [1, 2, 352800]

            # 取得結果
            output = output.cpu().numpy()[0]
            mel_chunks.append(output[0])
            acc_chunks.append(output[1])

    # 4. 合併並存檔
    y_mel = np.concatenate(mel_chunks)[:total_samples]
    y_acc = np.concatenate(acc_chunks)[:total_samples]

    # 音量正規化
    y_mel = y_mel / (np.max(np.abs(y_mel)) + 1e-6) * 0.9
    y_acc = y_acc / (np.max(np.abs(y_acc)) + 1e-6) * 0.9

    sf.write(f"{OUTPUT_DIR}/pred_melody.wav", y_mel, SAMPLE_RATE)
    sf.write(f"{OUTPUT_DIR}/pred_accomp.wav", y_acc, SAMPLE_RATE)
    print("分離完成！")


if __name__ == "__main__":
    predict()