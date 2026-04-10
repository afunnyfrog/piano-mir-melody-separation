import sys
import os
import pandas as pd
import torch
import librosa
import numpy as np
import soundfile as sf
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# 1. 環境配置
# ==========================================
PROJECT_ROOT = r"F:\project\piano-mir-melody-separation"
sys.path.append(os.path.join(PROJECT_ROOT, "separation_mel_"))
sys.path.append(os.path.join(PROJECT_ROOT, "separation_accomp_"))

from separation_mel_.utils_melody.u_net import AudioUNet as MelodyUNet
from separation_accomp_.utils_accomp.u_net import AudioUNet as AccompUNet

CSV_FILE = os.path.join(PROJECT_ROOT, "classical_dataset.csv")
MEL_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints_mel_task/best_model.pth")
ACC_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints_acc_task/best_model.pth")

MEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_mel_task")
ACC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_accomp")

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TEST_DURATION = 60.0 # 測試集批量處理建議先跑 60 秒

# ==========================================

def load_model(path, device, model_class):
    if not os.path.exists(path): return None
    model = model_class(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動批量測試 | 裝置: {device}")

    df = pd.read_csv(CSV_FILE)
    test_df = df[df['split_type'] == 'test']
    print(f"📋 準備處理 {len(test_df)} 首修正路徑後的測試歌曲。")

    model_mel = load_model(MEL_MODEL_PATH, device, MelodyUNet)
    model_acc = load_model(ACC_MODEL_PATH, device, AccompUNet)

    os.makedirs(MEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ACC_OUTPUT_DIR, exist_ok=True)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="批量推論中"):
        input_audio = row['file_path'] 
        song_id = row['song_id']
        
        if not os.path.exists(input_audio):
            print(f"⚠️ 仍然找不到檔案: {input_audio}")
            continue 

        safe_name = str(song_id).replace(" ", "_").replace(",", "")
        y, _ = librosa.load(input_audio, sr=SAMPLE_RATE, duration=TEST_DURATION)
        waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
        
        with torch.no_grad():
            spec_mel = model_mel(waveform_tensor).cpu().numpy()[0, 0]
            spec_acc = model_acc(waveform_tensor).cpu().numpy()[0, 0]

        stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.exp(1.j * np.angle(stft_full))
        
        # 存檔邏輯
        for spec, out_dir, prefix in [(spec_mel, MEL_OUTPUT_DIR, "melodyResult"), (spec_acc, ACC_OUTPUT_DIR, "accompResult")]:
            T_min = min(phase.shape[1], spec.shape[1])
            mag = np.exp((spec[:, :T_min] * 10.0) - 10.0)
            full_mag = np.vstack([mag, np.zeros((1, T_min))])
            y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
            save_path = os.path.join(out_dir, f"{prefix}_test_{safe_name}.wav")
            sf.write(save_path, y_recon * (0.9 / (np.max(np.abs(y_recon)) + 1e-8)), SAMPLE_RATE)

    print(f"\n🎉 批量測試完成！結果已儲存。")

if __name__ == "__main__":
    main()