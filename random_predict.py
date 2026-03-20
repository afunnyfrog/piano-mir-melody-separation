import sys
import os
import pandas as pd
import torch
import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# 1. 路徑自適應：確保 import 不會 ModuleNotFoundError
# ---------------------------------------------------------
# 將兩個工具包的根目錄都加入 sys.path
PROJECT_ROOT = r"C:\Users\tt\Desktop\project"
sys.path.append(os.path.join(PROJECT_ROOT, "separation_mel_"))
sys.path.append(os.path.join(PROJECT_ROOT, "separation_accomp_"))

# 分別載入各自的類別定義
from separation_mel_.utils_melody.u_net import AudioUNet as MelodyUNet
from separation_accomp_.utils_accomp.u_net import AudioUNet as AccompUNet

# ==========================================
#               參數設定
# ==========================================
CSV_FILE = os.path.join(PROJECT_ROOT, "classical_dataset.csv")
MEL_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints_mel_task/best_model.pth")
ACC_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints_acc_task/best_model.pth")

MEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_mel_task")
ACC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results_accomp")

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TEST_DURATION = 60.0  # 設為 None 處理全曲

def load_model(path, device, model_class):
    print(f"🔄 正在載入權重: {os.path.basename(path)}")
    model = model_class(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_and_save(model, waveform_tensor, original_y, save_path):
    with torch.no_grad():
        mapped_spec = model(waveform_tensor).cpu().numpy()[0, 0]
    
    # 執行你原本最穩定的還原邏輯
    stft_full = librosa.stft(original_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1.j * np.angle(stft_full))
    T_min = min(phase.shape[1], mapped_spec.shape[1])
    
    # 逆正規化 (norm * 10) - 10
    log_recon = (mapped_spec[:, :T_min] * 10.0) - 10.0
    mag_recon = np.exp(log_recon)
    
    full_mag = np.vstack([mag_recon, np.zeros((1, T_min))])
    y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
    
    # 音量最大化 (防爆音)
    max_amp = np.max(np.abs(y_recon))
    if max_amp > 1e-7:
        y_recon = y_recon * (0.9 / max_amp)
        sf.write(save_path, y_recon, SAMPLE_RATE)
        return True
    return False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用裝置: {device}")

    # 1. 隨機選曲
    df = pd.read_csv(CSV_FILE)
    random_row = df.sample(n=1).iloc[0]
    song_id = random_row['song_id']
    split_type = random_row['split_type']
    input_audio = random_row['file_path']
    safe_name = str(song_id).replace(" ", "_").replace(",", "")

    print(f"🎵 挑選歌曲: {song_id} ({split_type})")

    # 2. 載入模型
    model_mel = load_model(MEL_MODEL_PATH, device, MelodyUNet)
    model_acc = load_model(ACC_MODEL_PATH, device, AccompUNet)

    # 3. 讀取音訊
    y, _ = librosa.load(input_audio, sr=SAMPLE_RATE, duration=TEST_DURATION)
    waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)

    # 4. 分別執行預測與存檔 (互不干擾，最穩)
    os.makedirs(MEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ACC_OUTPUT_DIR, exist_ok=True)

    mel_save_path = os.path.join(MEL_OUTPUT_DIR, f"melodyResult_{split_type}_{safe_name}.wav")
    acc_save_path = os.path.join(ACC_OUTPUT_DIR, f"accompResult_{split_type}_{safe_name}.wav")

    print("⏳ 旋律預測中...")
    process_and_save(model_mel, waveform_tensor, y, mel_save_path)
    
    print("⏳ 伴奏預測中...")
    process_and_save(model_acc, waveform_tensor, y, acc_save_path)

    print(f"\n✅ 處理完成！\n🎼 旋律: {os.path.basename(mel_save_path)}\n🎹 伴奏: {os.path.basename(acc_save_path)}")

if __name__ == "__main__":
    main()