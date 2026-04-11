import torch
import librosa
import numpy as np
import soundfile as sf
import os
import glob
import sys

# 將子目錄加入路徑以便匯入模型
sys.path.append(os.path.join(os.getcwd(), "separation_mel_"))
sys.path.append(os.path.join(os.getcwd(), "separation_accomp_"))

from separation_mel_.utils_melody.u_net import AudioUNet as MelodyUNet
from separation_accomp_.utils_accomp.u_net import AudioUNet as AccompUNet

# ==========================================
#               參數設定
# ==========================================
# 1. 模型路徑
MEL_MODEL_PATH = os.path.join("best_path", "best_mel_model.pth")
ACC_MODEL_PATH = os.path.join("best_path", "best_acc_model.pth")

# 2. 輸入與輸出
# 可以是單一檔案路徑，或是一個包含音訊檔的目錄
INPUT_PATH = "input"
OUTPUT_DIR = os.path.join("results", "combined")

# 3. 推論設定
TEST_DURATION = 60.0  # 測試音訊長度 (秒)，設為 None 則處理整首
USE_DEVICE = "cuda"   # "cuda" 或 "cpu"

# 4. 音訊處理參數 (必須與訓練時一致)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TARGET_BINS = 1024

# 5. 快取設定
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
# ==========================================

def load_model(path, model_class, device):
    """通用的模型載入函式"""
    if not os.path.exists(path):
        print(f"[ERROR] 找不到模型檔案: {path}")
        return None
    
    model = model_class(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    
    model.eval()
    return model

def reconstruct_audio(mapped_spec, phase, original_y):
    """將模型輸出的頻譜還原為波形"""
    T_min = min(phase.shape[1], mapped_spec.shape[1])
    
    # 逆正規化
    current_mapped = mapped_spec[:, :T_min]
    log_recon = (current_mapped * 10.0) - 10.0
    mag_recon = np.exp(log_recon)
    
    # 補回第 1025 點並結合相位
    full_mag = np.vstack([mag_recon, np.zeros((1, T_min))])
    y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
    
    # 音量最大化 (調降係數至 0.5 避免爆音)
    max_amp = np.max(np.abs(y_recon))
    if max_amp > 1e-7:
        y_recon = y_recon * (0.5 / max_amp)
    return y_recon

def process_file(file_path, mel_model, acc_model, device):
    """處理單一音訊檔案，同時分離旋律與伴奏"""
    file_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    
    mel_save_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_melody.wav")
    acc_save_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_accomp.wav")
    
    print(f"\n[PROCESS] 正在處理: {file_name} ...")
    
    try:
        # 1. 讀取音訊
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TEST_DURATION)
        waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
        
        # 取得原始相位
        stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.exp(1.j * np.angle(stft_full))
        
        # 2. 推論
        with torch.no_grad():
            print("  > 正在分離旋律...")
            spec_mel = mel_model(waveform_tensor).cpu().numpy()[0, 0]
            print("  > 正在分離伴奏...")
            spec_acc = acc_model(waveform_tensor).cpu().numpy()[0, 0]
            
        # 3. 還原並儲存
        print("  > 正在還原旋律音訊...")
        y_mel = reconstruct_audio(spec_mel, phase, y)
        sf.write(mel_save_path, y_mel, SAMPLE_RATE)
        
        print("  > 正在還原伴奏音訊...")
        y_acc = reconstruct_audio(spec_acc, phase, y)
        sf.write(acc_save_path, y_acc, SAMPLE_RATE)
        
        print(f"[SUCCESS] 已完成: {file_name}")
            
    except Exception as e:
        print(f"[ERROR] 處理檔案 {file_name} 時發生錯誤: {e}")

def main():
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(USE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 載入兩個模型
    print("正在載入模型...")
    mel_model = load_model(MEL_MODEL_PATH, MelodyUNet, device)
    acc_model = load_model(ACC_MODEL_PATH, AccompUNet, device)
    
    if mel_model is None or acc_model is None:
        print("[ERROR] 模型載入失敗，請檢查路徑。")
        return

    # 2. 找出檔案
    if os.path.isdir(INPUT_PATH):
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
        print(f"[INFO] 在目錄中找到 {len(audio_files)} 個音訊檔案。")
    elif os.path.isfile(INPUT_PATH):
        audio_files = [INPUT_PATH]
    else:
        print(f"[ERROR] 找不到路徑: {INPUT_PATH}")
        return

    # 3. 執行批次處理
    for file_path in audio_files:
        process_file(file_path, mel_model, acc_model, device)

    print("\n[DONE] 全部處理完成！請至結果資料夾檢查輸出。")

if __name__ == "__main__":
    main()
