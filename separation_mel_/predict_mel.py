import torch
import librosa
import numpy as np
import soundfile as sf
import os
import glob
from utils_melody.u_net import AudioUNet

# ==========================================
#               參數設定
# ==========================================
# 1. 路徑設定
MODEL_PATH = os.path.join("checkpoints_mel_task", "best_model.pth")
# 可以是單一檔案路徑，或是一個包含音訊檔的目錄
INPUT_PATH = "input"
OUTPUT_DIR = os.path.join("results", "mel_task")

# 2. 推論設定
TEST_DURATION = 60.0  # 測試音訊長度 (秒)，設為 None 則處理整首
USE_DEVICE = "cuda"   # "cuda" 或 "cpu"

# 3. 模型設定 (必須與訓練時一致)
N_CHANNELS = 1        # 輸入通道
N_CLASSES = 1         # 輸出通道

# 4. 音訊處理參數 (必須與訓練時一致)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TARGET_BINS = 1024

# 5. 快取設定 (解決權限警告)
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
# ==========================================

def process_file(file_path, model, device):
    """處理單一音訊檔案"""
    file_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    save_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_melody.wav")
    
    print(f"\n[PROCESS] 正在處理: {file_name} ...")
    
    try:
        # 3. 讀取音訊波形
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TEST_DURATION)
        
        # 準備模型輸入: (1, L)
        waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
        
        # 4. 推論
        print("正在進行分離運算...")
        with torch.no_grad():
            # 取得模型生成的頻譜 [1, 1, 1024, T]
            mapped_spec = model(waveform_tensor).cpu().numpy()[0, 0]
            
        # 6. 還原音訊
        print("正在從生成的頻譜還原音訊...")
        stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.exp(1.j * np.angle(stft_full))
        T_min = min(phase.shape[1], mapped_spec.shape[1])
        
        # --- 逆正規化 ---
        current_mapped = mapped_spec[:, :T_min]
        log_recon = (current_mapped * 10.0) - 10.0
        mag_recon = np.exp(log_recon)
        
        full_mag = np.vstack([mag_recon, np.zeros((1, T_min))])
        y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
        
        # 音量最大化 (調降係數至 0.5 避免爆音)
        max_amp = np.max(np.abs(y_recon))
        if max_amp > 1e-7:
            y_recon = y_recon * (0.5 / max_amp)
            sf.write(save_path, y_recon, SAMPLE_RATE)
            print(f"[SAVE] 已儲存生成結果: {save_path}")
        else:
            print(f"[ERROR] 檔案 {file_name} 還原後的波形振幅過小，無法輸出聲音。")
            
    except Exception as e:
        print(f"[ERROR] 處理檔案 {file_name} 時發生錯誤: {e}")

def main():
    # 確保快取與輸出目錄存在
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(USE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 載入模型架構
    print(f"正在初始化模型架構 ({N_CHANNELS} in, {N_CLASSES} out)...")
    model = AudioUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    
    # 2. 載入模型/權重
    if os.path.exists(MODEL_PATH):
        print(f"[UPDATE] 正在載入: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(device)
            print("[INFO] 檢測到完整模型物件，已直接載入。")
        elif isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("[INFO] 檢測到 Checkpoint 字典，已從 'model_state_dict' 載入。")
            else:
                model.load_state_dict(checkpoint)
                print("[INFO] 檢測到 State Dict 字典，已載入權重。")
        else:
            raise TypeError(f"不支援的載入類型: {type(checkpoint)}")
            
        print("[SUCCESS] 載入成功！")
    else:
        print(f"[ERROR] 找不到模型檔案: {MODEL_PATH}")
        return
    
    model.eval()

    # 找出所有要處理的檔案
    if os.path.isdir(INPUT_PATH):
        # 支援多種格式: wav, mp3, flac
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
        print(f"[INFO] 在目錄中找到 {len(audio_files)} 個音訊檔案。")
    elif os.path.isfile(INPUT_PATH):
        audio_files = [INPUT_PATH]
    else:
        print(f"[ERROR] 找不到路徑: {INPUT_PATH}")
        return

    # 批次處理
    for file_path in audio_files:
        process_file(file_path, model, device)

    print("\n[DONE] 全部處理完成！請至結果資料夾檢查輸出。")

if __name__ == "__main__":
    main()