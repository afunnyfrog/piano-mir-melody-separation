import torch
import librosa
import numpy as np
import soundfile as sf
import os
from utils_melody.u_net import AudioUNet

# ==========================================
#               參數設定
# ==========================================
# 1. 路徑設定
MODEL_PATH = r"F:\project\piano-mir-melody-separation\checkpoints_mel_task\best_model.pth"
INPUT_AUDIO = r"G:\project_data\two_line_midi\flac_output\mix_audio_flac\Classical_Classical_John Philip Sousa_Hands Across the Sea_mixed.flac"
OUTPUT_DIR = "./results_mel_task"
OUTPUT_FILENAME = "melody_no_hint.wav"

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

def main():
    # 確保快取與輸出目錄存在
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(USE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 載入模型架構
    print(f"正在初始化模型架構 ({N_CHANNELS} in, {N_CLASSES} out)...")
    model = AudioUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    
    # 2. 載入模型/權重 (支援完整物件與 state_dict 字典)
    if os.path.exists(MODEL_PATH):
        print(f"[UPDATE] 正在載入: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # 情況 1: 載入的是完整的模型物件 (nn.Module)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(device)
            print("[INFO] 檢測到完整模型物件，已直接載入。")
        
        # 情況 2: 載入的是字典 (可能包含 model_state_dict 或本身就是 state_dict)
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
        print(f"[ERROR] 找不到檔案: {MODEL_PATH}")
        return
    
    model.eval()

    # 3. 讀取音訊波形
    print(f"正在讀取音訊: {INPUT_AUDIO} ...")
    y, sr = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE, duration=TEST_DURATION)
    
    # 準備模型輸入: (1, L)
    waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
    
    # 4. 準備 MIDI Hints (全零)
    # 計算 STFT 後的時間幀數: L // HOP_LENGTH + 1
    num_frames = len(y) // HOP_LENGTH + 1
    midi_hints = torch.zeros((1, 2, TARGET_BINS, num_frames), dtype=torch.float32).to(device)

    # 4. 推論
    print("正在進行分離運算 (譜映射生成模式)...")
    with torch.no_grad():
        # 取得模型生成的頻譜 [1, 1, 1024, T]
        mapped_spec = model(waveform_tensor).cpu().numpy()[0, 0]
        
    print(f"DEBUG: 模型輸出最大值: {mapped_spec.max():.4f}, 最小值: {mapped_spec.min():.4f}")
    if mapped_spec.max() < 1e-3:
        print("[WARN] 警告：模型輸出幾乎全為零，可能需要更多訓練或檢查輸入！")

    # 6. 還原音訊
    print("正在從生成的頻譜還原音訊...")
    stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1.j * np.angle(stft_full))
    T_min = min(phase.shape[1], mapped_spec.shape[1])
    
    # --- 逆正規化 (必須與 Dataset 邏輯一致) ---
    current_mapped = mapped_spec[:, :T_min]
    # 逆向縮放： (norm * 10) - 10
    log_recon = (current_mapped * 10.0) - 10.0
    # 轉回線性 Magnitude
    mag_recon = np.exp(log_recon)
    
    # 補回第 1025 點並結合原始相位
    full_mag = np.vstack([mag_recon, np.zeros((1, T_min))])
    y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
    
    # 音量最大化 (增加防抖動處理)
    max_amp = np.max(np.abs(y_recon))
    if max_amp > 1e-7:
        y_recon = y_recon * (0.9 / max_amp)
    else:
        print("[ERROR] 錯誤：還原後的波形振幅過小，無法輸出聲音。")
        return
    
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    sf.write(save_path, y_recon, SAMPLE_RATE)
    print(f"[SAVE] 已儲存生成結果: {save_path}")

    print("\n[DONE] 處理完成！請至結果資料夾檢查輸出。")

if __name__ == "__main__":
    main()