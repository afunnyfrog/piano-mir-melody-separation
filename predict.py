import torch
import librosa
import numpy as np
import soundfile as sf
import os
from utils.u_net import AudioUNet

# ==========================================
#               參數設定
# ==========================================
# 1. 路徑設定
MODEL_PATH = "./checkpoints_multi_task/best_model.pth"
INPUT_AUDIO = r"G:\project_data\two_line_midi\flac_output\mix_audio_flac\Classical_Classical_Wolfgang Amadeus Mozart_Andante_mixed.flac"

OUTPUT_DIR = "./results_no_hint"

# 2. 推論設定
TEST_DURATION = 60.0  # 測試音訊長度 (秒)，設為 None 則處理整首
USE_DEVICE = "cuda"   # "cuda" 或 "cpu"

# 3. 音訊處理參數 (必須與訓練時一致)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TARGET_BINS = 1024

# 4. 快取設定 (解決權限警告)
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
# ==========================================

def main():
    # 確保快取與輸出目錄存在
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(USE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 載入模型架構 (3 輸入通道, 4 輸出通道)
    print("正在初始化模型架構 (3 in, 4 out)...")
    model = AudioUNet(n_channels=3, n_classes=4).to(device)
    
    # 2. 載入權重 (處理字典格式與 weights_only 警告)
    if os.path.exists(MODEL_PATH):
        print(f"🔄 正在載入權重: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 權重載入成功！")
    else:
        print(f"❌ 找不到權重檔: {MODEL_PATH}")
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

    # 5. 推論
    print("正在進行分離運算 (無提示模式)...")
    with torch.no_grad():
        # 模型內部會自動處理 STFT 與 Padding
        preds = model(waveform_tensor, midi_hints)
        
    # preds shape: (1, 4, 1024, Time)
    # 通道索引: 0=Melody 音訊, 1=Accomp 音訊
    preds = preds.cpu().numpy()[0] 
    mask_mel = preds[0]
    mask_acc = preds[1]

    # 6. 還原音訊
    print("正在還原音訊並儲存檔案...")
    stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft_full)
    phase = np.exp(1.j * np.angle(stft_full))
    
    # 對齊長度 (確保輸出遮罩與原始頻譜時間軸一致)
    T_min = min(magnitude.shape[1], mask_mel.shape[1])
    magnitude = magnitude[:, :T_min]
    phase = phase[:, :T_min]

    for mask, name in [(mask_mel, 'melody'), (mask_acc, 'accompaniment')]:
        # 取得對應長度的遮罩並補回第 1025 點
        current_mask = mask[:, :T_min]
        full_mask = np.vstack([current_mask, np.zeros((1, T_min))])
        
        # 應用遮罩並執行反向 STFT
        sep_mag = magnitude * full_mask
        y_recon = librosa.istft(sep_mag * phase, hop_length=HOP_LENGTH)
        
        # 音量最大化 (避免聲音太小)
        max_amp = np.max(np.abs(y_recon))
        if max_amp > 1e-6:
            y_recon = y_recon * (0.9 / max_amp)
        
        save_path = os.path.join(OUTPUT_DIR, f"{name}_no_hint.wav")
        sf.write(save_path, y_recon, SAMPLE_RATE)
        print(f"💾 已儲存: {save_path}")

    print("\n🎉 處理完成！請至結果資料夾檢查輸出。")

if __name__ == "__main__":
    main()
