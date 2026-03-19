import torch
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import optuna
from utils.u_net import AudioUNet

# ==========================================
#               參數設定
# ==========================================
# 1. 模型與路徑設定
# 如果想指定特定 Trial，請填寫數字（例如 5）；若填 None 則自動尋找最佳 Trial
TRIAL_NUMBER = None 
STUDY_DB = "sqlite:///wage_melody_optimization.db"
STUDY_NAME = "wage_melody_optimization"
OPTUNA_BASE_DIR = "./optuna_wage_studies"

# 備用路徑 (如果沒找到 Optuna 記錄則使用此路徑)
FALLBACK_MODEL_PATH = "./checkpoints_mel_task/best_model.pth"

INPUT_AUDIO = r"C:\Users\cebit\Desktop\專題生成\classified_dataset\mix_audio_flac\Soundtracks_Video Game Music_Mario Kart_Mario_Kart_Character_Select_mixed.flac"
OUTPUT_DIR = "./results_mel_task"
OUTPUT_FILENAME = "melody_extracted.wav" # 模型輸出的是旋律
SPEC_FILENAME = "spectrogram_comparison.png"

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

# 5. 快取設定
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
# ==========================================

def get_best_model_path():
    """ 從 Optuna 資料庫自動尋找最佳模型的路徑 """
    if TRIAL_NUMBER is not None:
        path = os.path.join(OPTUNA_BASE_DIR, f"trial_{TRIAL_NUMBER}", "best_model.pth")
        if os.path.exists(path): return path
        print(f"⚠️ 找不到指定的 Trial {TRIAL_NUMBER}，切換至自動搜尋模式。")

    if os.path.exists("wage_melody_optimization.db"):
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
            best_trial = study.best_trial
            path = os.path.join(OPTUNA_BASE_DIR, f"trial_{best_trial.number}", "best_model.pth")
            if os.path.exists(path):
                print(f"🎯 自動偵測到最佳試驗: Trial {best_trial.number} (Loss: {best_trial.value:.4f})")
                return path
        except Exception as e:
            print(f"⚠️ 無法讀取 Optuna 資料庫: {e}")

    print(f"ℹ️ 使用備用模型路徑: {FALLBACK_MODEL_PATH}")
    return FALLBACK_MODEL_PATH

def plot_spectrogram_comparison(original_mag, reconstructed_mag, save_path):
    """ 繪製原始與還原頻譜的對比圖 """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(original_mag, ref=np.max), 
                             sr=SAMPLE_RATE, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title('Original Spectrogram (Mix)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(reconstructed_mag, ref=np.max), 
                             sr=SAMPLE_RATE, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    plt.title('Extracted Melody Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 已儲存頻譜對比圖: {save_path}")

def main():
    os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(USE_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 1. 取得模型路徑並載入
    model_path = get_best_model_path()
    model = AudioUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    
    if os.path.exists(model_path):
        print(f"🔄 正在載入權重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 權重載入成功！")
    else:
        print(f"❌ 找不到權重檔: {model_path}")
        return
    
    model.eval()

    # 2. 讀取音訊
    if not os.path.exists(INPUT_AUDIO):
        print(f"❌ 找不到輸入音訊: {INPUT_AUDIO}")
        return
    print(f"正在讀取音訊: {INPUT_AUDIO} ...")
    y, sr = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE, duration=TEST_DURATION)
    
    # 3. 推論
    print("正在進行分離運算...")
    waveform_tensor = torch.from_numpy(y).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 模型輸出: [1, 1, 1024, T]
        mapped_spec = model(waveform_tensor).cpu().numpy()[0, 0]
        
    print(f"DEBUG: 模型輸出範圍 [{mapped_spec.min():.4f}, {mapped_spec.max():.4f}]")

    # 4. 還原音訊
    print("正在還原音訊...")
    stft_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.exp(1.j * np.angle(stft_full))
    T_min = min(phase.shape[1], mapped_spec.shape[1])
    
    # 逆正規化: (norm * 10) - 10
    log_recon = (mapped_spec[:, :T_min] * 10.0) - 10.0
    mag_recon = np.exp(log_recon)
    
    # 補回剩餘頻段並結合原始相位
    full_mag = np.zeros((N_FFT // 2 + 1, T_min))
    full_mag[:TARGET_BINS, :] = mag_recon
    y_recon = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
    
    # 音量標準化
    max_amp = np.max(np.abs(y_recon))
    if max_amp > 1e-7:
        y_recon = y_recon * (0.9 / max_amp)
    
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    sf.write(save_path, y_recon, SAMPLE_RATE)
    print(f"💾 已儲存結果: {save_path}")

    # 5. 繪製對比圖
    spec_save_path = os.path.join(OUTPUT_DIR, SPEC_FILENAME)
    plot_spectrogram_comparison(np.abs(stft_full[:, :T_min]), full_mag, spec_save_path)

    print("\n🎉 處理完成！")

if __name__ == "__main__":
    main()
