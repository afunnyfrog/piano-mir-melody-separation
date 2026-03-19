# analyze_results.py
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import optuna
import random
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources
from utils.u_net import AudioUNet

# ==========================================
#               參數設定
# ==========================================
STUDY_DB = "sqlite:///wage_melody_optimization.db"
STUDY_NAME = "wage_melody_optimization"
OPTUNA_BASE_DIR = "./optuna_wage_studies"
CSV_FILE = "classical_dataset.csv"
DATA_ROOT = r"C:\Users\cebit\Desktop\專題生成\classified_dataset"

# 評估參數
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
TARGET_BINS = 1024
# ==========================================

def get_best_model_info():
    """ 從 Optuna 資料庫讀取最佳試驗資訊 """
    if not os.path.exists("wage_melody_optimization.db"):
        print("❌ 找不到 Optuna 資料庫檔案。")
        return None
    
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB)
    best_trial = study.best_trial
    
    print("\n" + "="*40)
    print("🏆 Optuna 最佳訓練結果摘要")
    print(f"最佳試驗編號: Trial {best_trial.number}")
    print(f"最低驗證損失: {best_trial.value:.4f}")
    print("-" * 20)
    print("最佳參數設定:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
    print("="*40 + "\n")
    
    model_path = os.path.join(OPTUNA_BASE_DIR, f"trial_{best_trial.number}", "best_model.pth")
    return {"path": model_path, "params": best_trial.params}

def evaluate_best_model(model_info, num_samples=1):
    """ 挑選隨機樣本進行詳細的音訊指標評估 (SDR/SI-SDR) """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    model = AudioUNet(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(model_info["path"], map_location=device)
    model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
    model.eval()

    # 讀取資料集並挑選驗證集樣本
    df = pd.read_csv(CSV_FILE)
    val_files = df[df['split_type'] == 'val']['file_path'].tolist()
    test_samples = random.sample(val_files, min(num_samples, len(val_files)))

    results = []
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()

    for mix_path in test_samples:
        print(f"正在評估: {os.path.basename(mix_path)}...")
        
        # 取得 Ground Truth 路徑
        # 假設結構: .../mix_audio_flac/XXX_mixed.flac -> .../melody_audio_flac/XXX_melody.flac
        gt_mel_path = mix_path.replace("mix_audio_flac", "melody_audio_flac").replace("_mixed.flac", "_melody.flac")
        
        if not os.path.exists(gt_mel_path):
            print(f"⚠️ 找不到對應的旋律真值檔: {gt_mel_path}")
            continue

        # 讀取音訊 (前 30 秒)
        y_mix, _ = librosa.load(mix_path, sr=SAMPLE_RATE, duration=30.0)
        y_gt, _ = librosa.load(gt_mel_path, sr=SAMPLE_RATE, duration=30.0)

        # 模型推論
        with torch.no_grad():
            waveform_tensor = torch.from_numpy(y_mix).unsqueeze(0).to(device)
            mapped_spec = model(waveform_tensor).cpu().numpy()[0, 0]

        # 還原波形
        stft_mix = librosa.stft(y_mix, n_fft=N_FFT, hop_length=HOP_LENGTH)
        phase = np.exp(1.j * np.angle(stft_mix))
        T_min = min(phase.shape[1], mapped_spec.shape[1])
        
        # 逆正規化 (與 Dataset 邏輯一致: (norm * 10) - 10)
        mag_recon = np.exp((mapped_spec[:, :T_min] * 10.0) - 10.0)
        full_mag = np.zeros((N_FFT // 2 + 1, T_min))
        full_mag[:TARGET_BINS, :] = mag_recon
        
        y_pred = librosa.istft(full_mag * phase[:, :T_min], hop_length=HOP_LENGTH)
        
        # 對齊長度
        min_len = min(len(y_pred), len(y_gt))
        y_pred, y_gt = y_pred[:min_len], y_gt[:min_len]

        # 計算指標
        # bss_eval_sources 需要 [n_src, n_samples]
        sdr, sir, sar, _ = bss_eval_sources(y_gt[None, :], y_pred[None, :], compute_permutation=False)
        si_sdr = si_sdr_metric(torch.tensor(y_pred), torch.tensor(y_gt)).item()

        results.append({
            "Track": os.path.basename(mix_path),
            "SDR": sdr[0],
            "SI-SDR": si_sdr,
            "SIR": sir[0],
            "SAR": sar[0]
        })

    return pd.DataFrame(results)

def main():
    # 1. 分析 Optuna 訓練結果
    model_info = get_best_model_info()
    if not model_info: return

    # 2. 進行音訊質量評估
    print("🚀 正在使用最佳模型評估隨機驗證集樣本...")
    report_df = evaluate_best_model(model_info, num_samples=3)
    
    print("\n--- 音訊分離質量評估 (平均值) ---")
    print(report_df.mean(numeric_only=True))
    
    # 3. 儲存與繪圖
    report_df.to_csv("best_model_evaluation.csv", index=False)
    print(f"\n✅ 詳細報告已儲存至: best_model_evaluation.csv")

    # 簡單繪圖
    if not report_df.empty:
        report_df.set_index("Track")[["SDR", "SI-SDR"]].plot(kind="bar", figsize=(10, 5))
        plt.title("Best Model Performance on Validation Samples")
        plt.ylabel("dB")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("evaluation_results.png")
        print("📊 評估圖表已儲存至: evaluation_results.png")

if __name__ == "__main__":
    main()
