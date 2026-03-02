# analyze_results.py
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources

pred_melody = r"F:\專題\piano-mir-melody-separation\results\result_melody.wav"
pred_accomp = r"F:\專題\piano-mir-melody-separation\results\result_accompaniment.wav"

gt_melody = r"F:\專題\piano-mir-melody-separation\results\Classical_Classical Era_Muzio Clementi_Sonatina-1_melody.flac"
gt_accomp = r"F:\專題\piano-mir-melody-separation\results\Classical_Classical Era_Muzio Clementi_Sonatina-1_accomp.flac"

REPORT_PATH = "separation_report.csv"

# --- 修改 evaluate_separation 函數 ---
def evaluate_separation(pred_mel_path, pred_acc_path, gt_mel_path, gt_acc_path, sr=44100):
    # 讀取四個檔案
    pm, _ = librosa.load(pred_mel_path, sr=sr)
    pa, _ = librosa.load(pred_acc_path, sr=sr)
    gm, _ = librosa.load(gt_mel_path, sr=sr)
    ga, _ = librosa.load(gt_acc_path, sr=sr)
    
    # 長度對齊
    min_len = min(len(pm), len(pa), len(gm), len(ga))
    preds = np.array([pm[:min_len], pa[:min_len]]) # [2, samples]
    gts = np.array([gm[:min_len], ga[:min_len]])   # [2, samples]
    
    # 同時計算 SDR, SIR, SAR
    # 這樣 mir_eval 才知道誰是誰的干擾
    sdr, sir, sar, _ = bss_eval_sources(gts, preds, compute_permutation=False)
    
    # SI-SDR 還是得個別算 (torchmetrics 通常處理 1D)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    si_sdr_m = si_sdr_metric(torch.tensor(pm[:min_len]), torch.tensor(gm[:min_len])).item()
    si_sdr_a = si_sdr_metric(torch.tensor(pa[:min_len]), torch.tensor(ga[:min_len])).item()
    
    return {
        "Melody": {"SDR": sdr[0], "SIR": sir[0], "SAR": sar[0], "SI-SDR": si_sdr_m},
        "Accomp": {"SDR": sdr[1], "SIR": sir[1], "SAR": sar[1], "SI-SDR": si_sdr_a}
    }

def main():
    # 設定路徑 (請根據實際情況修改)
    # 假設我們比較多組測試結果
    tracks = ["Song1"]
    results = []

    # 模擬測試循環
    for track in tracks:
        print(f"正在分析 {track}...")
        
        # 這裡請替換成你實際的檔案名稱邏輯
        # 例如: pred_melody_Song1.wav vs gt_melody_Song1.wav
        metrics = evaluate_separation(pred_melody, pred_accomp, gt_melody, gt_accomp)
        
        results.append({"Track": track, "Part": "Melody", **metrics["Melody"]})
        results.append({"Track": track, "Part": "Accomp", **metrics["Accomp"]})

    # 轉成 DataFrame
    df = pd.DataFrame(results)
    print("\n--- 評估結果摘要 ---")
    print(df.groupby("Part")[["SDR", "SI-SDR", "SIR", "SAR"]].mean())

    # 5. 視覺化分析
    plot_results(df)

def plot_results(df, save_path="evaluation_chart.png"):
    plt.figure(figsize=(10, 6))
    
    plot_df = df.replace([np.inf, -np.inf], 50.0)
    
    metrics = ["SDR", "SI-SDR", "SIR", "SAR"]
    parts = plot_df['Part'].unique()
    
    # 計算平均值
    avg_results = plot_df.groupby("Part")[metrics].mean()

    x = np.arange(len(metrics))
    width = 0.35

    # 2. 繪圖
    ax = plt.subplot(111)
    if "Melody" in avg_results.index:
        ax.bar(x - width/2, avg_results.loc["Melody"], width, label='Melody', color='skyblue')
    if "Accomp" in avg_results.index:
        ax.bar(x + width/2, avg_results.loc["Accomp"], width, label='Accompaniment', color='salmon')

    plt.ylabel('Decibels (dB)')
    plt.title('Audio Separation Performance Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. 【新增】儲存成圖片
    # dpi=300 可以讓圖片更清晰，適合放進專題報告
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 圖表已儲存至: {save_path}")
    
    # plt.show()

if __name__ == "__main__":
    main()