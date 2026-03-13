# analyze_results.py
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources

# ==========================================
# --- 配置區塊 (在此修改參數) ---
# ==========================================
MODE = "accomp"               # 分析模式: "melody", "accomp", "both"
SAVE_CHART = "evaluation_chart.png"
REPORT_PATH = "separation_report.csv"
SAMPLE_RATE = 44100

# ✨ 手動設定要顯示在圖表上的指標 (可選: "SDR", "SI-SDR", "SIR", "SAR")
# 例如只想看 SDR 和 SI-SDR，就改為 ["SDR", "SI-SDR"]
VISIBLE_METRICS = ["SDR", "SI-SDR", "SAR"]

# 預估 (Prediction) 檔案路徑
PRED_MELODY_PATH = r"F:\專題\piano-mir-melody-separation\results_accomp\melody_no_hint.wav"
PRED_ACCOMP_PATH = r"F:\專題\piano-mir-melody-separation\results_accomp\accompaniment_no_hint.wav"

# 真值 (Ground Truth) 檔案路徑
GT_MELODY_PATH = r"G:\project_data\two_line_midi\flac_output\melody_audio_flac\Classical_Classical_John Philip Sousa_Hands Across the Sea_melody.flac"
GT_ACCOMP_PATH = r"G:\project_data\two_line_midi\flac_output\accomp_audio_flac\Classical_Classical_John Philip Sousa_Hands Across the Sea_accomp.flac"
# ==========================================
# ==========================================

def evaluate_separation(pred_mel_path=None, pred_acc_path=None, gt_mel_path=None, gt_acc_path=None, sr=44100):
    """
    支援彈性的評估，可以只給旋律、只給伴奏，或兩者都給。
    """
    data_to_eval = []
    keys = []
    
    # 檢查旋律檔案是否存在
    if pred_mel_path and gt_mel_path and os.path.exists(pred_mel_path) and os.path.exists(gt_mel_path):
        pm, _ = librosa.load(pred_mel_path, sr=sr)
        gm, _ = librosa.load(gt_mel_path, sr=sr)
        data_to_eval.append((pm, gm))
        keys.append("Melody")
    
    # 檢查伴奏檔案是否存在
    if pred_acc_path and gt_acc_path and os.path.exists(pred_acc_path) and os.path.exists(gt_acc_path):
        pa, _ = librosa.load(pred_acc_path, sr=sr)
        ga, _ = librosa.load(gt_acc_path, sr=sr)
        data_to_eval.append((pa, ga))
        keys.append("Accomp")

    if not data_to_eval:
        print("錯誤: 找不到指定的音訊檔案或路徑無效。")
        return {}

    # 長度對齊 (取最小長度)
    min_len = min([len(p) for p, g in data_to_eval] + [len(g) for p, g in data_to_eval])
    
    preds_list = [p[:min_len] for p, g in data_to_eval]
    gts_list = [g[:min_len] for p, g in data_to_eval]
    
    preds = np.array(preds_list) # [num_sources, samples]
    gts = np.array(gts_list)     # [num_sources, samples]
    
    # 計算 SDR, SIR, SAR
    sdr, sir, sar, _ = bss_eval_sources(gts, preds, compute_permutation=False)
    
    # SI-SDR
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    
    results = {}
    for i, key in enumerate(keys):
        si_sdr = si_sdr_metric(torch.tensor(preds_list[i]), torch.tensor(gts_list[i])).item()
        results[key] = {
            "SDR": sdr[i], 
            "SIR": sir[i], 
            "SAR": sar[i], 
            "SI-SDR": si_sdr
        }
    
    return results

def main():
    # 整合 argparse，但以最上方的配置為預設值
    parser = argparse.ArgumentParser(description="分析旋律分離結果")
    parser.add_argument("--mode", type=str, choices=["melody", "accomp", "both"], default=MODE,
                        help=f"分析模式 (預設: {MODE})")
    parser.add_argument("--save_chart", type=str, default=SAVE_CHART, help=f"圖表儲存路徑 (預設: {SAVE_CHART})")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help=f"取樣率 (預設: {SAMPLE_RATE})")
    args = parser.parse_args()

    current_mode = args.mode
    print(f"--- 啟動分析 ---")
    print(f"模式: {current_mode}")
    print(f"取樣率: {args.sr}")

    # 根據模式設定要讀取的檔案
    p_mel = PRED_MELODY_PATH if current_mode in ["melody", "both"] else None
    g_mel = GT_MELODY_PATH if current_mode in ["melody", "both"] else None
    p_acc = PRED_ACCOMP_PATH if current_mode in ["accomp", "both"] else None
    g_acc = GT_ACCOMP_PATH if current_mode in ["accomp", "both"] else None

    tracks = ["Song1"]
    results = []

    for track in tracks:
        print(f"正在分析 {track}...")
        metrics = evaluate_separation(p_mel, p_acc, g_mel, g_acc, sr=args.sr)
        
        for part, values in metrics.items():
            results.append({"Track": track, "Part": part, **values})

    if not results:
        print("沒有產生任何結果，請檢查路徑設定。")
        return

    # 轉成 DataFrame
    df = pd.DataFrame(results)
    print("\n--- 評估結果摘要 ---")
    summary = df.groupby("Part")[["SDR", "SI-SDR", "SIR", "SAR"]].mean()
    print(summary)

    # 儲存 CSV
    df.to_csv(REPORT_PATH, index=False)
    print(f"報告已儲存至: {REPORT_PATH}")

    # 視覺化分析
    plot_results(df, save_path=args.save_chart, mode_name=current_mode)

def plot_results(df, save_path="evaluation_chart.png", mode_name="both"):
    plt.figure(figsize=(10, 6))
    
    # 處理可能的 inf 值
    plot_df = df.replace([np.inf, -np.inf], 50.0)
    
    # ✨ 使用配置區塊中設定的指標
    metrics = [m for m in VISIBLE_METRICS if m in plot_df.columns]
    
    if not metrics:
        print("警告: VISIBLE_METRICS 中沒有有效的指標可供顯示。")
        return
    
    # 計算平均值
    avg_results = plot_df.groupby("Part")[metrics].mean()

    x = np.arange(len(metrics))
    width = 0.35

    ax = plt.subplot(111)
    
    if "Melody" in avg_results.index and "Accomp" in avg_results.index:
        ax.bar(x - width/2, avg_results.loc["Melody"], width, label='Melody', color='skyblue')
        ax.bar(x + width/2, avg_results.loc["Accomp"], width, label='Accompaniment', color='salmon')
    elif "Melody" in avg_results.index:
        ax.bar(x, avg_results.loc["Melody"], width*1.5, label='Melody', color='skyblue')
    elif "Accomp" in avg_results.index:
        ax.bar(x, avg_results.loc["Accomp"], width*1.5, label='Accompaniment', color='salmon')

    plt.ylabel('Decibels (dB)')
    plt.title(f'Audio Separation Performance (Mode: {mode_name})')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 圖表已儲存至: {save_path}")

if __name__ == "__main__":
    main()
