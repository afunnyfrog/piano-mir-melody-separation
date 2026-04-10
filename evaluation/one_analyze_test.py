import os
import glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources

# 忽略 mir_eval 的版本警告，讓 CMD 畫面更乾淨
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# --- 自動化配置區塊 ---
# ==========================================
PROJECT_ROOT = r"C:\Users\tt\Desktop\project"
MEL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_mel_task")
ACC_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_accomp")

GT_MEL_DIR = r"C:\Users\tt\Desktop\piano-mir-melody-separation--\data\flac_output\melody_audio_flac"
GT_ACC_DIR = r"C:\Users\tt\Desktop\piano-mir-melody-separation--\data\flac_output\accomp_audio_flac"

SAMPLE_RATE = 44100
VISIBLE_METRICS = ["SDR", "SI-SDR", "SIR", "SAR"]
# ==========================================

def get_latest_file(directory, pattern):
    list_of_files = glob.glob(os.path.join(directory, pattern))
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getmtime)

def find_gt_file(gt_dir, safe_name):
    if not os.path.exists(gt_dir): return None
    for f in os.listdir(gt_dir):
        f_clean = f.replace(".flac", "").replace("_melody", "").replace("_accomp", "").replace("_mixed", "").replace(" ", "_").replace(",", "")
        if f_clean == safe_name: return os.path.join(gt_dir, f)
    return None

def evaluate_separation(pred_paths, gt_paths):
    data_to_eval = []
    keys = []
    for key in ['Melody', 'Accomp']:
        p, g = pred_paths.get(key), gt_paths.get(key)
        if p and g and os.path.exists(p) and os.path.exists(g):
            y_p, _ = librosa.load(p, sr=SAMPLE_RATE)
            y_g, _ = librosa.load(g, sr=SAMPLE_RATE)
            data_to_eval.append((y_p, y_g))
            keys.append(key)
    if not data_to_eval: return []

    min_len = min([len(p) for p, g in data_to_eval] + [len(g) for p, g in data_to_eval])
    preds = np.array([p[:min_len] for p, g in data_to_eval])
    gts = np.array([g[:min_len] for p, g in data_to_eval])
    
    sdr, sir, sar, _ = bss_eval_sources(gts, preds, compute_permutation=False)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()
    
    results = []
    for i, key in enumerate(keys):
        si_sdr = si_sdr_metric(torch.tensor(preds[i]), torch.tensor(gts[i])).item()
        results.append({"Part": key, "SDR": sdr[i], "SIR": sir[i], "SAR": sar[i], "SI-SDR": si_sdr})
    return results

def main():
    print("\n--- 啟動分析 ---")
    latest_mel = get_latest_file(MEL_RESULTS_DIR, "melodyResult_*.wav")
    latest_acc = get_latest_file(ACC_RESULTS_DIR, "accompResult_*.wav")
    
    if not latest_mel and not latest_acc:
        print("❌ 找不到預測檔案。")
        return

    ref_file = latest_mel if latest_mel else latest_acc
    filename = os.path.basename(ref_file)
    parts = filename.replace(".wav", "").split("_")
    split_type = parts[1]
    safe_name = "_".join(parts[2:])
    
    print(f"模式: {'both' if latest_mel and latest_acc else ('melody' if latest_mel else 'accomp')}")
    print(f"取樣率: {SAMPLE_RATE}")
    print(f"正在分析 {safe_name}...")

    gt_mel = find_gt_file(GT_MEL_DIR, safe_name)
    gt_acc = find_gt_file(GT_ACC_DIR, safe_name)

    eval_results = evaluate_separation({'Melody': latest_mel, 'Accomp': latest_acc}, {'Melody': gt_mel, 'Accomp': gt_acc})
    
    if not eval_results:
        print("❌ 評估失敗。")
        return

    # ✨ 核心：在 CMD 印出結果摘要
    df = pd.DataFrame(eval_results)
    print("\n--- 評估結果摘要 ---")
    # 格式化輸出，只顯示需要的指標並設定索引
    summary = df.set_index("Part")[VISIBLE_METRICS]
    print(summary)

    # 儲存與繪圖邏輯
    for res in eval_results:
        task = res['Part'].lower()
        task_label = "melody" if task == "melody" else "accomp"
        
        # 存 CSV
        csv_path = os.path.join(PROJECT_ROOT, f"report_{task_label}_{safe_name}.csv")
        pd.DataFrame([res]).to_csv(csv_path, index=False)
        
        # 畫圖
        chart_path = os.path.join(PROJECT_ROOT, f"chart_{task_label}_{safe_name}.png")
        plt.figure(figsize=(6, 4))
        plt.bar(VISIBLE_METRICS, [res[m] for m in VISIBLE_METRICS], color='skyblue' if task=='melody' else 'salmon')
        plt.title(f"{res['Part']} Evaluation")
        plt.ylabel("dB")
        plt.savefig(chart_path)
        plt.close()

    print(f"\n報告已儲存至: report_[task]_{safe_name}.csv")
    print(f"📈 圖表已儲存至: chart_[task]_{safe_name}.png")

if __name__ == "__main__":
    main()