import os
import glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources

# 忽略警告以保持 CMD 畫面整潔
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置區塊 (請確保路徑與你的預測腳本一致)
# ==========================================
PROJECT_ROOT = r"F:\project\piano-mir-melody-separation"

# 預測結果目錄
MEL_PRED_DIR = os.path.join(PROJECT_ROOT, "results_mel_task")
ACC_PRED_DIR = os.path.join(PROJECT_ROOT, "results_accomp")

# 標準答案目錄 (GT) - 請確認這兩個路徑是否正確
GT_MEL_DIR = r"G:\project_data\two_line_midi\flac_output\melody_audio_flac"
GT_ACC_DIR = r"G:\project_data\two_line_midi\flac_output\accomp_audio_flac"

SAMPLE_RATE = 44100
FINAL_REPORT_PATH = os.path.join(PROJECT_ROOT, "batch_test_performance_report.csv")
SUMMARY_CHART_PATH = os.path.join(PROJECT_ROOT, "batch_test_summary_chart.png")

# 要計算與顯示的指標
VISIBLE_METRICS = ["SDR", "SI-SDR", "SIR", "SAR"]
# ==========================================

def find_gt_file(gt_dir, safe_name):
    """智慧匹配：將 GT 資料夾內的檔案轉為 safe 格式進行對比"""
    if not os.path.exists(gt_dir):
        return None
    for f in os.listdir(gt_dir):
        # 移除副檔名並過濾掉底線/空格差異進行比對
        f_clean = f.replace(".flac", "").replace("_melody", "").replace("_accomp", "").replace("_mixed", "").replace(" ", "_").replace(",", "")
        if f_clean == safe_name:
            return os.path.join(gt_dir, f)
    return None

def main():
    print("--- 🚀 啟動批量測試集效能評估 (SIR 修正版) ---")
    
    # 1. 抓取所有產出的旋律測試結果
    mel_files = glob.glob(os.path.join(MEL_PRED_DIR, "melodyResult_test_*.wav"))
    
    if not mel_files:
        print(f"❌ 在 {MEL_PRED_DIR} 找不到任何測試結果！")
        return

    all_results = []
    si_sdr_metric = ScaleInvariantSignalDistortionRatio()

    # 2. 開始循環評估
    for m_path in tqdm(mel_files, desc="數據分析中"):
        filename = os.path.basename(m_path)
        safe_name = filename.replace("melodyResult_test_", "").replace(".wav", "")
        
        # 配對所有路徑
        a_path = os.path.join(ACC_PRED_DIR, f"accompResult_test_{safe_name}.wav")
        gt_m_path = find_gt_file(GT_MEL_DIR, safe_name)
        gt_a_path = find_gt_file(GT_ACC_DIR, safe_name)

        # 檢查四個檔案是否齊全
        if all([gt_m_path, gt_a_path, os.path.exists(m_path), os.path.exists(a_path)]):
            # A. 載入所有音訊
            y_p_m, _ = librosa.load(m_path, sr=SAMPLE_RATE)
            y_p_a, _ = librosa.load(a_path, sr=SAMPLE_RATE)
            y_g_m, _ = librosa.load(gt_m_path, sr=SAMPLE_RATE)
            y_g_a, _ = librosa.load(gt_a_path, sr=SAMPLE_RATE)

            # B. 四軌長度對齊 (取最小公倍長度)
            min_l = min(len(y_p_m), len(y_p_a), len(y_g_m), len(y_g_a))
            
            # C. 構建 2D 矩陣 [2, N] (第一軌旋律, 第二軌伴奏)
            preds = np.array([y_p_m[:min_l], y_p_a[:min_l]])
            gts = np.array([y_g_m[:min_l], y_g_a[:min_l]])

            # D. 同時計算 (讓 SIR 有數值)
            sdr, sir, sar, _ = bss_eval_sources(gts, preds, compute_permutation=False)
            
            # E. 記錄兩者的結果
            for i, part in enumerate(["Melody", "Accomp"]):
                si_sdr = si_sdr_metric(torch.tensor(preds[i]), torch.tensor(gts[i])).item()
                all_results.append({
                    "Song": safe_name, 
                    "Part": part, 
                    "SDR": sdr[i], 
                    "SI-SDR": si_sdr, 
                    "SIR": sir[i], 
                    "SAR": sar[i]
                })
        else:
            print(f"⚠️ 跳過歌曲 {safe_name}: 檔案不齊全")

    # 3. 數據彙整
    if not all_results:
        print("❌ 評估失敗：無法成功匹配預測檔與 GT 檔。")
        return

    # 先建立 DataFrame
    df = pd.DataFrame(all_results)
    
    # 4. CMD 畫面輸出摘要
    print("\n" + "="*60)
    print("📊 批量測試集效能總結 (Averaged Metrics)")
    print("="*60)
    
    summary = df.groupby("Part")[VISIBLE_METRICS].mean()
    print(summary)
    print("="*60)
    
    # 5. 儲存 CSV 完整清單
    df.to_csv(FINAL_REPORT_PATH, index=False)
    print(f"\n💾 完整數據報告已儲存: {os.path.basename(FINAL_REPORT_PATH)}")

    # 6. 視覺化繪圖
    plt.figure(figsize=(10, 6))
    summary.T.plot(kind='bar', color=['skyblue', 'salmon'], ax=plt.gca())
    plt.title("Average Performance across Test Set (80 Epochs)")
    plt.ylabel("Decibels (dB)")
    plt.xticks(rotation=0)
    plt.legend(title="Components")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig(SUMMARY_CHART_PATH, dpi=300, bbox_inches='tight')
    plt.close() # 關閉畫布以釋放記憶體
    print(f"📈 總體統計圖表已儲存: {os.path.basename(SUMMARY_CHART_PATH)}")

if __name__ == "__main__":
    main()