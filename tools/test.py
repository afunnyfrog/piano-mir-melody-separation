import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- 設定您的檔案路徑 (請確保這些檔案是重新轉檔後正常的檔案) ---
wav_path = r"F:\專題\piano-mir-melody-separation\wav_output\accomp_audio\Ambient_Ambient_Roger_Eno_While_The_City_Sleeps_(Album_Version)_accomp.wav"
flac_path = r"F:\專題\piano-mir-melody-separation\wav_output\accomp_audio_flac\Ambient_Ambient_Roger Eno_While The City Sleeps (Album Version)_accomp.flac"

def safe_compare_existing_files():
    print("--- 安全比較模式 (Safe Read-Only) ---")
    
    # 檢查檔案是否存在
    if not os.path.exists(wav_path) or not os.path.exists(flac_path):
        print("錯誤：找不到檔案！請確認您已經重新轉檔生成了它們。")
        return

    print(f"正在讀取檔案 (絕對不會修改內容)...")
    
    # 1. 讀取 (Read-Only)
    # duration=10 只讀前10秒，節省時間
    y_wav, sr_wav = librosa.load(wav_path, sr=None, duration=10)
    y_flac, sr_flac = librosa.load(flac_path, sr=None, duration=10)

    # 2. 比較波形數據
    # 裁切到相同長度
    min_len = min(len(y_wav), len(y_flac))
    y_wav = y_wav[:min_len]
    y_flac = y_flac[:min_len]
    
    diff = np.abs(y_wav - y_flac)
    max_diff = np.max(diff)
    
    print(f"\n最大差異值 (Max Difference): {max_diff:.9f}")
    
    if max_diff < 1e-5:
        print(">> 結果：兩個檔案【完全一致】。FLAC 轉檔成功且無損。")
    else:
        print(">> 結果：發現差異。可能是音量設定不同，或轉檔參數不一致。")

    # 3. 繪圖
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_wav, label='WAV', alpha=0.7)
    plt.plot(y_flac, label='FLAC', alpha=0.7, linestyle='--')
    plt.title("Waveform Overlay")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(diff, color='red')
    plt.title(f"Difference (Max: {max_diff:.1e})")
    
    plt.tight_layout()
    plt.show() # 如果沒視窗，請改用 plt.savefig('safe_compare.png')

if __name__ == "__main__":
    safe_compare_existing_files()