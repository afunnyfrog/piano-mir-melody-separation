import os
import shutil
import librosa
import soundfile as sf
from tqdm import tqdm

# --- 設定 ---
DATA_DIR = r"C:\Users\richa\Documents\專題數據\midi_batch"
MOVE_BAD_FILES = True 
QUARANTINE_DIR = "./bad_data_quarantine"

def main():
    print(f"🧹 開始「極限」檢查資料集：{DATA_DIR}")
    
    raw_dir = os.path.join(DATA_DIR, 'raw')
    mel_dir = os.path.join(DATA_DIR, 'melody')
    acc_dir = os.path.join(DATA_DIR, 'accomp')

    if not os.path.exists(raw_dir): return

    filenames = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith('.wav')])
    
    if MOVE_BAD_FILES:
        os.makedirs(os.path.join(QUARANTINE_DIR, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(QUARANTINE_DIR, 'melody'), exist_ok=True)
        os.makedirs(os.path.join(QUARANTINE_DIR, 'accomp'), exist_ok=True)

    bad_files_list = []

    for fname in tqdm(filenames, desc="極限掃描中"):
        is_bad = False
        reason = ""
        paths = {
            'raw': os.path.join(raw_dir, fname),
            'mel': os.path.join(mel_dir, fname),
            'acc': os.path.join(acc_dir, fname)
        }

        if not is_bad:
            try:
                for key, p in paths.items():
                    # 測試 1: 用 SoundFile 嘗試跳到最後 (檢測 fseek error)
                    with sf.SoundFile(p) as f:
                        if f.frames > 0:
                            f.seek(f.frames - 1)
                            f.read(1)
                        else:
                            raise ValueError("Frames 為 0")
                    
                    # 測試 2: Librosa 讀取最後 1 秒 (檢測資料完整性)
                    # 取得總長度
                    dur = librosa.get_duration(path=p)
                    if dur > 0:
                        # 嘗試讀取最後 0.1 秒
                        librosa.load(p, sr=44100, offset=max(0, dur-0.1), duration=0.1)

            except Exception as e:
                is_bad = True
                reason = f"{key} 損壞: {str(e)}"

        if is_bad:
            bad_files_list.append(f"{fname} | {reason}")
            if MOVE_BAD_FILES:
                try:
                    for key in ['raw', 'mel', 'acc']:
                        src = paths[key]
                        dst = os.path.join(QUARANTINE_DIR, {'raw':'raw','mel':'melody','acc':'accomp'}[key], fname)
                        if os.path.exists(src): shutil.move(src, dst)
                except: pass

    print(f"\n掃描完成！發現 {len(bad_files_list)} 個壞檔。")
    if bad_files_list:
        with open("bad_files_report_extreme.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(bad_files_list))

if __name__ == "__main__":
    main()