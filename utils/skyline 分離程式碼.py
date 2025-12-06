import pretty_midi
import numpy as np
import os
import shutil
import time

# --- v10 終極修正版 ---
#
# 1. 假設您已經在「步驟一」中執行了 "mkdir C:\tools\fluidsynth\bin"
#    - 這是為了修復 "import pretty_midi" 載入不完整的問題。
#
# 2. 我們使用 v8 的正確函式呼叫 'pretty_midi.piano_roll_to_instrument'
#    - v9 證實了 'utilities' 是錯誤的路徑。
#
# --------------------------


# --- 1. 請修改以下路徑 ---

# 您下載的 ADL Piano MIDI 檔案所在的資料夾
SOURCE_ADL_DIR = r"C:\Users\cebit\Downloads\adl-piano-midi\adl-piano-midi"

# 處理完後，用來訓練的資料要存在哪裡
OUTPUT_DATA_DIR = r"C:\Users\cebit\Desktop\專題生成\1111 skyhigh訓練資料"

# --------------------------


print(f"--- MIDI 批次拆分腳本 (Skyline 法) (v10 - 終極修正版) ---")
print(f"來源根目錄: {SOURCE_ADL_DIR}")
print(f"輸出資料夾: {OUTPUT_DATA_DIR}\n")
print(f"[狀態] v10: 假設 'mkdir C:\\tools\\fluidsynth\\bin' 已成功執行。")
print(f"[狀態] v10: 正在使用 'pretty_midi.piano_roll_to_instrument' 函式。")

# 建立輸出資料夾
OUTPUT_X_DIR = os.path.join(OUTPUT_DATA_DIR, "train_X_mixed")
OUTPUT_Y_MELODY_DIR = os.path.join(OUTPUT_DATA_DIR, "train_Y_melody")
OUTPUT_Y_ACCOMP_DIR = os.path.join(OUTPUT_DATA_DIR, "train_Y_accomp")

try:
    os.makedirs(OUTPUT_X_DIR, exist_ok=True)
    os.makedirs(OUTPUT_Y_MELODY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_Y_ACCOMP_DIR, exist_ok=True)
    print(f"[狀態] v10: 輸出資料夾已準備就緒。")
except Exception as e:
    print(f"[錯誤] 無法建立輸出資料夾: {e}")
    exit()


# 設定鋼琴捲簾的解析度 (fs=100 代表 10ms 一個 time step)
FS = 100

def apply_skyline_split(midi_file_path, unique_base_name):
    """
    對單一 MIDI 檔案應用 Skyline 演算法並儲存 X, Y_melody, Y_accomp
    """
    try:
        # 1. 載入 MIDI
        pm = pretty_midi.PrettyMIDI(midi_file_path)

        # 2. 取得鋼琴捲簾
        piano_roll = pm.get_piano_roll(fs=FS)  # Shape: (128, T)

        if piano_roll.shape[1] == 0:
            return False, "Empty file"

        # 3. 創建空白的輸出捲簾
        melody_roll = np.zeros_like(piano_roll)
        accomp_roll = np.zeros_like(piano_roll)

        # 4. Skyline 演算法
        for t in range(piano_roll.shape[1]):
            notes_at_t = piano_roll[:, t]
            playing_pitches = np.where(notes_at_t > 0)[0]

            if playing_pitches.size > 0:
                highest_pitch = playing_pitches[-1]
                melody_roll[highest_pitch, t] = notes_at_t[highest_pitch]
                if playing_pitches.size > 1:
                    other_pitches = playing_pitches[:-1]
                    accomp_roll[other_pitches, t] = notes_at_t[other_pitches]

        # --- ✅ 5. 將 piano-roll 手動轉回 PrettyMIDI 物件 ----

        def roll_to_pretty_midi(roll, fs, program=0):
            """
            手動將 piano-roll 轉回 PrettyMIDI Instrument
            """
            inst = pretty_midi.Instrument(program=program)

            # roll shape: (128, T)
            # 我們找出每個 pitch 的啟動區段
            for pitch in range(128):
                velocity_vector = roll[pitch]
                nonzero = np.where(velocity_vector > 0)[0]

                if len(nonzero) == 0:
                    continue

                # 尋找連續的 note 區段
                boundaries = np.where(np.diff(nonzero) > 1)[0]
                segment_starts = np.insert(nonzero[boundaries + 1], 0, nonzero[0])
                segment_ends = np.append(nonzero[boundaries], nonzero[-1])

                # 將每個 note 加進 instrument
                for start, end in zip(segment_starts, segment_ends):
                    start_time = start / fs
                    end_time = (end + 1) / fs

                    # clamp velocity 到合法範圍 1~127
                    velocity = min(max(int(velocity_vector[start]), 1), 127)

                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    inst.notes.append(note)


            pm_obj = pretty_midi.PrettyMIDI()
            pm_obj.instruments.append(inst)
            return pm_obj

        # ✅ Melody
        pm_melody = roll_to_pretty_midi(melody_roll, FS)
        pm_melody.write(os.path.join(OUTPUT_Y_MELODY_DIR, f"{unique_base_name}_melody.mid"))

        # ✅ Accompaniment
        pm_accomp = roll_to_pretty_midi(accomp_roll, FS)
        pm_accomp.write(os.path.join(OUTPUT_Y_ACCOMP_DIR, f"{unique_base_name}_accomp.mid"))

        # 6. 複製 X (原始混合檔)
        shutil.copy(
            midi_file_path,
            os.path.join(OUTPUT_X_DIR, f"{unique_base_name}_mixed.mid")
        )

        return True, "Success"

    except Exception as e:
        return False, str(e)


# --- 主迴圈 (使用 os.walk) ---

try:
    # --- 步驟 A: 預先計算檔案總數 (用於顯示進度) ---
    print(f"[狀態] v10: 正在掃描檔案總數 (請稍候)...")
    total_midi_files = 0
    for dirpath, dirnames, filenames in os.walk(SOURCE_ADL_DIR):
        for filename in filenames:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                total_midi_files += 1

    if total_midi_files == 0:
        print(f"[嚴重錯誤] 在 {SOURCE_ADL_DIR} 中找不到任何 .mid 或 .midi 檔案！")
        print("請檢查 SOURCE_ADL_DIR 路徑是否設定正確。")
        exit()

    print(f"總共找到 {total_midi_files} 個 MIDI 檔案。\n")

    # --- 步驟 B: 開始處理 ---
    print(f"[開始] v10: 正在處理 {total_midi_files} 個 MIDI 檔案...")
    start_time = time.time()
    file_count = 0
    success_count = 0
    error_count = 0

    for dirpath, dirnames, filenames in os.walk(SOURCE_ADL_DIR):
        for filename in filenames:
            if not (filename.endswith('.mid') or filename.endswith('.midi')):
                continue

            file_count += 1

            full_file_path = os.path.join(dirpath, filename)

            # 建立唯一的檔案基礎名稱
            relative_path = os.path.relpath(full_file_path, SOURCE_ADL_DIR)
            safe_path_name = relative_path.replace(os.sep, "_")
            unique_base_name = os.path.splitext(safe_path_name)[0]

            # 執行拆分
            success, message = apply_skyline_split(full_file_path, unique_base_name)

            if success:
                success_count += 1
            else:
                error_count += 1
                if "Empty file" not in message: # 只印出真正的錯誤
                    print(f"  [!] 處理 {relative_path} 失敗: {message}")

            # 每 1000 個檔案回報 einmal 進度
            if file_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  ...進度: {file_count} / {total_midi_files} (成功 {success_count} / 失敗 {error_count}) - 耗時: {elapsed:.2f} 秒")

except FileNotFoundError:
    print(f"[嚴重錯誤] 找不到來源資料夾: {SOURCE_ADL_DIR}")
    print("請檢查 SOURCE_ADL_DIR 路徑是否設定正確！")
    exit()
except Exception as e:
    print(f"[嚴重錯誤] 讀取資料夾時發生意外: {e}")
    exit()


# --- 最終報告 ---
end_time = time.time()
total_time = end_time - start_time

print(f"\n--- 處理完成 ---")
print(f"總共掃描 {file_count} / {total_midi_files} 個 MIDI 檔案。")
print(f"成功分離並儲存 {success_count} 組訓練樣本。")
print(f"失敗 {error_count} 個 (可能為空檔案或格式錯誤)。")
print(f"總耗時: {total_time:.2f} 秒")
print(f"您的 (X, Y) 訓練資料已儲存在: {OUTPUT_DATA_DIR}")
