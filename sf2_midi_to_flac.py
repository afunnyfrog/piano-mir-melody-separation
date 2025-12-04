import os
import subprocess
import pretty_midi
import threading
from concurrent.futures import ThreadPoolExecutor

# --- 設定區 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FLUIDSYNTH_BIN_PATH = r"C:\fluidsynth\bin"
FLUIDSYNTH_EXE = os.path.join(FLUIDSYNTH_BIN_PATH, "fluidsynth.exe")
SOUNDFONT_PATH = r"C:\project_data\sf2\FluidR3Mono_GM2-315.SF2"

INPUT_BASE_DIR = r"C:\project_data\two_line_midi\midi_v3"
OUTPUT_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "wav_output")

# 設定為 None 代表跑完全部 (正式轉檔用)
TEST_LIMIT = None

DIR_MAPPING = [
    (os.path.join(INPUT_BASE_DIR, "train_X_mixed"), os.path.join(OUTPUT_BASE_DIR, "mix_audio_flac")),
    (os.path.join(INPUT_BASE_DIR, "train_Y_melody"), os.path.join(OUTPUT_BASE_DIR, "melody_audio_flac")),
    (os.path.join(INPUT_BASE_DIR, "train_Y_accomp"), os.path.join(OUTPUT_BASE_DIR, "accomp_audio_flac")),
]

# --- 全域變數：用來計算進度 ---
progress_lock = threading.Lock() # 確保計數器在多執行緒下準確
completed_count = 0              # 目前已完成的數量
total_files_count = 0            # 總檔案數

# --- 核心功能 ---

def convert_single_file_force_piano(args):
    global completed_count
    midi_path, flac_output_path = args

    # 【斷點續傳機制】
    # 只要檔案存在，就直接跳過，並增加計數 (視為已完成)
    if os.path.exists(flac_output_path):
        with progress_lock:
            completed_count += 1
            # 即使是跳過，如果剛好湊滿 100 也可以顯示一下，或是選擇安靜跳過
            # 這裡我們選擇：只有真正執行轉檔才顯示，或者是跳過的量很大時顯示
            # 為了讓您知道進度，這裡也納入計算
            if completed_count % 100 == 0:
                print(f"進度報告: 已處理 {completed_count}/{total_files_count} 首 (包含跳過已存在的檔案)")
        return

    # --- 以下是轉檔邏輯 ---
    temp_midi_path = flac_output_path.replace(".flac", "_temp_piano.mid")

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        for instrument in pm.instruments:
            instrument.program = 0
            instrument.is_drum = False
        pm.write(temp_midi_path)

        cmd = [
            FLUIDSYNTH_EXE, "-ni", "-g", "1.0", "-r", "44100",
            "-F", flac_output_path, "-T", "flac",
            SOUNDFONT_PATH, temp_midi_path
        ]
        
        # 執行轉檔
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # 【進度回報機制】
        with progress_lock:
            completed_count += 1
            # 每 100 首印出一行
            if completed_count % 100 == 0:
                print(f"--> 進度更新: 已完成 {completed_count} / {total_files_count} 首")

    except Exception as e:
        print(f"[Error] {midi_path} 失敗: {e}")
        # 失敗也算處理過嗎？看您定義。這裡不算入 completed_count 以免誤導，或者另外開 fail_count
    
    finally:
        if os.path.exists(temp_midi_path):
            try:
                os.remove(temp_midi_path)
            except OSError:
                pass

def main():
    global total_files_count
    
    if FLUIDSYNTH_BIN_PATH not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + FLUIDSYNTH_BIN_PATH

    tasks = []
    print("正在掃描檔案...")
    
    for input_dir, output_dir in DIR_MAPPING:
        if not os.path.exists(input_dir):
            print(f"[Warning] 找不到資料夾: {input_dir}")
            continue
        os.makedirs(output_dir, exist_ok=True)
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    midi_path = os.path.join(root, file)
                    file_name_no_ext = os.path.splitext(file)[0]
                    flac_output_path = os.path.join(output_dir, file_name_no_ext + ".flac")
                    tasks.append((midi_path, flac_output_path))

    total_files_count = len(tasks)
    
    # 應用測試限制
    if TEST_LIMIT is not None and total_files_count > TEST_LIMIT:
        print(f"【測試模式】只處理前 {TEST_LIMIT} 個檔案。")
        tasks = tasks[:TEST_LIMIT]
        total_files_count = TEST_LIMIT
    
    print(f"共 {total_files_count} 個檔案，開始轉檔 (每 100 首回報一次)...")

    if len(tasks) > 0:
        # 建議 max_workers 設為 CPU 核心數 (例如 4 或 8)
        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(convert_single_file_force_piano, tasks)
        print(f"\n全部完成！最終數量: {completed_count}/{total_files_count}")
    else:
        print("沒有任務需要執行。")

if __name__ == "__main__":
    main()