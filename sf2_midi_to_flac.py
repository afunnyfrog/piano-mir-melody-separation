import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# --- 設定區 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FLUIDSYNTH_BIN_PATH = r"C:\fluidsynth\bin"
FLUIDSYNTH_EXE = os.path.join(FLUIDSYNTH_BIN_PATH, "fluidsynth.exe")
SOUNDFONT_PATH = r"C:\project_data\sf2\FluidR3Mono_GM2-315.SF2"

INPUT_BASE_DIR = r"C:\project_data\two_line_midi\midi_v3"
OUTPUT_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "wav_output")

# 【控制開關】限制處理數量 (測試用)
# 設定為 5 先跑 5 個試試看，確認沒聲音且有檔案產出後，再改為 None
TEST_LIMIT = 5

DIR_MAPPING = [
    #(os.path.join(INPUT_BASE_DIR, "train_X_mix"), os.path.join(OUTPUT_BASE_DIR, "mix_audio_flac")),
    #(os.path.join(INPUT_BASE_DIR, "train_Y_melody"), os.path.join(OUTPUT_BASE_DIR, "melody_audio_flac")),
    (os.path.join(INPUT_BASE_DIR, "train_Y_accomp"), os.path.join(OUTPUT_BASE_DIR, "accomp_audio_flac")),
]

# --- 核心功能 ---

def convert_single_file(args):
    midi_path, flac_output_path = args

    if os.path.exists(flac_output_path):
        return

    # 【修正重點】
    # 參數順序必須是：[執行檔] -> [選項 -ni -F -T] -> [SoundFont] -> [MIDI檔]
    cmd = [
        FLUIDSYNTH_EXE,
        "-ni",                  # 不進入互動模式
        "-g", "1.0",            # 音量
        "-r", "44100",          # 採樣率
        "-F", flac_output_path, # 【關鍵】先告訴它要輸出到檔案，啟用快速渲染模式
        "-T", "flac",           # 指定格式
        SOUNDFONT_PATH,         # 接著放 SoundFont
        midi_path               # 最後放 MIDI 檔
    ]

    try:
        # 執行指令
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[OK] {os.path.basename(midi_path)} -> FLAC")
    except subprocess.CalledProcessError as e:
        print(f"[Error] {midi_path} 失敗: {e.stderr.decode('utf-8', errors='ignore')}")

def main():
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

    total_found = len(tasks)
    
    # 應用數量限制
    if TEST_LIMIT is not None and total_found > TEST_LIMIT:
        print(f"【測試模式】只處理前 {TEST_LIMIT} 個檔案。")
        tasks = tasks[:TEST_LIMIT]
    else:
        print(f"準備轉換 {total_found} 個檔案...")

    if len(tasks) > 0:
        # 這裡建議不要開太多 workers，以免硬碟寫入跟不上，設定 4 左右比較剛好
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(convert_single_file, tasks)
        print(f"\n轉換結束！")
    else:
        print("沒有任務需要執行。")

if __name__ == "__main__":
    main()