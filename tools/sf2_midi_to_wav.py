import os
import subprocess
import shutil
import tempfile
import uuid
import stat

# --- 1. 路徑設定 ---
# 取得專案根目錄 (假設此腳本在 tools/ 資料夾下)
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# 請將 fluidsynth 放入專案目錄下的 tools/fluidsynth，或修改以下路徑
FLUIDSYNTH_BIN_PATH = os.path.join(PROJECT_ROOT, "tools", "fluidsynth", "bin")
FLUIDSYNTH_EXE = os.path.join(FLUIDSYNTH_BIN_PATH, "fluidsynth.exe")
# 音色庫檔案
SOUNDFONT_PATH = os.path.join(PROJECT_ROOT, "data", "sf2", "FluidR3Mono_GM2-315.SF2")

# 輸入資料夾 (MIDI 來源)
INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "two_line_midi")

# 輸出資料夾 (改到 F 槽，跟你的程式放在一起，絕對有權限)
OUTPUT_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "wav_output")

# 定義對應關係
DIR_MAPPING = [
    # (輸入位置, 輸出位置)
    (os.path.join(INPUT_BASE_DIR, "raw_midi_files"), os.path.join(OUTPUT_BASE_DIR, "raw_audio")),
    (os.path.join(INPUT_BASE_DIR, "train_Y_melody"), os.path.join(OUTPUT_BASE_DIR, "melody_audio")),
    (os.path.join(INPUT_BASE_DIR, "train_Y_accomp"), os.path.join(OUTPUT_BASE_DIR, "accomp_audio")),
]

total_processed = 0

def convert_via_temp(midi_source_path, output_dir, final_wav_name):
    # 使用系統暫存區進行轉換
    sys_temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex[:8]
    
    # 暫存路徑
    temp_midi_path = os.path.join(sys_temp_dir, f"tmp_{unique_id}.mid")
    temp_wav_path = os.path.join(sys_temp_dir, f"tmp_{unique_id}.wav")
    
    final_dest_path = os.path.join(output_dir, final_wav_name)
    abs_soundfont = os.path.abspath(SOUNDFONT_PATH)

    try:
        # A. 複製 MIDI 到暫存區
        shutil.copy2(midi_source_path, temp_midi_path)
        
        # B. 轉換
        command = [
            FLUIDSYNTH_EXE,
            "-ni", "-g", "1.0", "-r", "44100",
            "-F", temp_wav_path,
            abs_soundfont,
            temp_midi_path
        ]
        
        subprocess.run(command, check=False, capture_output=True)
        
        # C. 搬移到 F 槽
        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 1000:
            # 如果 F 槽已經有檔案，先刪除
            if os.path.exists(final_dest_path):
                try:
                    os.chmod(final_dest_path, stat.S_IWRITE)
                    os.remove(final_dest_path)
                except:
                    pass
            
            shutil.move(temp_wav_path, final_dest_path)
            return True
        return False

    except Exception as e:
        # print(f"[錯誤] {e}") # 保持版面乾淨，有需要再打開
        return False
        
    finally:
        # D. 清理暫存
        for f in [temp_midi_path, temp_wav_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

def main():
    global total_processed
    # 殺死背景程序
    subprocess.run(["taskkill", "/F", "/IM", "fluidsynth.exe"], capture_output=True)

    print(f"--- 開始轉換 ---")
    print(f"輸入來源 (C槽): {INPUT_BASE_DIR}")
    print(f"輸出目標 (F槽): {OUTPUT_BASE_DIR}")
    print("-" * 30)

    # 建立輸出目錄 (這次是在 F 槽建立，不會被拒絕)
    for _, dst_dir in DIR_MAPPING:
        if not os.path.exists(dst_dir):
            try:
                os.makedirs(dst_dir)
            except PermissionError:
                print(f"[嚴重錯誤] 無法在 F 槽建立資料夾: {dst_dir}")
                return

    for src_dir, dst_dir in DIR_MAPPING:
        if not os.path.exists(src_dir):
            print(f"[跳過] 找不到來源: {src_dir}") 
            continue

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.mid', '.midi'))]
        total_files = len(files)
        print(f"\n目錄: {src_dir} (共 {total_files} 首)")

        for i, file_name in enumerate(files):
            input_path = os.path.join(src_dir, file_name)
            safe_name = os.path.splitext(file_name)[0].replace(" ", "_") + ".wav"
            
            if convert_via_temp(input_path, dst_dir, safe_name):
                total_processed += 1
            
            if total_processed > 0 and (total_processed % 50 == 0 or i == total_files - 1):
                print(f"已完成 {total_processed} 首... (剛完成: {safe_name})")

    print(f"\n--- 任務結束，共成功轉換 {total_processed} 首 ---")
    print(f"檔案已儲存於: {OUTPUT_BASE_DIR}")
    subprocess.run(["taskkill", "/F", "/IM", "fluidsynth.exe"], capture_output=True)

if __name__ == "__main__":
    main()