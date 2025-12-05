import os
import random
import pandas as pd
from pathlib import Path

def create_dataset_manifest(
    data_dir, 
    output_csv="dataset.csv", 
    extensions=['.wav', '.mp3', '.flac'], 
    split_ratios=(0.8, 0.1, 0.1), 
    seed=42
):
    
    if sum(split_ratios) != 1.0:
        raise ValueError("切分比例總和必須等於 1.0")

    print(f"正在掃描資料夾: {data_dir} ...")
    
    base_path = Path(data_dir)
    all_files = []
    
    # 遞迴搜尋所有指定的副檔名
    for ext in extensions:
        # 搜尋到的所有檔案
        found_files = list(base_path.rglob(f"*{ext}"))
        
        for f in found_files:
            # === 關鍵修改：只保留 Mixed 音訊 ===
            # 方法 1: 檢查檔名結尾 (最準確)
            if not f.name.endswith('_mixed.flac'):
                continue
            
            # 方法 2 (備用): 檢查路徑中是否包含 'mix_audio_flac' 資料夾
            # if 'mix_audio_flac' not in str(f):
            #     continue
                
            all_files.append(f)
    
    # 轉換成絕對路徑字串，並排序
    all_files = sorted([str(p.resolve()) for p in all_files])
    
    total_files = len(all_files)
    if total_files == 0:
        print("錯誤：找不到任何符合條件的 '_mixed.flac' 檔案。")
        return

    print(f"共找到 {total_files} 個 Mixed 音訊檔案 (已過濾 Melody/Accomp)。")

    # --- 以下邏輯保持不變 ---
    random.seed(seed)
    random.shuffle(all_files)

    n_train = int(total_files * split_ratios[0])
    n_val = int(total_files * split_ratios[1])
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"切分結果 -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    data_list = []
    for f in train_files:
        data_list.append({'song_id': os.path.basename(f).replace('_mixed.flac',''), 'file_path': f, 'split_type': 'train'})
    for f in val_files:
        data_list.append({'song_id': os.path.basename(f).replace('_mixed.flac',''), 'file_path': f, 'split_type': 'val'})
    for f in test_files:
        data_list.append({'song_id': os.path.basename(f).replace('_mixed.flac',''), 'file_path': f, 'split_type': 'test'})

    df = pd.DataFrame(data_list)
    df = df[['song_id', 'split_type', 'file_path']]
    
    df.to_csv(output_csv, index=False)
    print(f"成功生成資料清單！檔案已儲存至: {output_csv}")

# --- 主程式 ---
if __name__ == "__main__":
    # 使用 Raw String (r"...") 避免路徑錯誤
    MY_DATA_DIR = r"C:\project_data\two_line_midi\flac_output" 
    
    create_dataset_manifest(
        data_dir=MY_DATA_DIR,
        output_csv="dataset.csv",
        split_ratios=(0.8, 0.1, 0.1), 
        seed=2023
    )