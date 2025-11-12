import pandas as pd

# 檔案路徑和欄位名稱
FILE_PATH = r'F:\專題\piano-mir-melody-separation\utils\maestro-v3.0.0.csv'
COMPOSER_COL = 'canonical_composer'
TITLE_COL = 'canonical_title'

# 輸出檔案的名稱
OUTPUT_FILE_COUNTS = 'composer_piece_counts.csv'
OUTPUT_FILE_SUMMARY = 'summary_statistics.csv'

try:
    # 1. 讀取 CSV 檔案 (使用您指定的絕對路徑)
    df = pd.read_csv(FILE_PATH)

    # 檢查欄位是否存在
    if COMPOSER_COL not in df.columns or TITLE_COL not in df.columns:
        print(f"錯誤：CSV 檔案中缺少 '{COMPOSER_COL}' 或 '{TITLE_COL}' 欄位。")
        print(f"偵測到的欄位: {df.columns.tolist()}")
    else:
        
        # --- 需求 2: 呈現不同作曲家及不同曲子的名稱與「各自的」數目 ---
        # 我們將計算 "貝多芬的月光奏鳴曲" 出現了幾次, "貝多芬的第五號" 出現了幾次...
        # 這符合您 "曲名A/數量" 的格式要求，並適合存成 CSV。
        
        print(f"--- 正在計算 需求 2 (分組計數) ---")
        
        # 1. 根據「作曲家」和「曲名」同時分組，並計算每個組合的出現次數
        # .size() 會計算每個 (作曲家, 曲名) 組合的資料筆數
        # .reset_index(name='數量') 會將結果轉為 DataFrame，並將計數命名為 '數量'
        piece_counts_df = df.groupby([COMPOSER_COL, TITLE_COL]).size().reset_index(name='數量')
        
        # 2. 將此結果輸出成 CSV 檔案
        #    encoding='utf-8-sig' 確保中文在 Excel 中能正確顯示
        piece_counts_df.to_csv(OUTPUT_FILE_COUNTS, index=False, encoding='utf-8-sig')
        
        print(f"成功！需求 2 的結果已儲存至: {OUTPUT_FILE_COUNTS}")
        print("以下是前 5 筆範例資料:")
        print(piece_counts_df.head())
        print("-" * 40)
        

        # --- 需求 3: 加總不重複作曲家以及不重複曲子 ---
        
        print(f"--- 正在計算 需求 3 (總結統計) ---")

        # 3a. 加總不重複「作曲家」
        total_unique_composers = df[COMPOSER_COL].nunique()
        
        # 3b. 加總不重複「曲名」 (在所有資料中)
        total_unique_pieces = df[TITLE_COL].nunique()
        
        print(f"不重複作曲家總數: {total_unique_composers}")
        print(f"不重複曲子總數 (跨所有作曲家): {total_unique_pieces}")
        
        # 3c. 建立一個新的 DataFrame 來儲存總結資料
        summary_data = {
            '統計項目': ['不重複作曲家總數', '不重複曲子總數 (跨所有作曲家)'],
            '總數': [total_unique_composers, total_unique_pieces]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # 3d. 將總結資料輸出成 CSV 檔案
        summary_df.to_csv(OUTPUT_FILE_SUMMARY, index=False, encoding='utf-8-sig')
    
        print(f"成功！需求 3 的結果已儲存至: {OUTPUT_FILE_SUMMARY}")
        print(summary_df)


except FileNotFoundError:
    print(f"錯誤 (FileNotFoundError): 找不到檔案 '{FILE_PATH}'")
    print("請再次確認您的絕對路徑是否正確。")
except KeyError as e:
    print(f"錯誤 (KeyError): 找不到欄位 {e}。")
    print(f"請確認您的 CSV 檔案中確實包含 '{COMPOSER_COL}' 和 '{TITLE_COL}' 這兩個欄位標題。")