import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time  # 新增時間計算功能

# 引入我們剛寫好的模組
from utils.dataset import AudioDataset
from utils.u_net import AudioUNet

# --- 設定 ---
DATA_DIR = r"C:\Users\richa\Documents\專題數據\midi_batch"
CHECKPOINT_DIR = "./checkpoints_softmax"
BATCH_SIZE = 20
EPOCHS = 50
LEARNING_RATE = 5e-5
LOG_INTERVAL = 20 # 每 20 個 Batch 印一次進度

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    print("正在讀取資料集...")
    train_dataset = AudioDataset(DATA_DIR)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        
        # --- 2. i5-14500 專屬加速設定 ---
        # 設定為 6，剛好對應你的 6 個 P-core (效能核心)
        # 不要設太大 (例如 12 或 14)，在 Windows 上反而會變慢
        num_workers=6,  
        
        # --- 3. 記憶體加速 ---
        pin_memory=True,         # 必開：加速 CPU 傳顯存
        
        # --- 4. 保持活躍 (關鍵) ---
        # 讓這 6 個 P-core 執行緒在 Epoch 結束後不解散
        # 這樣下一個 Epoch 開始時就不用重新暖機，解決震盪問題
        persistent_workers=True, 
        
        # --- 5. 預先囤貨 ---
        # 讓每個 Worker 預先多讀 3 份資料
        # 6 workers * 3 = 18 份資料隨時準備好餵給 GPU
        prefetch_factor=3,
        
        drop_last=True
    )

    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #手動計算loss
    #頻率權重矩陣
    freq_weights = torch.linspace(1.0, 5.0, steps=1024).to(device)
    # 調整形狀以符合廣播機制: (Batch, Channel, Freq, Time) -> (1, 1, 1024, 1)
    freq_weights = freq_weights.view(1, 1, 1024, 1)

    print(f"開始訓練！總共 {EPOCHS} 輪，每輪有 {len(train_loader)} 個 Batch。")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time() # 計時開始
        
        # --- 改用 enumerate，移除 tqdm ---
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            #輸出是mask
            pred_masks = model(data)
            # 分離後的聲音 = 原曲 (data) * 濾鏡 (pred_masks)
            # data 是 (B, 1, F, T)，pred_masks 是 (B, 2, F, T)，會自動廣播
            separated_spectrograms = pred_masks * data
            
            # ### 修改 4：計算加權 L1 Loss ###
            # 原本是: L1 loss = criterion(predictions, target)
            # -----------------------------------------------------------
            # 1. 計算絕對誤差 |預測值 - 真實值|
            abs_diff = torch.abs(separated_spectrograms - target)
            # 2. 乘上頻率權重 (高頻錯誤會被放大)
            weighted_diff = abs_diff * freq_weights
            # 3. 取平均值當作 Loss
            loss = torch.mean(weighted_diff)
            
            target = target.to(device)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # --- 控制輸出頻率 ---
            # 每 20 個 Batch 才印一次，或是最後一個 Batch 一定要印
            if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == len(train_loader):
                # 計算經過時間
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{EPOCHS}] "
                      f"Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Time: {elapsed:.1f}s")

        # 每一輪結束的總結
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")
        print("-" * 60)

        # 存檔
        save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    print("訓練全部完成！")

if __name__ == "__main__":
    main()