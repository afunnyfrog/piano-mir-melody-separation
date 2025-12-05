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
DATA_DIR = r"C:\project_data\two_line_midi\flac_output"
CHECKPOINT_DIR = "./checkpoints_direct_mse"
BATCH_SIZE = 20
EPOCHS = 50
LEARNING_RATE = 1e-3
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
            # 這裡出來的 predictions 直接視為 "預測的 Log 頻譜圖"
            predictions = model(data)
            # ---  直接比較 (Direct Regression) ---
            # 不再乘回原曲 (Masking)，而是直接要求模型畫出跟 Target 一樣的圖
            # 這樣最簡單，數學上最不會打架
            loss = criterion(predictions, target)
            
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