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
CHECKPOINT_DIR = "./checkpoints_direct_L1"
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
    train_dataset = AudioDataset(csv_file="dataset.csv", split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=6,  
        pin_memory=True,        
        persistent_workers=True, 
        prefetch_factor=3,
        drop_last=True)
    
    val_dataset = AudioDataset(csv_file="dataset.csv", split="val")
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # 驗證不需要打亂
        num_workers=4, # 驗證通常比較快，worker 少一點沒關係
        pin_memory=True
    )
    

    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    best_val_loss = float('inf')

    print(f"開始訓練！總共 {EPOCHS} 輪，每輪有 {len(train_loader)} 個 Batch。")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        start_time = time.time() # 計時開始
        # ===========================
        #       Training 階段
        # ===========================
        model.train()
        train_loss_accum = 0
        # --- 改用 enumerate，移除 tqdm ---
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # predictions 直接視為 "預測的 Log 頻譜圖"
            predictions = model(data)
            # -拆分通道計算 Loss
            # loss = |結果旋律 - 目標旋律|+|結果伴奏 - 目標伴奏|
            pred_melody = predictions[:, 0, :, :]
            pred_accomp = predictions[:, 1, :, :]
        
            target_melody = target[:, 0, :, :]
            target_accomp = target[:, 1, :, :]
        
            loss_melody = criterion(pred_melody, target_melody)
            loss_accomp = criterion(pred_accomp, target_accomp)
            loss = 1.5*loss_melody + 1.0*loss_accomp
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            # --- 控制輸出頻率 ---
            # 每 20 個 Batch 才印一次，或是最後一個 Batch 一定要印
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Train Loss: {loss.item():.4f}")
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # ===========================
        #      Validation 階段
        # ===========================
        model.eval() # 切換模式 (重要！)
        val_loss_accum = 0
        
        with torch.no_grad(): # 關閉梯度計算，節省記憶體
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                predictions = model(data)
                pred_melody = predictions[:, 0, :, :]
                pred_accomp = predictions[:, 1, :, :]
                target_melody = target[:, 0, :, :]
                target_accomp = target[:, 1, :, :]
            
                loss = criterion(pred_melody, target_melody) + criterion(pred_accomp, target_accomp)
                val_loss_accum += loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # ===========================
        #      結算與存檔
        # ===========================
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            curr_lr = optimizer.param_groups[0]['lr'] # 獲取當前 LR 方便列印
        else:
            curr_lr = 0.0

        end_time = time.time()
        epoch_duration = end_time - start_time
    
        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"Time: {epoch_duration:.2f}s | LR: {curr_lr:.6f}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        

        # 策略 1: 保存最佳模型 (Best Checkpoint)
        if avg_val_loss < best_val_loss:
            print(f"   🏆 Validation Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        
        # 策略 2: 定期保存 (例如每 10 輪存一次，或是只存最後一輪)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"))
        
        print("-" * 60)

    # 訓練結束，存最後一個
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    print("訓練全部完成！")

if __name__ == "__main__":# ===========================
    #       Training 階段
    # ===========================
    main()