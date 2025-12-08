import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time  # 新增時間計算功能

# 引入我們剛寫好的模組
from utils.dataset import AudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss 

# --- 設定 ---
DATA_DIR = r"C:\project_data\two_line_midi\flac_output"
CHECKPOINT_DIR = "./checkpoints_direct_L1_MSS"
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
    
    criterion = AudioSeparationLoss(
        alpha_l1=1.0,        # L1 基礎權重
        alpha_spectral=2.0,  # Multi-Scale Spectral 權重 (更重要)
        alpha_sisdr=0.0,     # 暫時不使用時域 Loss (需要相位重建)
        melody_weight=1.5,   # 旋律比伴奏更重要
        accomp_weight=1.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    """ 不同的學習率方式
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 前 30% 時間 warm-up
        anneal_strategy='cos'
    )
    """
    
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
        train_loss_detail = {
            'l1': 0, 'l1_melody': 0, 'l1_accomp': 0, 'spectral': 0
        }
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # predictions 直接視為 "預測的 Log 頻譜圖"
            predictions = model(data)
            # 複合 Loss
            loss, loss_dict = criterion(predictions, target)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            # scheduler.step() OneCycleLR方式
            
            train_loss_accum += loss.item()
            for key in train_loss_detail.keys():
                if key in loss_dict:
                    train_loss_detail[key] += loss_dict[key]
            
            # --- 控制輸出頻率 ---
            # 每 20 個 Batch 才印一次，或是最後一個 Batch 一定要印
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | LR: {curr_lr:.6f}")
                print(f"  └─ L1: {loss_dict['l1']:.4f} (M:{loss_dict['l1_melody']:.4f} A:{loss_dict['l1_accomp']:.4f}) "
                      f"Spectral: {loss_dict['spectral']:.4f}")
        
        avg_train_loss = train_loss_accum / len(train_loader)
        for key in train_loss_detail.keys():
            train_loss_detail[key] /= len(train_loader)
        
        # ===========================
        #      Validation 階段
        # ===========================
        model.eval() # 切換模式 (重要！)
        val_loss_accum = 0
        val_loss_detail = {
            'l1': 0, 'l1_melody': 0, 'l1_accomp': 0, 'spectral': 0
        }
        
        with torch.no_grad(): # 關閉梯度計算，節省記憶體
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                predictions = model(data)
                loss, loss_dict = criterion(predictions, target)
            
                val_loss_accum += loss.item()
                for key in val_loss_detail.keys():
                    if key in loss_dict:
                        val_loss_detail[key] += loss_dict[key]
        
        avg_val_loss = val_loss_accum / len(val_loader)
        for key in val_loss_detail.keys():
            val_loss_detail[key] /= len(val_loader)
        
        # ===========================
        #      結算與存檔
        # ===========================
        end_time = time.time()
        epoch_duration = end_time - start_time
        curr_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary")
        print(f"{'='*60}")
        print(f"⏱️  時間: {epoch_duration:.2f}s | 學習率: {curr_lr:.6f}")
        print(f"📊 Train Loss: {avg_train_loss:.4f}")
        print(f"   ├─ L1: {train_loss_detail['l1']:.4f} (M:{train_loss_detail['l1_melody']:.4f} A:{train_loss_detail['l1_accomp']:.4f})")
        print(f"   └─ Spectral: {train_loss_detail['spectral']:.4f}")
        print(f"📈 Val Loss: {avg_val_loss:.4f}")
        print(f"   ├─ L1: {val_loss_detail['l1']:.4f} (M:{val_loss_detail['l1_melody']:.4f} A:{val_loss_detail['l1_accomp']:.4f})")
        print(f"   └─ Spectral: {val_loss_detail['spectral']:.4f}")
        

        # 策略 1: 保存最佳模型 (Best Checkpoint)
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            print(f"🏆 Validation Loss Improved by {improvement:.4f}! Saving model...")
            best_val_loss = avg_val_loss
            
            # 保存完整狀態
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        # 策略 2: 定期保存 (例如每 10 輪存一次，或是只存最後一輪)
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"))
        
        print("-" * 60)

    # 訓練結束，存最後一個
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'final_val_loss': avg_val_loss,
    }, os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    print("=" * 60)
    print("🎉 訓練全部完成！")
    print(f"最佳 Validation Loss: {best_val_loss:.4f}")
    print(f"模型已保存至: {CHECKPOINT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()