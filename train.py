import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib
matplotlib.use('Agg') # 設為非互動式後端，防止多執行緒 GUI 錯誤
import matplotlib.pyplot as plt

# 引入自定義模組
from tools.turn_STFT_dataset import AudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# ==========================================
#               參數設定
# ==========================================
# 1. 資料與儲存設定
CSV_FILE = "classical_dataset.csv" 
CHECKPOINT_DIR = "./checkpoints_mel_task"
RESUME_BEST = False  # 是否繼承目前最佳權重
RESUME_FROM = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# 2. 訓練超參數
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
RUN_EPOCHS = 40
WARMUP_EPOCHS = 10      # 預熱 Epoch 數 
T_0 = 10               # 每個週期的 Epoch 數 (固定，因為 T_MULT=1)
T_MULT = 1             # 設定為 1，代表週期長度固定不變
ETA_MIN = 1e-6       # 學習率排程器的最小學習率

# 3. 模型設定 (必須與訓練/推論一致)
N_CHANNELS = 1       # 輸入通道: 1 代表僅讀取原始音訊頻譜
N_CLASSES = 1        # 輸出通道: 1 代表模型輸出旋律遮罩

# 4. 損失函數權重 (AudioSeparationLoss)
ALPHA_L1 = 3.0
ALPHA_SPECTRAL = 2.0
ALPHA_SISDR = 5.0 
ALPHA_SIMILARITY = 5.0 # 伴奏/旋律互斥相似度權重 (處罰旋律中的伴奏殘留)
MELODY_WEIGHT = 7.5    # 旋律擬合權重 (現在是主要目標)
ACCOMP_WEIGHT = 1.0    # 被扣除伴奏部分的參考權重

# 5. 其他設定
PRINT_FREQ = 20      # 每 20 個 batch 輸出一次進度
NUM_WORKERS = 4      # DataLoader 的並行執行數
PIN_MEMORY = True    # 加速 GPU 記憶體傳輸
DROP_LAST = True     # 捨棄最後一個不完整的 batch
# ==========================================

def main():
    # 解決某些環境下無法建立 torch kernel 快取目錄的問題
    os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
    if not os.path.exists(os.environ['PYTORCH_KERNEL_CACHE_PATH']):
        os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # ===========================
    #    Dataset 與 DataLoader
    # ===========================
    print("正在初始化魯棒多任務資料集...")

    train_dataset = AudioDataset(csv_file=CSV_FILE, split="train")
    val_dataset = AudioDataset(csv_file=CSV_FILE, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST
    )

    # ===========================
    #        模型建置
    # ===========================
    model = AudioUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)

    # 設定損失函數
    criterion = AudioSeparationLoss(
        alpha_l1=ALPHA_L1,
        alpha_spectral=ALPHA_SPECTRAL,
        alpha_sisdr=ALPHA_SISDR,
        alpha_similarity=ALPHA_SIMILARITY,
        melody_weight=MELODY_WEIGHT,
        accomp_weight=ACCOMP_WEIGHT
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 整合預熱機制
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[WARMUP_EPOCHS]
    )

    # ===========================
    #      斷點續訓邏輯
    # ===========================
    start_epoch = 0
    best_val_loss = float('inf')
    train_history, val_history = [], []

    if RESUME_BEST and RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"🔄 發現存檔，正在載入: {RESUME_FROM}")
        try:
            checkpoint = torch.load(RESUME_FROM, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            train_history = checkpoint.get('train_history', [])
            val_history = checkpoint.get('val_history', [])
            print(f"✅ 載入成功！目前進度: 第 {start_epoch} 輪")
        except Exception as e:
            print(f"❌ 載入存檔失敗: {e}，將從頭開始。")
    elif not RESUME_BEST:
        print("⏭️ 已設定不繼承權重，將從頭開始訓練。")

    end_epoch = start_epoch + RUN_EPOCHS
    print("-" * 40)

    # ===========================
    #        訓練迴圈
    # ===========================
    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()

        # --- Training ---
        model.train()
        train_loss_accum = 0

        for batch_idx, (waveforms, targets, target_audios) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            target_audios = target_audios.to(device)

            optimizer.zero_grad()
            # 模型現在回傳: pred_mel, pred_mel_audio, pred_acc_spec
            predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
            
            if predictions.shape[3] != targets.shape[3]:
                min_time = min(predictions.shape[3], targets.shape[3])
                predictions = predictions[:, :, :, :min_time]
                targets = targets[:, :, :, :min_time]
                pred_others = pred_others[:, :, :, :min_time]

            # 傳入所有輔助項進行 Loss 計算
            loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
            
            if torch.isnan(loss):
                print(f"⚠️ 警告: 第 {epoch+1} 輪 Batch {batch_idx} 偵測到 NaN Loss，正在跳過...")
                optimizer.zero_grad()
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            if batch_idx % PRINT_FREQ == 0:
                print(f"Epoch [{epoch+1}/{end_epoch}] Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss_accum / len(train_loader)
        train_history.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for waveforms, targets, target_audios in val_loader:
                waveforms = waveforms.to(device)
                targets = targets.to(device)
                target_audios = target_audios.to(device)
                
                predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
                
                if predictions.shape[3] != targets.shape[3]:
                    min_time = min(predictions.shape[3], targets.shape[3])
                    predictions = predictions[:, :, :, :min_time]
                    targets = targets[:, :, :, :min_time]
                    pred_others = pred_others[:, :, :, :min_time]
                    
                loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
                
                if not torch.isnan(loss):
                    val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        val_history.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        duration = time.time() - start_time
        print(f"==> Epoch {epoch + 1} | Time: {duration:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label='Train Loss')
        plt.plot(val_history, label='Val Loss')
        plt.legend(); plt.grid(True); plt.savefig('loss_curve.png'); plt.close()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_history': train_history,
                'val_history': val_history
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("🏆 Best Model Saved!")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth"))

    print("="*60)
    print(f"🎉 任務完成！目前總進度: {end_epoch} 輪。")

if __name__ == "__main__":
    main()
