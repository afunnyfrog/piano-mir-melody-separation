import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# 引入自定義模組
from tools.turn_STFT_dataset import AudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# ==========================================
#               參數設定
# ==========================================
CSV_FILE = "dataset.csv" 
CHECKPOINT_DIR = "./checkpoints_multi_task"

BATCH_SIZE = 24
LEARNING_RATE = 5e-5
RUN_EPOCHS = 40
PRINT_FREQ = 20  # 每 20 個 batch 輸出一次

RESUME_BEST = False  # 是否繼承目前最佳權重
RESUME_FROM = os.path.join(CHECKPOINT_DIR, "best_model.pth")

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
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # ===========================
    #        模型建置
    # ===========================
    model = AudioUNet(n_channels=3, n_classes=4).to(device)

    criterion = AudioSeparationLoss(
        alpha_l1=1.0,
        alpha_spectral=0.05,
        melody_weight=2.0,
        accomp_weight=1.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ===========================
    #      斷點續訓邏輯
    # ===========================
    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_BEST and RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"🔄 發現存檔，正在載入: {RESUME_FROM}")
        try:
            # 加入 weights_only=False 消除 FutureWarning
            checkpoint = torch.load(RESUME_FROM, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
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

        for batch_idx, (waveforms, midi_hints, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            midi_hints = midi_hints.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(waveforms, midi_hints)
            
            if predictions.shape[3] != targets.shape[3]:
                min_time = min(predictions.shape[3], targets.shape[3])
                predictions = predictions[:, :, :, :min_time]
                targets = targets[:, :, :, :min_time]

            loss, _ = criterion(predictions, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            # 每 PRINT_FREQ 次輸出進度
            if batch_idx % PRINT_FREQ == 0:
                print(f"Epoch [{epoch+1}/{end_epoch}] Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for waveforms, midi_hints, targets in val_loader:
                waveforms, midi_hints, targets = waveforms.to(device), midi_hints.to(device), targets.to(device)
                predictions = model(waveforms, midi_hints)
                
                if predictions.shape[3] != targets.shape[3]:
                    min_time = min(predictions.shape[3], targets.shape[3])
                    predictions = predictions[:, :, :, :min_time]
                    targets = targets[:, :, :, :min_time]
                    
                loss, _ = criterion(predictions, targets)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        # --- Summary & Saving ---
        duration = time.time() - start_time
        print(f"==> Epoch {epoch + 1} | Time: {duration:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("🏆 Best Model Saved!")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth"))

    print("="*60)
    print(f"🎉 任務完成！目前總進度: {end_epoch} 輪。")

if __name__ == "__main__":
    main()
