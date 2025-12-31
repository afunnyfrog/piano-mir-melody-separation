import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm

# 引入自定義模組
from utils.dataset import WeightedAudioDataset
# from utils.dataset import DirectAudioDataset # 如果您改用 DirectAudioDataset，請打開這行並註解上面那行
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# ==========================================
#               參數設定
# ==========================================
# 指向您分類好的資料夾根目錄 (或是 flac_output，視您使用的 dataset class 而定)
DATA_DIR = r"C:\Users\cebit\Desktop\專題生成\classified_dataset"
CHECKPOINT_DIR = "./checkpoints_dynamic_8s"

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
SEGMENT_SECONDS = 2.0

# ✨ [關鍵設定] 每個 Epoch 只隨機抽取 1000 筆資料
SAMPLES_PER_EPOCH = 1000

# ✨ [接續設定] 設定要接續的存檔路徑
RESUME_FROM = os.path.join(CHECKPOINT_DIR, "best_model.pth")
# RESUME_FROM = None # 如果要從頭開始，請取消註解這一行並註解上面那行

# ✨ [新功能] 設定「這次要跑幾輪」
# 每次執行程式，就會在目前的基礎上再多跑 10 輪
RUN_EPOCHS = 20

# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # ===========================
    #    Dataset 與 DataLoader
    # ===========================
    print("正在初始化動態資料集...")

    # 訓練集
    train_dataset = WeightedAudioDataset(
        root_dir=DATA_DIR,
        split="train",
        segment_seconds=SEGMENT_SECONDS,
        samples_per_epoch=SAMPLES_PER_EPOCH
    )

    # 驗證集
    val_dataset = WeightedAudioDataset(
        root_dir=DATA_DIR,
        split="val",
        segment_seconds=SEGMENT_SECONDS,
        samples_per_epoch=200
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Windows 若報錯請改 0
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
    model = AudioUNet(n_channels=1, n_classes=2).to(device)

    criterion = AudioSeparationLoss(
        alpha_l1=1.0,
        alpha_spectral=0.05,  # 給予極小的頻譜補償，確保旋律音質不會沙啞
        melody_weight=5.0,  # ✨ 從 2.0 提升到 5.0：強迫模型優先學好旋律
        accomp_weight=0.5  # ✨ 從 1.0 降低到 0.5：對伴奏的錯誤更有容忍度
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ===========================
    #      斷點續訓邏輯 (修正版)
    # ===========================
    start_epoch = 0
    best_val_loss = float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"🔄 發現存檔，正在載入: {RESUME_FROM}")
        try:
            checkpoint = torch.load(RESUME_FROM, map_location=device)

            # --- 自動判斷存檔格式 ---
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新版完整存檔
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint['epoch']
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']

                print(f"✅ 完整載入成功！目前進度: 第 {start_epoch} 輪 (Best Loss: {best_val_loss:.4f})")

            else:
                # 舊版存檔
                model.load_state_dict(checkpoint)
                print(f"⚠️ 警告：舊版存檔 (無 Epoch 資訊)。將從第 1 輪開始。")

        except Exception as e:
            print(f"❌ 載入存檔失敗: {e}")
            print("   將從頭開始訓練。")

    else:
        print("🆕 無存檔或路徑錯誤，將從頭開始訓練。")

    # ===========================
    #    ✨ [關鍵修改] 計算目標輪數
    # ===========================
    end_epoch = start_epoch + RUN_EPOCHS

    print("-" * 60)
    print(f"原本進度: {start_epoch} 輪")
    print(f"這次任務: 再跑 {RUN_EPOCHS} 輪")
    print(f"目標終點: 第 {end_epoch} 輪")
    print("-" * 60)

    # ===========================
    #        訓練迴圈
    # ===========================
    # ✨ range 改為從 start_epoch 到 end_epoch
    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()

        # --- Training ---
        model.train()
        train_loss_accum = 0

        # Progress bar 顯示目前的 Epoch / 目標 Epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}", leave=False)

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss, _ = criterion(predictions, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss_accum += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                predictions = model(data)
                loss, _ = criterion(predictions, target)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        # --- Summary & Saving ---
        duration = time.time() - start_time
        print(
            f"Epoch {epoch + 1} | Time: {duration:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 策略 1: 存最佳模型
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

        # 策略 2: 定期備份 (每 10 輪存一次檔名帶數字的)
        # 注意：這邊建議用 epoch + 1 做判斷，而不是相對次數
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # 記得把 scheduler 也存進去
                'val_loss': avg_val_loss,
            }, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth"))

        print("-" * 60)

    print("="*60)
    print(f"🎉 階段性任務完成！已跑完 {RUN_EPOCHS} 輪。")
    print(f"目前總進度: {end_epoch} 輪。")
    print("休息一下，下次繼續！")

if __name__ == "__main__":
    main()
