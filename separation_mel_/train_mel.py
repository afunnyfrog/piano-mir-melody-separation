import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib
matplotlib.use('Agg') # 設為非互動式後端，防止多執行緒 GUI 錯誤
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import mlflow
import argparse

# 引入自定義模組
from tools.turn_STFT_dataset import AudioDataset
from utils_melody.u_net import AudioUNet
from utils_melody.loss import AudioSeparationLoss

@dataclass
class TrainConfig:
    # 1. 資料與儲存設定
    csv_file: str = "classical_dataset.csv" 
    checkpoint_dir: str = "./checkpoints_mel_task"
    resume_best: bool = False  # 是否繼承目前最佳權重
    resume_from: str = "./checkpoints_mel_task/best_model.pth"

    # 2. 訓練超參數
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    run_epochs: int = 80
    warmup_epochs: int = 10      # 預熱 Epoch 數 
    t_0: int = 10               # 每個週期的 Epoch 數 (固定，因為 T_MULT=1)
    t_mult: int = 1             # 設定為 1，代表週期長度固定不變
    eta_min: float = 1e-6       # 學習率排程器的最小學習率

    # 3. 模型設定 (必須與訓練/推論一致)
    n_channels: int = 1       # 輸入通道: 1 代表僅讀取原始音訊頻譜
    n_classes: int = 1        # 輸出通道: 1 代表模型輸出旋律遮罩

    # 4. 損失函數權重 (AudioSeparationLoss)
    alpha_l1: float = 3.0
    alpha_spectral: float = 2.0
    alpha_sisdr: float = 5.0 
    alpha_similarity: float = 5.0 # 伴奏/旋律互斥相似度權重 (處罰旋律中的伴奏殘留)
    melody_weight: float = 2.5    # 旋律擬合權重 (現在是主要目標)
    accomp_weight: float = 1.0    # 被扣除伴奏部分的參考權重

    # 5. 其他設定
    print_freq: int = 20      # 每 20 個 batch 輸出一次進度
    num_workers: int = 4      # DataLoader 的並行執行數
    pin_memory: bool = True    # 加速 GPU 記憶體傳輸
    drop_last: bool = True     # 捨棄最後一個不完整的 batch


def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Train Melody Separation Model')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--run_epochs', type=int, default=None)
    parser.add_argument('--t_0', type=int, default=None)
    parser.add_argument('--t_mult', type=int, default=None)
    parser.add_argument('--eta_min', type=float, default=None)
    args = parser.parse_args()

    # 初始化 MLflow
    mlflow.set_experiment('piano-mir-melody-separation')
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # 建立配置物件
    cfg = TrainConfig()
    
    # 使用命令行參數覆蓋配置
    if args.learning_rate is not None: cfg.learning_rate = args.learning_rate
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.run_epochs is not None: cfg.run_epochs = args.run_epochs
    if args.t_0 is not None: cfg.t_0 = args.t_0
    if args.t_mult is not None: cfg.t_mult = args.t_mult
    if args.eta_min is not None: cfg.eta_min = args.eta_min

    with mlflow.start_run(run_name='Melody-Separation-Training'):
        # 紀錄參數到 MLflow
        mlflow.log_params(asdict(cfg))

        # 解決某些環境下無法建立 torch kernel 快取目錄的問題
        os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
        if not os.path.exists(os.environ['PYTORCH_KERNEL_CACHE_PATH']):
            os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用裝置: {device}")

        if not os.path.exists(cfg.checkpoint_dir):
            os.makedirs(cfg.checkpoint_dir)

        # ===========================
        #    Dataset 與 DataLoader
        # ===========================
        print("正在初始化魯棒多任務資料集...")

        train_dataset = AudioDataset(csv_file=cfg.csv_file, split="train")
        val_dataset = AudioDataset(csv_file=cfg.csv_file, split="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last
        )

        # ===========================
        #        模型建置
        # ===========================
        model = AudioUNet(n_channels=cfg.n_channels, n_classes=cfg.n_classes).to(device)

        # 設定損失函數
        criterion = AudioSeparationLoss(
            alpha_l1=cfg.alpha_l1,
            alpha_spectral=cfg.alpha_spectral,
            alpha_sisdr=cfg.alpha_sisdr,
            alpha_similarity=cfg.alpha_similarity,
            melody_weight=cfg.melody_weight,
            accomp_weight=cfg.accomp_weight
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        # 整合預熱機制
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.t_0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[cfg.warmup_epochs]
        )

        # ===========================
        #      斷點續訓邏輯
        # ===========================
        start_epoch = 0
        best_val_loss = float('inf')
        train_history, val_history = [], []

        if cfg.resume_best and cfg.resume_from and os.path.exists(cfg.resume_from):
            print(f"[UPDATE] 發現存檔，正在載入: {cfg.resume_from}")
            try:
                checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                train_history = checkpoint.get('train_history', [])
                val_history = checkpoint.get('val_history', [])
                print(f"[SUCCESS] 載入成功！目前進度: 第 {start_epoch} 輪")
            except Exception as e:
                print(f"[ERROR] 載入存檔失敗: {e}，將從頭開始。")
        elif not cfg.resume_best:
            print("[SKIP] 已設定不繼承權重，將從頭開始訓練。")

        end_epoch = start_epoch + cfg.run_epochs
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
                    print(f"[WARN] 警告: 第 {epoch+1} 輪 Batch {batch_idx} 偵測到 NaN Loss，正在跳過...")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_accum += loss.item()
                
                if batch_idx % cfg.print_freq == 0:
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

            # --- Report Metrics to MLflow ---
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)

            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch + 1)
            scheduler.step()

            duration = time.time() - start_time
            print(f"==> Epoch {epoch + 1} | Time: {duration:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            plt.figure(figsize=(10, 5))
            plt.plot(train_history, label='Train Loss')
            plt.plot(val_history, label='Val Loss')
            plt.legend(); plt.grid(True); plt.savefig('loss_curve_mel.png'); plt.close()

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
                }, os.path.join(cfg.checkpoint_dir, "best_model.pth"))
                print("Best Model Saved!")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_history': train_history,
                    'val_history': val_history
                }, os.path.join(cfg.checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

        print("="*60)
        print(f"[DONE] 任務完成！目前總進度: {epoch + 1} 輪。")

if __name__ == "__main__":
    main()
