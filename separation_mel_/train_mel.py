import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import mlflow
import mlflow.pytorch
import argparse
import sys
import warnings
import logging

# 屏蔽 MLflow 和相關庫的警告日誌
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.*")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# 引入自定義模組
from tools.turn_STFT_dataset import AudioDataset
from utils_melody.u_net import AudioUNet
from utils_melody.loss import AudioSeparationLoss

@dataclass
class TrainConfig:
    csv_file: str = "classical_dataset.csv" 
    checkpoint_dir: str = "./checkpoints_mel_task"
    resume_best: bool = False  
    resume_from: str = "./checkpoints_mel_task/best_model.pth"
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    run_epochs: int = 80
    warmup_epochs: int = 10      
    t_0: int = 10               
    t_mult: int = 1             
    eta_min: float = 1e-6       
    n_channels: int = 1       
    n_classes: int = 1        
    alpha_l1: float = 3.0
    alpha_spectral: float = 2.0
    alpha_sisdr: float = 5.0 
    alpha_similarity: float = 5.0 
    melody_weight: float = 2.5    
    accomp_weight: float = 1.0    
    print_freq: int = 20      
    num_workers: int = 4      
    pin_memory: bool = True    
    drop_last: bool = True     


def main():
    parser = argparse.ArgumentParser(description='Train Melody Separation Model')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--run_epochs', type=int, default=None)
    parser.add_argument('--t_0', type=int, default=None)
    parser.add_argument('--t_mult', type=int, default=None)
    parser.add_argument('--eta_min', type=float, default=None)
    args = parser.parse_args()

    mlflow.set_experiment('piano-mir-melody-separation')
    cfg = TrainConfig()
    
    if args.learning_rate is not None: cfg.learning_rate = args.learning_rate
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.run_epochs is not None: cfg.run_epochs = args.run_epochs
    if args.t_0 is not None: cfg.t_0 = args.t_0
    if args.t_mult is not None: cfg.t_mult = args.t_mult
    if args.eta_min is not None: cfg.eta_min = args.eta_min

    with mlflow.start_run(run_name='Melody-Separation-Training') as run:
        mlflow.log_params(asdict(cfg))
        mlflow.set_tag("Task Type", "Melody Separation")
        mlflow.set_tag("Train Mode", "Normal" if not cfg.resume_best else "Resume")

        os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.getcwd(), '.torch_kernel_cache')
        os.makedirs(os.environ['PYTORCH_KERNEL_CACHE_PATH'], exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用裝置: {device}")

        if not os.path.exists(cfg.checkpoint_dir):
            os.makedirs(cfg.checkpoint_dir)

        train_dataset = AudioDataset(csv_file=cfg.csv_file, split="train")
        val_dataset = AudioDataset(csv_file=cfg.csv_file, split="val")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last)

        sample_batch = next(iter(val_loader))
        input_example = sample_batch[0][0:1].cpu().numpy()

        model = AudioUNet(n_channels=cfg.n_channels, n_classes=cfg.n_classes).to(device)
        criterion = AudioSeparationLoss(alpha_l1=cfg.alpha_l1, alpha_spectral=cfg.alpha_spectral, alpha_sisdr=cfg.alpha_sisdr, alpha_similarity=cfg.alpha_similarity, melody_weight=cfg.melody_weight, accomp_weight=cfg.accomp_weight).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs)
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.t_0, T_mult=cfg.t_mult, eta_min=cfg.eta_min)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[cfg.warmup_epochs])

        start_epoch = 0
        best_val_loss = float('inf')
        train_history, val_history = [], []

        if cfg.resume_best and cfg.resume_from and os.path.exists(cfg.resume_from):
            try:
                checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']; best_val_loss = checkpoint['best_val_loss']
                train_history = checkpoint.get('train_history', []); val_history = checkpoint.get('val_history', [])
                print(f"[SUCCESS] 載入成功！第 {start_epoch} 輪")
            except Exception as e: print(f"[ERROR] 載入失敗: {e}")

        print("-" * 40)
        
        try:
            for epoch in range(start_epoch, start_epoch + cfg.run_epochs):
                current_run_status = mlflow.get_run(run.info.run_id).info.status
                if current_run_status in ["KILLED", "FINISHED"]:
                    print(f"\n[INFO] MLflow 狀態變更為 {current_run_status}，正在停止訓練...")
                    break
                
                start_time = time.time()
                model.train()
                train_loss_accum = 0

                for batch_idx, (waveforms, targets, target_audios) in enumerate(train_loader):
                    waveforms, targets, target_audios = waveforms.to(device), targets.to(device), target_audios.to(device)
                    optimizer.zero_grad()
                    predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
                    if predictions.shape[3] != targets.shape[3]:
                        min_time = min(predictions.shape[3], targets.shape[3])
                        predictions, targets, pred_others = predictions[:, :, :, :min_time], targets[:, :, :, :min_time], pred_others[:, :, :, :min_time]

                    loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
                    if torch.isnan(loss): continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss_accum += loss.item()

                avg_train_loss = train_loss_accum / len(train_loader)
                train_history.append(avg_train_loss)

                model.eval()
                val_loss_accum = 0
                with torch.no_grad():
                    for waveforms, targets, target_audios in val_loader:
                        waveforms, targets, target_audios = waveforms.to(device), targets.to(device), target_audios.to(device)
                        predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
                        if predictions.shape[3] != targets.shape[3]:
                            min_time = min(predictions.shape[3], targets.shape[3])
                            predictions, targets, pred_others = predictions[:, :, :, :min_time], targets[:, :, :, :min_time], pred_others[:, :, :, :min_time]
                        loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
                        if not torch.isnan(loss): val_loss_accum += loss.item()

                avg_val_loss = val_loss_accum / len(val_loader)
                val_history.append(avg_val_loss)

                mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", current_lr, step=epoch + 1)
                scheduler.step()

                print(f"==> Epoch {epoch + 1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
                    torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss}, best_model_path)
                    
                    model.cpu()
                    # 顯式指定 serialization_format="pickle" 並屏蔽日誌來去除警告
                    mlflow.pytorch.log_model(
                        model, 
                        name="best_model", 
                        registered_model_name="Piano-Melody-Separation", 
                        input_example=input_example,
                        serialization_format="pickle"
                    )
                    model.to(device)
                    print(" [SAVED] Best Model Logged to MLflow!")

        except KeyboardInterrupt:
            print("\n" + "!"*30)
            print("偵測到 Ctrl+C！正在安全保存進度...")
            interrupted_path = os.path.join(cfg.checkpoint_dir, "interrupted_model.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, interrupted_path)
            model.cpu()
            mlflow.pytorch.log_model(
                model, 
                name="interrupted_model", 
                input_example=input_example,
                serialization_format="pickle"
            )
            print("中斷進度已上傳至 MLflow。")
            sys.exit(0)

    print("="*60 + "\n任務完成！")

if __name__ == "__main__":
    main()
