import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import optuna
from tqdm import tqdm

# 引入自定義模組
from tools.turn_STFT_dataset import AudioDataset
from utils.u_net import AudioUNet
from utils.loss import AudioSeparationLoss

# ==========================================
#               固定參數設定
# ==========================================
CSV_FILE = "classical_dataset.csv" 
BASE_CHECKPOINT_DIR = "./optuna_wage_studies"
RUN_EPOCHS = 40  # 依照原腳本設定
WARMUP_EPOCHS = 10
T_0 = 10
T_MULT = 1
ETA_MIN = 1e-6
N_CHANNELS = 1
N_CLASSES = 1

def objective(trial):
    # --- 1. 定義超參數搜尋空間 ---
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    alpha_l1 = trial.suggest_float("alpha_l1", 1.0, 5.0)
    alpha_spectral = trial.suggest_float("alpha_spectral", 1.0, 5.0)
    alpha_sisdr = trial.suggest_float("alpha_sisdr", 1.0, 10.0)
    alpha_similarity = trial.suggest_float("alpha_similarity", 1.0, 10.0)
    melody_weight = trial.suggest_float("melody_weight", 5.0, 15.0)
    
    batch_size = trial.suggest_categorical("batch_size", [8, 16])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_dir = os.path.join(BASE_CHECKPOINT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # --- 2. 初始化模型與 Loss ---
    model = AudioUNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    criterion = AudioSeparationLoss(
        alpha_l1=alpha_l1,
        alpha_spectral=alpha_spectral,
        alpha_sisdr=alpha_sisdr,
        alpha_similarity=alpha_similarity,
        melody_weight=melody_weight,
        accomp_weight=1.0 # 固定為 1.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 學習率排程器
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

    # --- 3. 資料載入 ---
    train_dataset = AudioDataset(csv_file=CSV_FILE, split="train")
    val_dataset = AudioDataset(csv_file=CSV_FILE, split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    best_val_loss = float('inf')

    # --- 4. 訓練迴圈 ---
    for epoch in range(RUN_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{RUN_EPOCHS} [Train]", leave=False)
        for waveforms, targets, target_audios in train_pbar:
            waveforms, targets, target_audios = waveforms.to(device), targets.to(device), target_audios.to(device)
            optimizer.zero_grad()
            
            predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
            
            # 對齊維度
            if predictions.shape[3] != targets.shape[3]:
                min_time = min(predictions.shape[3], targets.shape[3])
                predictions, targets, pred_others = predictions[:,:,:,:min_time], targets[:,:,:,:min_time], pred_others[:,:,:,:min_time]

            loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
            
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 5. 驗證階段 ---
        model.eval()
        val_loss_accum = 0.0
        val_pbar = tqdm(val_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{RUN_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for waveforms, targets, target_audios in val_pbar:
                waveforms, targets, target_audios = waveforms.to(device), targets.to(device), target_audios.to(device)
                predictions, pred_audios, pred_others = model(waveforms, return_audio=True)
                
                if predictions.shape[3] != targets.shape[3]:
                    min_time = min(predictions.shape[3], targets.shape[3])
                    predictions, targets, pred_others = predictions[:,:,:,:min_time], targets[:,:,:,:min_time], pred_others[:,:,:,:min_time]
                    
                loss, _ = criterion(predictions, targets, pred_audio=pred_audios, target_audio=target_audios, pred_other=pred_others)
                if not torch.isnan(loss): 
                    val_loss_accum += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step()
        print(f"[Trial {trial.number} Epoch {epoch+1}] Train Loss: {train_loss_accum/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pth"))

        # 向 Optuna 回報並檢查剪枝
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    study_name = "wage_melody_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    print("🚀 開始 wage code 貝氏優化...")
    study.optimize(objective, n_trials=30)

    print("\n🏆 優化完成！最佳參數:", study.best_params)
