import argparse
import mlflow
import optuna
import subprocess
import os
import sys

def objective(trial, args):
    # 1. 定義超參數採樣範圍
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 12])
    eta_min = trial.suggest_float("eta_min", 1e-6, 4e-6, log=True)
    t_0 = trial.suggest_categorical("t_0", [5, 10])
    t_mult = trial.suggest_categorical("t_mult", [1])

    # 2. 準備執行指令
    script_path = os.path.join(
        "separation_mel_" if args.task == 'melody' else "separation_accomp_",
        f"train_{args.task[:3]}.py"
    )
    
    cmd = [
        sys.executable, script_path,
        "--learning_rate", str(lr),
        "--batch_size", str(batch_size),
        "--eta_min", str(eta_min),
        "--t_0", str(t_0),
        "--t_mult", str(t_mult),
        "--run_epochs", str(args.epochs)
    ]

    print(f"\n[Trial {trial.number}] 正在執行: {' '.join(cmd)}")

    # 3. 開啟 MLflow Nested Run 來紀錄這個 Trial
    with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        
        # 執行訓練
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Error] Trial {trial.number} 失敗:\n{result.stderr}")
            return float('inf')

        # 4. 解析最後回傳的指標 (從訓練腳本的存檔中讀取最新的 val_loss)
        # 這裡為了簡單起見，我們假設訓練腳本會將結果存入一個約定的地方，
        # 或者我們可以從 stdout 解析最後一行的 Val Loss。
        try:
            # 簡單解析 stdout 的最後一行 "Val: 0.XXXX"
            for line in reversed(result.stdout.splitlines()):
                if "Val:" in line:
                    val_loss = float(line.split("Val:")[1].split("|")[0].strip())
                    mlflow.log_metric("final_val_loss", val_loss)
                    return val_loss
        except Exception as e:
            print(f"[Error] 無法解析 Trial {trial.number} 的指標: {e}")
            
        return float('inf')

def main():
    parser = argparse.ArgumentParser(description='MLflow + Optuna Hyperparameter Optimization')
    parser.add_argument('--task', type=str, choices=['melody', 'accomp'], required=True)
    parser.add_argument('--jobs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    # 初始化 MLflow Experiment
    mlflow.set_experiment('piano-mir-melody-separation')

    # 建立一個 Parent Run 來包裹整個 HPO 過程
    with mlflow.start_run(run_name=f"HPO-{args.task.capitalize()}"):
        mlflow.log_param("task_type", args.task)
        mlflow.log_param("total_jobs", args.jobs)
        
        # 建立 Optuna Study
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.jobs)

        print("\n" + "="*40)
        print(f"調優完成！最佳參數: {study.best_params}")
        print(f"最佳 Val Loss: {study.best_value}")
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

if __name__ == "__main__":
    main()
