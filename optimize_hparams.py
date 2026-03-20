import argparse
from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna

def main():
    # 1. 設定命令列參數解析
    parser = argparse.ArgumentParser(description='ClearML Hyperparameter Optimization for Piano-MIR')
    parser.add_argument('--task', type=str, choices=['melody', 'accomp'], required=True,
                        help='要調優的任務類型: melody 或 accomp')
    parser.add_argument('--jobs', type=int, default=10, help='總共嘗試的參數組合數量 (預設: 10)')
    parser.add_argument('--concurrent', type=int, default=1, help='同時執行的實驗數量 (預設: 1)')
    parser.add_argument('--epochs', type=int, default=10, help='每個實驗跑幾個 Epoch (預設: 10)')
    args = parser.parse_args()

    # 2. 根據參數決定任務名稱與基底任務
    if args.task == 'melody':
        target_name = 'Melody-Separation-Training'
        hpo_task_name = 'HPO - Melody Task'
    else:
        target_name = 'Accomp-Separation-Training'
        hpo_task_name = 'HPO - Accomp Task'

    # 3. 初始化調優管理任務
    task = Task.init(
        project_name='piano-mir-melody-separation',
        task_name='HPO-Controller',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    # 4. 找到對應的基底任務 ID
    # 增加篩選條件：只抓取已完成或正在執行的，避免抓到空的失敗任務
    target_tasks = Task.get_tasks(
        project_name='piano-mir-melody-separation',
        task_name=target_name,
        task_filter={'status': ['completed', 'published', 'stopped', 'in_progress']}
    )

    if not target_tasks:
        print(f" 找不到任何有效的基底任務: {target_name}。請先手動執行一次 {target_name} 並確保它成功回報指標。")
        return

    # 抓取最新的那一個
    target_task = target_tasks[-1]
    base_task_id = target_task.id
    print(f"找到基底任務 ID: {base_task_id} (名稱: {target_name}, 狀態: {target_task.status})")

    # 檢查參數是否存在於該任務 (Debug 用)
    params = list(target_task.get_parameters().keys())
    print(f" 該任務偵測到的參數路徑範例: {params[:5] if params else '無'}")

     
    hyper_parameters = [
        # 核心訓練參數
        UniformParameterRange('General/learning_rate', min_value=4e-6, max_value=2e-4),
        UniformParameterRange('General/weight_decay', min_value=1e-6, max_value=1e-3),
        DiscreteParameterRange('General/batch_size', values=[8, 12, 16]),
        UniformParameterRange('General/eta_min', min_value=5e-7, max_value=5e-6),
        DiscreteParameterRange('General/t_0', values=[5, 10, 15]),
        DiscreteParameterRange('General/t_mult', values=[1, 2, 3]),
    ]

    # 6. 設定優化器
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=hyper_parameters,
        objective_metric_title='Loss',
        objective_metric_series='Val',
        objective_metric_sign='min',
        optimizer_class=OptimizerOptuna,
        execution_queue='default',
        max_number_of_concurrent_tasks=args.concurrent,
        total_max_jobs=args.jobs,
        # 重點修正：必須指定每個任務跑幾次迭代
        max_iteration_per_job=args.epochs,
    )

    # 7. 開始執行
    print(f" 開始進行 {args.task} 的超參數調優 (每個 Job 跑 {args.epochs} Epochs)...")
    optimizer.start()
    optimizer.wait()
    optimizer.stop()
    print(f" {args.task} 的調優已完成！")


if __name__ == "__main__":
    main()
