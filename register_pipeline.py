from clearml import Task
from clearml.automation import PipelineController

def main():
    # 統一使用你在 train_accomp.py 中設定的專案名稱
    PROJECT_NAME = 'piano-mir-melody-separation'
    
    # 1. 初始化 Pipeline 控制器
    pipe = PipelineController(
        name='Piano-MIR-HPO-Pipeline',
        project=PROJECT_NAME,
        version='2.1.0'
    )

    pipe.add_step(
        name='hpo_accomp',
        base_task_project=PROJECT_NAME,
        base_task_name='HPO-Controller', # 指向你的 HPO 腳本任務
        execution_queue='default',       # 明確指定隊列
        parameter_override={
            'Args/task': 'accomp',
            'Args/jobs': 5,
            'Args/epochs': 5
        }
    )

    pipe.add_step(
        name='hpo_melody',
        parents=['hpo_accomp'],
        base_task_project=PROJECT_NAME,
        base_task_name='HPO-Controller',
        execution_queue='default',       # 明確指定隊列
        parameter_override={
            'Args/task': 'melody',
            'Args/jobs': 5,
            'Args/epochs': 5
        }
    )

    # 4. 啟動 Pipeline
    # 這會讓整個流程自動化：先調優伴奏 -> 再調優旋律
    pipe.start_local()
    
    print(f"🚀 流水線已在專案 '{PROJECT_NAME}' 中啟動！")
    
    # 等待完成
    pipe.wait()
    pipe.stop()

if __name__ == "__main__":
    main()
