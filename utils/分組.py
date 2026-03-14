import torch
import os

# 設定您的存檔路徑
CHECKPOINT_PATH = r"C:\Users\cebit\Downloads\piano-mir-melody-separation-main\piano-mir-melody-separation-main\checkpoints_dynamic_8s\best_model.pth"
# 或者 "./checkpoints_dynamic_8s/best_model.pth"

if os.path.exists(CHECKPOINT_PATH):
    print(f"正在檢查: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    print("\n--- 存檔內的 Keys ---")
    print(checkpoint.keys())

    if 'scheduler_state_dict' in checkpoint:
        print("\n✅ 恭喜！這個存檔包含 Scheduler 狀態。")
        # 您甚至可以印出來看它現在的學習率設定
        print("內容預覽:", checkpoint['scheduler_state_dict'])
    else:
        print("\n⚠️ 警告：這個存檔裡面【沒有】Scheduler 狀態！")
else:
    print("❌ 找不到檔案，請確認路徑。")