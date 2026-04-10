import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 讀取資料
df = pd.read_csv('batch_test_performance_report.csv')

# 2. 資料清理
# 雖然修正了 SIR，但保險起見還是將殘留的 inf 替換為 NaN 避免繪圖崩潰
df_clean = df.replace([np.inf, -np.inf], np.nan)

# 3. 準備繪圖資料 (包含全部四個核心指標)
metrics_to_plot = ['SDR', 'SI-SDR', 'SIR', 'SAR']
df_melted = df_clean.melt(id_vars=['Part'], value_vars=metrics_to_plot, 
                         var_name='Metric', value_name='Decibels (dB)')

# 4. 繪圖設定
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# 畫出分組箱型圖
ax = sns.boxplot(x='Metric', y='Decibels (dB)', hue='Part', data=df_melted, 
                 palette={'Melody': 'skyblue', 'Accomp': 'salmon'},
                 width=0.7, fliersize=4)

# 5. 優化圖表外觀
plt.title('Separation Performance Distribution (Test Set - 80 Epochs)', fontsize=16, fontweight='bold')
plt.xlabel('Evaluation Metrics', fontsize=12)
plt.ylabel('Score (dB)', fontsize=12)
plt.legend(title='Audio Part', loc='upper right')

# 加上水平網格線輔助閱讀
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 6. 儲存圖片
save_name = 'separation_boxplot_full_metrics.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"✅ 箱型圖已生成：{save_name}")