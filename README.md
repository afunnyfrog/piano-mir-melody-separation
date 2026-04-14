# Piano Melody and Accompaniment Separation (Piano-MIR)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

這是一個基於深度學習 (U-Net) 的鋼琴旋律與伴奏分離專案。透過訓練專屬模型，能將混合的鋼琴音訊精準拆解為純旋律與純伴奏兩條音軌。

## 📂 專案結構說明
專案已完成模組化整理，各目錄功能如下：

*   **`run_separation.py`**: **主程式入口**，用於對整首曲目執行旋律與伴奏的分離。
*   **`separation_mel_/` & `separation_accomp_/`**: 核心模型目錄，包含各自的訓練腳本 (`train_*.py`) 與模型架構定義。
*   **`tools/`**: 各種預處理工具。
    *   `render_midi_to_*.py`: 將 MIDI 渲染為音訊。
    *   `manifest_generator.py`: 產生訓練所需的 CSV 清單。
    *   `check_audio_integrity.py`: 檢查資料集音訊是否損毀。
    *   `visualize_prediction.py`: 視覺化模型的推論頻譜。
*   **`evaluation/`**: 效能評估與分析。
    *   `one_predict_test.py`: 隨機抽樣一首歌曲進行快速分離測試。
    *   `one_analyze_test.py`: 自動評估最新產出的分離結果 (SDR, SI-SDR 指標)。
    *   `batch_*.py`: 批次執行推論與分析。
    *   `separation_boxplot.py`: 產生指標分布盒鬚圖。
*   **`mlflow_training_tools/`**: 模型開發進階工具，包含超參數優化 (`optimize_hparams.py`) 與訓練歷程記錄。
*   **`data_info/`**: 存放資料集清單 (`dataset.csv`) 與錯誤報告。
*   **`input/`**: 預設存放待分離的原始音訊檔案。
*   **`results/`**: 分離後的音訊檔案、視覺化圖表與效能報表。

## 🚀 快速上手

### 1. 環境配置
詳細步驟請參考 [INSTALL.md](INSTALL.md)。
```bash
pip install -r requirements.txt
```

### 2. 執行分離 (主程式)
將您的鋼琴音訊放入 `input/` 資料夾，執行：
```bash
python run_separation.py
```

### 3. 快速測試與分析
若要隨機選取資料集中的一首歌並即時查看分離數據：
```bash
# 1. 執行分離測試
python evaluation/one_predict_test.py

# 2. 自動分析該次分離結果
python evaluation/one_analyze_test.py
```

## 📥 預訓練模型 (Pre-trained Models)

為了方便快速使用，我們提供了訓練好的模型權重。請下載後將其放入對應的資料夾中：

| 模型功能 | 下載連結 | 建議存放路徑 |
| :--- | :--- | :--- |
| **旋律分離 (Melody)** | [點此下載 (Google Drive)](https://drive.google.com/file/d/1w9jpWapVts-igm3DtdciZuM9PWBk6_--/view?usp=sharing) | `best_path/best_mel_model.pth` |
| **伴奏分離 (Accomp)** | [點此下載 (Google Drive)](https://drive.google.com/file/d/1Z0gwlniOVEL8wZHe_67rd9ROAkq5SReM/view?usp=drive_link) | `best_path/best_acc_model.pth` |

---

## 🧠 技術細節

### 引用資料集 (Dataset Citation)
本專案訓練所採用的核心數據集為 **ADL Piano MIDI**。該數據集包含 11,086 首涵蓋多種風格的鋼琴作品，來源於 Lakh MIDI 數據集及網路公開資源。

如果您使用本專案或相關數據，請引用以下論文：

```bibtex
@article{ferreira_aiide_2020,
  title={Computer-Generated Music for Tabletop Role-Playing Games},
  author={Ferreira, Lucas N and Lelis, Levi HS and Whitehead, Jim},
  booktitle = {Proceedings of the 16th AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  series = {AIIDE'20},
  year={2020},
}
```

### 模型架構
- **Backbone**: U-Net 架構，採用多層卷積與跳躍連接。
- **Input**: 短時傅立葉轉換 (STFT) 頻譜圖 (Log-magnitude Spectrogram)。
- **Loss**: 結合了 **L1 Loss**、**Multi-Scale Spectral Loss** 與 **SI-SDR Loss**，確保音質與訊號還原度。

### 效能指標 (Evaluation Metrics)
我們使用 `mir_eval` 標準進行量化分析：
- **SDR** (Source-to-Distortion Ratio)
- **SI-SDR** (Scale-Invariant SDR)
- **SIR** (Source-to-Interference Ratio)
- **SAR** (Source-to-Artifacts Ratio)

## 📊 開發日誌與權重
- 本專案整合了 **MLflow** 進行實驗管理，您可以透過 `mlflow_training_tools/` 追蹤訓練過程。
- 最佳模型權重建議存放於 `checkpoints/` 資料夾。

---
*本專案由 AI 輔助整理與優化。*
