# Piano Melody and Accompaniment Separation (Piano-MIR)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

這是一個專注於鋼琴音訊的旋律與伴奏分離專案。利用深度學習模型 (U-Net)
從混合的鋼琴音訊中提取出純旋律 (Melody) 與純伴奏 (Accompaniment) 音軌。

## 📌 專案特色
  - 專屬鋼琴優化：針對鋼琴音色的頻譜特性進行訓練。
  - 雙任務架構：提供獨立的旋律與伴奏分離模型（Mel-Task 與 Acc-Task）。
  - 完整流水線：包含 MIDI 渲染、資料預處理、模型訓練、自動化推論與效能評估。
  - 多維度評估：整合 mir_eval 與 SI-SDR 等專業音訊指標。

📂 建議專案架構
為了保持專案整潔，建議您將檔案手動整理如下：

  1 .
  2 ├── separation_mel_/         # 旋律分離任務核心程式碼
  3 │   ├── utils_melody/        # 模型定義 (U-Net)、Dataset
  4 │   └── tools/               # 資料處理工具 (MIDI to Audio, STFT)
  5 ├── separation_accomp_/      # 伴奏分離任務核心程式碼
  6 │   ├── utils_accomp/        # 模型定義、Dataset
  7 │   └── tools/               # 資料處理工具
  8 ├── data_info/               # 存放資料集清單 (手動移動: dataset.csv, bad_files.txt
    等)
  9 ├── results/                 # 存放輸出結果
  10 │   ├── plots/               # 視覺化圖表 (手動移動: *.png)
  11 │   └── reports/             # 效能報表 (手動移動: *.csv, *.xlsx)
  12 ├── predict_all.py           # 整合推論腳本
  13 ├── analyze_latest.py        # 效能分析工具
  14 ├── INSTALL.md               # 詳細環境安裝指南
  15 └── README.md                # 專案說明文件

## 🚀 快速上手

1. 環境安裝
請參考 INSTALL.md (INSTALL.md) 完成環境配置。
  1 pip install -r requirements.txt

2. 資料預處理
如果您有 MIDI 檔案，可以使用以下腳本渲染為 FLAC/WAV：
  1 python separation_mel_/tools/sf2_midi_to_flac.py

3. 模型訓練
分別針對旋律或伴奏進行訓練：

  1 python separation_mel_/train_mel.py
  2 python separation_accomp_/train_accomp.py

4. 執行推論 (分離音訊)
使用 predict_all.py 進行整合推論：
  1 python predict_all.py

## 🧠 技術細節

模型架構
  - Backbone: U-Net 架構。
  - Encoder/Decoder: 包含多層卷積與跳躍連接 (Skip Connections)，並在深層加入 Dropout
    防止過擬合。
  - Input: 短時傅立葉轉換 (STFT) 產出的頻譜圖 (Spectrogram)。

## 損失函數 (Loss Functions)
模型結合了三種損失函數以確保音質與信號準確度：
  1. L1 Loss: 基礎頻譜誤差。
  2. Multi-Scale Spectral Loss: 捕捉不同時間尺度的頻譜特徵。
  3. SI-SDR Loss: 優化時間域的信號失真比。

## 📊 效能評估
本專案使用以下指標進行驗證：
  - SDR (Source-to-Distortion Ratio)
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - SIR (Source-to-Interference Ratio)
  - SAR (Source-to-Artifacts Ratio)

評估結果與圖表將自動產出於 results/ 資料夾下。
