# piano-mir-melody-separation

A deep learning model to separate piano melody from accompaniment tracks.
一個用來分離鋼琴旋律與伴奏音軌的深度學習模型

## 安裝環境

請參考 [INSTALL.md](INSTALL.md)

## 程式介紹

* 訓練模型
  * `train.py`：負責設定訓練參數、學習率參數、資料夾位址以及重複訓練
  * `u-net.py`：基本u-net架構，並在深層加入drop out
  * `loss.py`：特製損失函數，使用`L1 loss`、`Multi-Scale Spectral Loss`、`SI-SDR Loss`來計算損失。在訓練以及驗證時使用
* 音訊資料處理
  * `sf2_midi_to_flac.py`：使用sf2音色庫，將midi檔案渲染成—無損壓縮音訊檔案—.flac。
  * `sf2_midi_to_wav.py`：使用sf2音色庫，將midi檔案渲染成音訊檔案.wav。
* 訓練資料處理
  * `generate_manifest.py`：負責將資料集切分，並使用 `data.csv`儲存以利於程式讀取。
  * `dataset.py`：在曲目時長內，隨機切分固定時長音訊，並且透過短時傅立葉轉換(STFT)將音訊轉為頻譜圖提供訓練
* 模型權重測試
  * `predict.py`：透過模型所產出的權重與模型，將完整的音檔輸入，並且產出旋律與伴奏音檔
* 其他
  * `test_audio_file.py`：檢查音訊檔案是否完整，避免有損壞，檢查`flac`檔  
  * `test.py`：區分wav以及flac檔案經過頻譜圖比對差異
  * `debug_visualize.py`：於`predict.py`前快速檢視模型輸出行為。
  * `clean_dataset.py`：同為檢查音訊檔案是否完整，檢查`wav`檔
  