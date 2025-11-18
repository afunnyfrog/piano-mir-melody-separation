import tensorflow as tf
import librosa
import numpy as np
import glob
import os

# --- 設定參數 ---
SAMPLE_RATE = 44100
DURATION = 3.0      # 訓練時每次看 3 秒
CHUNK_SIZE = int(SAMPLE_RATE * DURATION) # 3秒對應的樣本點數 (44100 * 3)
BATCH_SIZE = 8      # 一次訓練幾筆資料

# --- 1. 讀取並隨機切割的函式 (核心邏輯) ---
def load_and_crop_wav(original_path, melody_path, accomp_path):
    """
    這就是你圖表中的 T(t) 隨機切分
    輸入是完整的檔案路徑，輸出是切好的 3 秒波形 (numpy array)
    """
    # 由於 librosa 讀取太慢，且不支援 TF graph，我們通常用 tf.audio 或 soundfile
    # 為了簡單起見，這裡演示邏輯。實際高效能訓練會用 tf.io.read_file + tf.audio.decode_wav
    
    # 這裡使用 NumPy wrapper 讓 librosa 可以跑在 TF dataset 裡
    def _crop_numpy(orig_str, mel_str, acc_str):
        # 轉回字串
        orig_f = orig_str.decode('utf-8')
        mel_f = mel_str.decode('utf-8')
        acc_f = acc_str.decode('utf-8')
        
        # 1. 取得音檔總長度 (不讀取整個檔案，只讀 header，速度快)
        total_duration = librosa.get_duration(path=orig_f)
        
        # 2. 隨機決定開始時間 (Random Start)
        if total_duration > DURATION:
            max_start = total_duration - DURATION
            start_time = np.random.uniform(0, max_start)
        else:
            start_time = 0

        # 3. 載入那 3 秒鐘 (Load Chunk)
        # librosa 支援 offset (開始時間) 和 duration (長度)
        X, _ = librosa.load(orig_f, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
        Y_mel, _ = librosa.load(mel_f, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
        Y_acc, _ = librosa.load(acc_f, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
        
        # 確保長度一致 (如果不足 3 秒要補 0)
        # ... (padding logic here if needed) ...
        
        # 簡單處理：如果長度不對直接丟棄或強制 resize (這裡略過細節)
        if len(X) != CHUNK_SIZE:
             # 邊界狀況處理...
             pass 

        return X, Y_mel, Y_acc

    # 使用 tf.numpy_function 包裝 python code
    X, Y_mel, Y_acc = tf.numpy_function(
        _crop_numpy, 
        [original_path, melody_path, accomp_path], 
        [tf.float32, tf.float32, tf.float32]
    )
    return X, Y_mel, Y_acc

# --- 2. 轉頻譜圖 (STFT) ---
def to_spectrogram(wave):
    # 轉 STFT
    stft = tf.signal.stft(wave, frame_length=2048, frame_step=512)
    # 取能量 (Magnitude)
    magnitude = tf.abs(stft)
    # 轉 Log Scale (這對模型比較好學) -> 類似 dB
    log_spectrogram = tf.math.log(magnitude + 1e-6)
    # 增加一個通道維度 (為了符合 CNN 輸入 [H, W, 1])
    return log_spectrogram[..., tf.newaxis]

def process_pipeline(orig_path, mel_path, acc_path):
    # A. 隨機切割
    wav_orig, wav_mel, wav_acc = load_and_crop_wav(orig_path, mel_path, acc_path)
    
    # B. 轉圖片
    spec_orig = to_spectrogram(wav_orig)
    spec_mel = to_spectrogram(wav_mel)
    spec_acc = to_spectrogram(wav_acc)
    
    # C. 正規化 (選用，將數值縮放到 0~1 或 -1~1)
    # ...
    
    # 回傳 (Input, Target)
    # Target 是一個 dict 或 tuple，對應模型的兩個輸出
    return spec_orig, (spec_mel, spec_acc)

# --- 3. 建立 Dataset 物件 (供油管線) ---
def get_dataset(data_dir):
    # 搜尋所有檔案
    orig_files = sorted(glob.glob(os.path.join(data_dir, "*_original.wav")))
    mel_files = sorted(glob.glob(os.path.join(data_dir, "*_melody.wav")))
    acc_files = sorted(glob.glob(os.path.join(data_dir, "*_accompaniment.wav")))

    # 建立 TensorFlow Dataset
    ds = tf.data.Dataset.from_tensor_slices((orig_files, mel_files, acc_files))
    
    # 隨機打亂 (Shuffle)
    ds = ds.shuffle(buffer_size=1000)
    
    # 套用處理流程 (Map) - 這裡會平行處理，加速讀取
    ds = ds.map(process_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 批次打包 (Batch)
    ds = ds.batch(BATCH_SIZE)
    
    # 預取 (Prefetch) - GPU 算的時候 CPU 預先準備下一批
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds