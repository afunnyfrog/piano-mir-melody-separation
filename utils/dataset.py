import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import random
import soundfile as sf
import pandas as pd

# --- 設定參數 ---
SAMPLE_RATE = 44100
DURATION = 3.0
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
SPEC_SHAPE = (1, 1024, 256) 



class AudioDataset(Dataset):
    def __init__(self, csv_file, split="train"):
        self.df = pd.read_csv(csv_file)
        
        self.data_list = self.df[self.df['split_type'] == split]['file_path'].tolist() # <--- 修改這裡：只拿對應的檔案
        
        if len(self.data_list) == 0:
            print(f"警告：在 {csv_file} 中找不到任何 {split} 的資料！")
        else:
            print(f"成功載入 {split} 資料集，共 {len(self.data_list)} 筆。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.load_with_retry(idx, retry_count=0)

    def _log_error(self, error_msg):
        log_file = "bad_files.txt"
        existing_content = set()
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    existing_content = set(line.strip() for line in f)
            except: pass

        if error_msg not in existing_content:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(error_msg + "\n")
            except: pass

    # --- 新增：預先檢查函式 ---
    def _validate_file(self, path):
        """
        使用 soundfile 快速檢查檔案是否損壞，防止 librosa 卡死
        """
        try:
            with sf.SoundFile(path) as f:
                # 檢查 1: 檔案是否能開啟
                if not f.seekable():
                    raise ValueError("檔案不支援 Seek (Not Seekable)")
                
                # 檢查 2: 嘗試跳到檔案末端 (這是抓出 psf_fseek failed 的關鍵)
                f.seek(f.frames - 1)
                f.read(1)
        except Exception as e:
            raise ValueError(f"檔案結構損壞 (SoundFile Check Failed): {e}")

    def load_with_retry(self, idx, retry_count):
        # 2. 修改路徑取得邏輯
        # CSV 裡面的 file_path 已經是完整路徑 (例如: /data/mix_audio_flac/song1_mixed.flac)
        orig_path = self.data_list[idx] # <--- 修改這裡：直接從 list 拿路徑
        fname = os.path.basename(orig_path) # 取得檔名用來報錯顯示
        
        if retry_count > 5:
            print(f"❌ 放棄檔案 {fname}")
            return torch.zeros(SPEC_SHAPE), torch.zeros((2, 1024, 256))

        # 3. 推算 Melody 和 Accomp 的路徑
        # 資料夾結構是平行的：
        try:
            # 使用字串取代來切換資料夾和檔名後綴
            # <--- 修改這裡：路徑推算邏輯
            mel_path = orig_path.replace('mix_audio_flac', 'melody_audio_flac') \
                                .replace('_mixed.flac', '_melody.flac')
            
            acc_path = orig_path.replace('mix_audio_flac', 'accomp_audio_flac') \
                                .replace('_mixed.flac', '_accomp.flac')

            # --- 以下邏輯保持不變 ---
            # 檢查檔案
            self._validate_file(orig_path)
            self._validate_file(mel_path)
            self._validate_file(acc_path)

            # 1. 取得長度
            total_duration = librosa.get_duration(path=orig_path)
            if total_duration == 0: raise ValueError("音訊長度為 0")

            # [訓練策略] 訓練集隨機切，驗證集通常建議固定切(這裡先維持隨機沒關係，之後可優化)
            if total_duration > DURATION:
                start_time = np.random.uniform(0, total_duration - DURATION)
            else:
                start_time = 0

            # 2. 載入音訊
            wav_orig, _ = librosa.load(orig_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
            wav_mel, _ = librosa.load(mel_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
            wav_acc, _ = librosa.load(acc_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)

            # 3. 補零或裁切 (Fix length)
            if len(wav_orig) < CHUNK_SIZE:
                wav_orig = librosa.util.fix_length(wav_orig, size=CHUNK_SIZE)
                wav_mel = librosa.util.fix_length(wav_mel, size=CHUNK_SIZE)
                wav_acc = librosa.util.fix_length(wav_acc, size=CHUNK_SIZE)
            else:
                wav_orig = wav_orig[:CHUNK_SIZE]
                wav_mel = wav_mel[:CHUNK_SIZE]
                wav_acc = wav_acc[:CHUNK_SIZE]

            # 4. 轉頻譜圖
            spec_orig = self.wav_to_spec(wav_orig)
            spec_mel = self.wav_to_spec(wav_mel)
            spec_acc = self.wav_to_spec(wav_acc)
            
            target = torch.cat([spec_mel, spec_acc], dim=0) 
            
            return spec_orig, target

        except Exception as e:
            error_msg = f"{fname} | Error: {str(e)}"
            self._log_error(error_msg)
            
            # 隨機換下一首嘗試
            new_idx = random.randint(0, len(self.data_list) - 1)
            return self.load_with_retry(new_idx, retry_count + 1)

    def wav_to_spec(self, wav):
        stft = librosa.stft(wav, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        log_spec = np.log(magnitude + 1e-6)
        log_spec = log_spec[:1024, :256]
        
        if log_spec.shape[1] < 256:
            pad_width = 256 - log_spec.shape[1]
            log_spec = np.pad(log_spec, ((0,0), (0, pad_width)))

        min_val = log_spec.min()
        max_val = log_spec.max()
        div = (max_val - min_val)
        if div == 0: div = 1e-6
        norm_spec = (log_spec - min_val) / (div + 1e-6)
        tensor_spec = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
        return tensor_spec