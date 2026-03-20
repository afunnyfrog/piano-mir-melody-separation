import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import random
import soundfile as sf
import pandas as pd
import pretty_midi

# =================================================================
#                         核心參數設定區
# =================================================================
CONFIG = {
    # --- 音訊處理參數 ---
    "SAMPLE_RATE": 44100,
    "DURATION": 4.0,           
    "N_FFT": 2048,             
    "HOP_LENGTH": 512,         
    "TARGET_BINS": 1024,       

    # --- 魯棒性訓練設定 (Dropout) ---
    "PROB_DROP_MELODY_HINT": 0.5,  # 旋律提示丟棄機率
    "PROB_DROP_ACCOMP_HINT": 0.5,  # 伴奏提示丟棄機率

    # --- 路徑設定 ---
    "LOG_FILE": "bad_files.txt",

    # --- 資料夾名稱映射 ---
    "DIR_MIX": "mix_audio_flac",
    "DIR_MELODY": "melody_audio_flac",
    "DIR_ACCOMP": "accomp_audio_flac",
    
    "MIDI_SUB_MEL": "train_Y_melody",  
    "MIDI_SUB_ACC": "train_Y_accomp",  
    
    "EXT_MIX": "_mixed.flac",
    "EXT_MELODY": "_melody.flac",
    "EXT_ACCOMP": "_accomp.flac",
    "EXT_MIDI": ".mid"
}

CONFIG["CHUNK_SIZE"] = int(CONFIG["SAMPLE_RATE"] * CONFIG["DURATION"])
# =================================================================

class AudioDataset(Dataset):
    def __init__(self, csv_file, split="train"):
        self.split = split
        self.df = pd.read_csv(csv_file)
        self.data_list = self.df[self.df['split_type'] == split]['file_path'].tolist()
        
        if len(self.data_list) == 0:
            print(f"警告：在 {csv_file} 中找不到任何 {split} 的資料！")
        else:
            print(f"✅ 成功載入 [{split}] 資料集 (獨立通道多任務模式)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.load_with_retry(idx, retry_count=0)

    def _log_error(self, error_msg):
        if not os.path.exists(CONFIG["LOG_FILE"]):
            with open(CONFIG["LOG_FILE"], "w") as f: pass
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(f"{error_msg}\n")

    def _validate_file(self, path):
        try:
            with sf.SoundFile(path) as f:
                f.seek(f.frames - 1)
                f.read(1)
        except Exception as e:
            raise ValueError(f"檔案損壞: {e}")

    def _generate_midi_mask(self, midi_path, start_time, num_frames):
        mask = np.zeros((CONFIG["TARGET_BINS"], num_frames), dtype=np.float32)
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            fs = CONFIG["SAMPLE_RATE"] / CONFIG["HOP_LENGTH"]
            piano_roll = pm.get_piano_roll(fs=fs)
            
            start_frame = int(start_time * fs)
            end_frame = start_frame + num_frames
            
            for note_num in range(128):
                freq = librosa.midi_to_hz(note_num)
                bin_idx = int(np.round(freq * CONFIG["N_FFT"] / CONFIG["SAMPLE_RATE"]))
                
                if bin_idx < CONFIG["TARGET_BINS"]:
                    if piano_roll.shape[1] > start_frame:
                        segment = piano_roll[note_num, start_frame:end_frame]
                        actual_len = min(len(segment), num_frames)
                        mask[bin_idx, :actual_len] = (segment[:actual_len] > 0).astype(np.float32)
            
            return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        except:
            return torch.zeros((1, CONFIG["TARGET_BINS"], num_frames), dtype=torch.float32)
    def load_with_retry(self, idx, retry_count):
        if retry_count > 5:
            # 失敗 fallback: 回傳 3 個張量
            return torch.zeros(CONFIG["CHUNK_SIZE"]), \
                torch.zeros((2, CONFIG["TARGET_BINS"], 352)), \
                torch.zeros((2, CONFIG["CHUNK_SIZE"]))

        orig_path = self.data_list[idx]
        fname = os.path.basename(orig_path)
        
        try:
            mel_audio_path = orig_path.replace(CONFIG["DIR_MIX"], CONFIG["DIR_MELODY"]).replace(CONFIG["EXT_MIX"], CONFIG["EXT_MELODY"])
            acc_audio_path = orig_path.replace(CONFIG["DIR_MIX"], CONFIG["DIR_ACCOMP"]).replace(CONFIG["EXT_MIX"], CONFIG["EXT_ACCOMP"])
            
            self._validate_file(orig_path)
            total_duration = librosa.get_duration(path=orig_path)
            start_time = np.random.uniform(0, max(0, total_duration - CONFIG["DURATION"]))
            
            # 1. 讀取音訊波形
            wav_orig, _ = librosa.load(orig_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            wav_orig = librosa.util.fix_length(wav_orig, size=CONFIG["CHUNK_SIZE"])
            
            # 2. 準備目標答案
            wav_mel, _ = librosa.load(mel_audio_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            wav_acc, _ = librosa.load(acc_audio_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            
            # 確保擷取後的波形長度完美對齊
            wav_mel = librosa.util.fix_length(wav_mel, size=CONFIG["CHUNK_SIZE"])
            wav_acc = librosa.util.fix_length(wav_acc, size=CONFIG["CHUNK_SIZE"])
            
            spec_mel = self.wav_to_spec(wav_mel)
            spec_acc = self.wav_to_spec(wav_acc)
            
            # 3. 組合答案: 僅保留 2 通道 [Audio_Mel, Audio_Acc]
            target = torch.cat([spec_mel, spec_acc], dim=0)
            
            # 4. 組合目標時域音訊供 SI-SDR 使用: [Melody_Wav, Accomp_Wav]
            target_audios = torch.cat([
                torch.from_numpy(wav_mel).unsqueeze(0),
                torch.from_numpy(wav_acc).unsqueeze(0)
            ], dim=0)
            
            return torch.from_numpy(wav_orig), target, target_audios

        except Exception as e:
            self._log_error(f"{fname} | {str(e)}")
            new_idx = random.randint(0, len(self.data_list) - 1)
            return self.load_with_retry(new_idx, retry_count + 1)

    def wav_to_spec(self, wav):
        stft = librosa.stft(wav, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        magnitude = np.abs(stft)
        
        # 使用 Log 轉換
        log_spec = np.log(magnitude + 1e-6)
        log_spec = log_spec[:CONFIG["TARGET_BINS"], :]
        
        # 譜映射 (Spectral Mapping) 建議使用固定的縮放
        # 這裡將 -10 (約 -80dB) 到 0 映射到 0~1 之間
        # 這樣模型才能學會「音量絕對值」的映射關係
        norm_spec = (log_spec + 10.0) / 10.0
        
        return torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
