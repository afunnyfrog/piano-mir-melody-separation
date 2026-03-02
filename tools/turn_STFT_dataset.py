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
    "MIDI_ROOT": r"G:\project_data\two_line_midi\midi_v3",
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
            # 失敗 fallback: 回傳全零張量 (352 是 16 的倍數，安全)
            return torch.zeros(CONFIG["CHUNK_SIZE"]), \
                torch.zeros((2, CONFIG["TARGET_BINS"], 352)), \
                torch.zeros((4, CONFIG["TARGET_BINS"], 352))

        orig_path = self.data_list[idx]
        fname = os.path.basename(orig_path)
        
        try:
            mel_audio_path = orig_path.replace(CONFIG["DIR_MIX"], CONFIG["DIR_MELODY"]).replace(CONFIG["EXT_MIX"], CONFIG["EXT_MELODY"])
            acc_audio_path = orig_path.replace(CONFIG["DIR_MIX"], CONFIG["DIR_ACCOMP"]).replace(CONFIG["EXT_MIX"], CONFIG["EXT_ACCOMP"])
            
            midi_fname = fname.replace(CONFIG["EXT_MIX"], CONFIG["EXT_MIDI"])
            midi_path_mel = os.path.join(CONFIG["MIDI_ROOT"], CONFIG["MIDI_SUB_MEL"], midi_fname)
            midi_path_acc = os.path.join(CONFIG["MIDI_ROOT"], CONFIG["MIDI_SUB_ACC"], midi_fname)

            self._validate_file(orig_path)
            total_duration = librosa.get_duration(path=orig_path)
            start_time = np.random.uniform(0, max(0, total_duration - CONFIG["DURATION"]))
            
            # 1. 讀取音訊波形
            wav_orig, _ = librosa.load(orig_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            wav_orig = librosa.util.fix_length(wav_orig, size=CONFIG["CHUNK_SIZE"])
            
            # 2. 準備目標答案
            wav_mel, _ = librosa.load(mel_audio_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            wav_acc, _ = librosa.load(acc_audio_path, sr=CONFIG["SAMPLE_RATE"], offset=start_time, duration=CONFIG["DURATION"])
            spec_mel = self.wav_to_spec(librosa.util.fix_length(wav_mel, size=CONFIG["CHUNK_SIZE"]))
            spec_acc = self.wav_to_spec(librosa.util.fix_length(wav_acc, size=CONFIG["CHUNK_SIZE"]))
            num_frames = spec_mel.shape[2]
            
            # 3. 產生 MIDI 遮罩
            mask_mel = self._generate_midi_mask(midi_path_mel, start_time, num_frames)
            mask_acc = self._generate_midi_mask(midi_path_acc, start_time, num_frames)
            
            # 4. 產生獨立的輸入提示 (獨立 Dropout)
            hint_mel = torch.zeros_like(mask_mel)
            hint_acc = torch.zeros_like(mask_acc)
            
            if self.split == "train":
                if random.random() > CONFIG["PROB_DROP_MELODY_HINT"]:
                    hint_mel = mask_mel
                if random.random() > CONFIG["PROB_DROP_ACCOMP_HINT"]:
                    hint_acc = mask_acc
            else:
                hint_mel, hint_acc = mask_mel, mask_acc
            
            # 合併為 2 通道提示: (2, F, T)
            midi_hints = torch.cat([hint_mel, hint_acc], dim=0)
            
            # 5. 組合 4 通道答案: [Audio_Mel, Audio_Acc, MIDI_Mel, MIDI_Acc]
            target = torch.cat([spec_mel, spec_acc, mask_mel, mask_acc], dim=0)
            
            return torch.from_numpy(wav_orig), midi_hints, target

        except Exception as e:
            self._log_error(f"{fname} | {str(e)}")
            new_idx = random.randint(0, len(self.data_list) - 1)
            return self.load_with_retry(new_idx, retry_count + 1)

    def wav_to_spec(self, wav):
        stft = librosa.stft(wav, n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        magnitude = np.abs(stft)
        log_spec = np.log(magnitude + 1e-6)
        log_spec = log_spec[:CONFIG["TARGET_BINS"], :]
        min_val, max_val = log_spec.min(), log_spec.max()
        norm_spec = (log_spec - min_val) / (max_val - min_val + 1e-6)
        return torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
