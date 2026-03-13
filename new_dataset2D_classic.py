import os
import random
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset


class WeightedAudioDataset(Dataset):
    def __init__(self, root_dir, split="train", sr=44100, segment_seconds=8.0, samples_per_epoch=1000):
        """
        修正版 Dataset:
        1. 移除 source_type 依賴，統一標準。
        2. 加入音量正規化，解決 Loss 卡住問題。
        """
        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        self.segment_length = int(sr * segment_seconds)
        self.samples_per_epoch = samples_per_epoch

        # 這裡假設您的 DATA_DIR 是指向 classified_dataset 或是 flac_output
        # 為了相容性，我們寫一個通用邏輯：優先找 melody_audio_flac

        # 嘗試路徑 A: 直接結構 (flac_output)
        if os.path.exists(os.path.join(root_dir, "melody_audio_flac")):
            self.mel_dir = os.path.join(root_dir, "melody_audio_flac")
            self.acc_dir = os.path.join(root_dir, "accomp_audio_flac")
            self.mix_dir = os.path.join(root_dir, "mix_audio_flac")
        # 嘗試路徑 B: 分類結構 (classified_dataset) -> 這裡我們簡單化，直接掃描 balanced
        elif os.path.exists(os.path.join(root_dir, "balanced", "melody_audio_flac")):
            self.mel_dir = os.path.join(root_dir, "balanced", "melody_audio_flac")
            self.acc_dir = os.path.join(root_dir, "balanced", "accomp_audio_flac")
            self.mix_dir = os.path.join(root_dir, "balanced", "mix_audio_flac")
        else:
            raise FileNotFoundError(f"找不到 melody_audio_flac 資料夾，請檢查路徑: {root_dir}")
        # 強制指定到 balanced 資料夾
        balanced_path = os.path.join(root_dir, "balanced")
        self.mel_dir = os.path.join(balanced_path, "melody_audio_flac")

        # 這裡加一行強制輸出的 Debug
        print(f"\n[DEBUG] 正在掃描路徑: {self.mel_dir}", flush=True)

        # 掃描並過濾 Classical
        self.files = [f for f in os.listdir(self.mel_dir)
                      if f.endswith(".flac") and f.startswith("Classical")]

        # ✨ 修改這裡：無論有沒有找到，都強制印出來
        print(f"✅ [{split}] 過濾完成！符合 'Classical' 的檔案總數: {len(self.files)}", flush=True)

        if len(self.files) == 0:
            # 如果是 0，印出前 5 個看到的檔案，看看為什麼沒匹配到
            all_files = os.listdir(self.mel_dir)[:5]
            print(f"❌ 找不到 Classical 開頭檔案。資料夾內範例檔案: {all_files}", flush=True)

        print(f"[{split}] 資料庫初始化完成:")
        print(f"  - 來源路徑: {self.mel_dir}")
        print(f"  - 檔案總數: {len(self.files)}")
        print(f"  - 訓練模式: 每輪隨機抽取 {self.samples_per_epoch} 筆")

    def _load_audio(self, path):
        try:
            data, _ = sf.read(path, dtype='float32')
            if len(data.shape) > 1: data = np.mean(data, axis=1)
            return torch.from_numpy(data)
        except Exception:
            return None

    # ✨ [修正重點 1] 移除 source_type 參數
    def _check_segment_quality(self, melody, accomp):
        abs_mel = torch.abs(melody)
        abs_acc = torch.abs(accomp)
        silence_thresh = 0.001

        mel_active = torch.sum(abs_mel > silence_thresh).item()
        acc_active = torch.sum(abs_acc > silence_thresh).item()
        total_samples = len(melody)

        # 條件 1: 基本活躍度 (5%)
        if mel_active < (total_samples * 0.05): return False
        if acc_active < (total_samples * 0.05): return False

        # 條件 2: 比例檢查 (嚴格篩選)
        # 這裡的門檻建議先放寬一點點 (0.05 ~ 0.1) 避免一直 retry 失敗
        ratio = mel_active / (acc_active + 1e-6)

        # ✨ 提高門檻：旋律活躍度必須至少佔伴奏的 30% 以上 (原本是 15%)
        # 這樣篩選出來的訓練資料，旋律線會非常清晰
        if ratio < 0.30:
            return False

        return True

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        max_retries = 20

        for _ in range(max_retries):
            if not self.files: break
            filename = random.choice(self.files)

            fname_mel = filename
            fname_acc = filename.replace("_melody.flac", "_accomp.flac")
            fname_mix = filename.replace("_melody.flac", "_mixed.flac")

            path_mel = os.path.join(self.mel_dir, fname_mel)
            path_acc = os.path.join(self.acc_dir, fname_acc)
            path_mix = os.path.join(self.mix_dir, fname_mix)

            if not (os.path.exists(path_acc) and os.path.exists(path_mix)): continue

            wav_mel = self._load_audio(path_mel)
            wav_acc = self._load_audio(path_acc)
            wav_mix = self._load_audio(path_mix)

            if wav_mel is None or wav_acc is None or wav_mix is None: continue

            min_len = min(wav_mel.shape[0], wav_acc.shape[0], wav_mix.shape[0])
            if min_len < self.segment_length: continue

            max_start = min_len - self.segment_length
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + self.segment_length

            chunk_mel = wav_mel[start_idx:end_idx]
            chunk_acc = wav_acc[start_idx:end_idx]
            chunk_mix = wav_mix[start_idx:end_idx]

            if chunk_mix.shape[0] != self.segment_length: continue

            # ✨ [修正重點 2] 強制正規化 (Peak Normalization)
            # 這能解決 Loss 降不下來的問題
            max_val = torch.max(torch.abs(chunk_mix))
            if max_val > 1e-6:
                scale_factor = 0.9 / max_val
                chunk_mix = chunk_mix * scale_factor
                chunk_mel = chunk_mel * scale_factor
                chunk_acc = chunk_acc * scale_factor

            # ✨ [修正重點 3] 呼叫時不再傳入 source_type
            if self._check_segment_quality(chunk_mel, chunk_acc):
                return chunk_mix.unsqueeze(0), torch.stack([chunk_mel, chunk_acc], dim=0)

        # Fallback
        zero_tensor = torch.zeros((1, self.segment_length))
        target_tensor = torch.zeros((2, self.segment_length))
        return zero_tensor, target_tensor
