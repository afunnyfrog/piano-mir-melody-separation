import os
import random
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset

class WeightedAudioDataset(Dataset):
    def __init__(self, root_dir=r"C:\Users\cebit\Desktop\專題生成\classified_dataset", split="train", sr=44100, segment_seconds=2.0, samples_per_epoch=200):
        """
        標準化 Dataset:
        1. 固定從三個主要資料夾讀取。
        2. 僅篩選 Classical 開頭的音檔。
        3. 保留高品質段落篩選與正規化邏輯。
        """
        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        self.segment_length = int(sr * segment_seconds)
        self.samples_per_epoch = samples_per_epoch

        # --- 定義三個資料夾路徑 ---
        self.mel_dir = os.path.join(root_dir, "melody_audio_flac")
        self.acc_dir = os.path.join(root_dir, "accomp_audio_flac")
        self.mix_dir = os.path.join(root_dir, "mix_audio_flac")

        # 檢查資料夾是否存在
        if not os.path.exists(self.mel_dir):
            raise FileNotFoundError(f"❌ 找不到旋律資料夾: {self.mel_dir}")

        # --- 篩選檔案 ---
        # 僅選取 Classical 開頭且後綴為 _melody.flac 的檔案 (避免重複掃描)
        self.files = [f for f in os.listdir(self.mel_dir)
                      if f.endswith(".flac") and f.startswith("Classical")]

        print(f"\n[INFO] 資料集初始化 - {split}")
        print(f"  - 旋律路徑: {self.mel_dir}")
        print(f"  - 匹配 'Classical' 檔案總數: {len(self.files)}")

        if len(self.files) == 0:
            print(f"⚠️ 警告: 在 {self.mel_dir} 中找不到 Classical 開頭的檔案！")

    def _load_audio(self, path):
        try:
            data, _ = sf.read(path, dtype='float32')
            if len(data.shape) > 1:
                data = np.mean(data, axis=1) # 轉單聲道
            return torch.from_numpy(data)
        except Exception as e:
            print(f"讀取錯誤: {path} -> {e}")
            return None

    def _check_segment_quality(self, melody, accomp):
        """ 品質檢查：確保旋律活躍度佔伴奏的一定比例 """
        abs_mel = torch.abs(melody)
        abs_acc = torch.abs(accomp)
        silence_thresh = 0.005 # 稍微提高靜音門檻

        mel_active = torch.sum(abs_mel > silence_thresh).item()
        acc_active = torch.sum(abs_acc > silence_thresh).item()
        total_samples = len(melody)

        # 旋律與伴奏都必須具備 5% 以上的活躍採樣
        if mel_active < (total_samples * 0.05) or acc_active < (total_samples * 0.05):
            return False

        # 旋律活躍度必須至少佔伴奏的 30% 以上，保證旋律足夠明顯
        ratio = mel_active / (acc_active + 1e-6)
        return ratio >= 0.30

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        max_retries = 30 # 提高重試次數

        for _ in range(max_retries):
            if not self.files: break
            filename = random.choice(self.files)

            # 假設檔案命名規則為: Classical_XXX_melody.flac
            # 對應轉換為 _accomp.flac 和 _mixed.flac
            fname_mel = filename
            fname_acc = filename.replace("_melody.flac", "_accomp.flac")
            fname_mix = filename.replace("_melody.flac", "_mixed.flac")

            path_mel = os.path.join(self.mel_dir, fname_mel)
            path_acc = os.path.join(self.acc_dir, fname_acc)
            path_mix = os.path.join(self.mix_dir, fname_mix)

            # 確保三者皆存在
            if not (os.path.exists(path_acc) and os.path.exists(path_mix)):
                continue

            wav_mel = self._load_audio(path_mel)
            wav_acc = self._load_audio(path_acc)
            wav_mix = self._load_audio(path_mix)

            if wav_mel is None or wav_acc is None or wav_mix is None:
                continue

            # 裁切段落
            min_len = min(wav_mel.shape[0], wav_acc.shape[0], wav_mix.shape[0])
            if min_len < self.segment_length:
                continue

            start_idx = random.randint(0, min_len - self.segment_length)
            end_idx = start_idx + self.segment_length

            chunk_mel = wav_mel[start_idx:end_idx]
            chunk_acc = wav_acc[start_idx:end_idx]
            chunk_mix = wav_mix[start_idx:end_idx]

            # ✨ 正規化 (Peak Normalization) - 統一音量基準
            max_val = torch.max(torch.abs(chunk_mix))
            if max_val > 1e-6:
                scale_factor = 0.9 / max_val
                chunk_mix *= scale_factor
                chunk_mel *= scale_factor
                chunk_acc *= scale_factor

            # 品質檢查
            if self._check_segment_quality(chunk_mel, chunk_acc):
                return chunk_mix.unsqueeze(0), torch.stack([chunk_mel, chunk_acc], dim=0)

        # 如果真的沒找到高品質段落，回傳零向量 (避免程式崩潰)
        return torch.zeros((1, self.segment_length)), torch.zeros((2, self.segment_length))