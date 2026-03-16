import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os
import soundfile as sf

# --- 設定參數 ---
SAMPLE_RATE = 44100
DURATION = 3.0
CHUNK_SIZE = int(SAMPLE_RATE * DURATION)
SPEC_SHAPE = (1, 1024, 256) 

class AudioDataset(Dataset):
    def __init__(self, data_dir, type="train", validate_on_init=True):
        self.mix_dir = os.path.join(data_dir, 'mix_audio_flac')
        self.mel_dir = os.path.join(data_dir, 'melody_audio_flac')
        self.acc_dir = os.path.join(data_dir, 'accomp_audio_flac')
        
        if not os.path.exists(self.mix_dir):
            raise FileNotFoundError(f"找不到 mix 資料夾：{self.mix_dir}")

        all_filenames = sorted([
            f for f in os.listdir(self.mix_dir) 
            if f.lower().endswith('.flac')
        ])
        
        if len(all_filenames) == 0:
            raise ValueError(f"在 {self.mix_dir} 找不到任何 flac 檔案！")
        
        print(f"找到 {len(all_filenames)} 個檔案，開始驗證...")
        
        # --- 驗證流程 ---
        if validate_on_init:
            self.filenames = []
            bad_files = []
            good_count = 0
            
            for i, fname in enumerate(all_filenames, 1):
                is_valid, error_msg = self._is_file_valid(fname)
                
                if is_valid:
                    self.filenames.append(fname)
                    good_count += 1
                    
                    # 每 100 首輸出一次進度
                    if good_count % 100 == 0:
                        print(f"✅ 已驗證 {good_count} 首正常檔案 (總進度: {i}/{len(all_filenames)})")
                else:
                    bad_files.append((fname, error_msg))
                    # 有壞檔立即輸出
                    print(f"❌ 壞檔 [{len(bad_files)}]: {fname}")
                    print(f"   原因: {error_msg}")
            
            # 最終統計
            print(f"\n{'='*80}")
            print(f"✅ 驗證完成: {len(self.filenames)} 正常, {len(bad_files)} 損壞")
            print(f"{'='*80}")
            
            # 記錄壞檔案
            if bad_files:
                with open("bad_files_list.txt", "w", encoding="utf-8") as f:
                    f.write(f"總共 {len(bad_files)} 個損壞檔案:\n\n")
                    for bf, err in bad_files:
                        f.write(f"{bf}\n  原因: {err}\n\n")
                print(f"📝 壞檔案清單已儲存至 bad_files_list.txt\n")
        else:
            self.filenames = all_filenames
            print("⚠️ 跳過驗證步驟")

    def _get_corresponding_filenames(self, mix_fname):
        """根據 mix 檔名生成對應的 melody 和 accomp 檔名"""
        # mix 檔名格式: "xxx_mixed.flac"
        # 需要轉換成: "xxx_melody.flac" 和 "xxx_accomp.flac"
        base_name = mix_fname.replace('_mixed.flac', '')
        mel_fname = base_name + '_melody.flac'
        acc_fname = base_name + '_accomp.flac'
        return mel_fname, acc_fname

    def _is_file_valid(self, fname):
        """
        檢查三個對應的檔案是否都正常
        返回: (是否正常, 錯誤訊息)
        """
        mel_fname, acc_fname = self._get_corresponding_filenames(fname)
        
        paths = {
            'mix': os.path.join(self.mix_dir, fname),
            'melody': os.path.join(self.mel_dir, mel_fname),
            'accomp': os.path.join(self.acc_dir, acc_fname)
        }
        
        for name, path in paths.items():
            # 檢查 1: 檔案存在
            if not os.path.exists(path):
                return False, f"{name} 檔案不存在: {path}"
            
            # 檢查 2: 檔案大小
            try:
                size = os.path.getsize(path)
                if size < 1000:  # 小於 1KB
                    return False, f"{name} 檔案過小 ({size} bytes)"
            except Exception as e:
                return False, f"{name} 無法取得檔案大小: {e}"
            
            # 檢查 3: SoundFile 驗證
            try:
                with sf.SoundFile(path) as f:
                    if f.frames == 0:
                        return False, f"{name} 音訊長度為 0"
                    
                    # 嘗試讀取最後一幀
                    try:
                        f.seek(max(0, f.frames - 1))
                        f.read(1)
                    except Exception as e:
                        return False, f"{name} 檔案結構損壞 (seek/read 失敗): {e}"
                        
            except Exception as e:
                return False, f"{name} SoundFile 無法開啟: {e}"
        
        return True, None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        mel_fname, acc_fname = self._get_corresponding_filenames(fname)
        
        orig_path = os.path.join(self.mix_dir, fname)
        mel_path = os.path.join(self.mel_dir, mel_fname)
        acc_path = os.path.join(self.acc_dir, acc_fname)

        try:
            # 1. 取得長度
            total_duration = librosa.get_duration(path=orig_path)
            if total_duration == 0:
                raise ValueError("音訊長度為 0")

            if total_duration > DURATION:
                start_time = np.random.uniform(0, total_duration - DURATION)
            else:
                start_time = 0

            # 2. 載入音訊
            wav_orig, _ = librosa.load(orig_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
            wav_mel, _ = librosa.load(mel_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)
            wav_acc, _ = librosa.load(acc_path, sr=SAMPLE_RATE, offset=start_time, duration=DURATION)

            # 3. 調整長度
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
            print(f"\n❌ 載入失敗: {fname} - {str(e)}")
            # 返回零張量避免訓練中斷
            return torch.zeros(SPEC_SHAPE), torch.zeros((2, 1024, 256))

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


# ===== 使用範例 =====
if __name__ == "__main__":
    # 第一次執行時會驗證所有檔案
    dataset = AudioDataset(
        data_dir=r"C:\project_data\two_line_midi\flac_output",
        validate_on_init=True
    )
    
    print(f"可用的資料數量: {len(dataset)}")
    
    # 測試讀取第一筆資料
    if len(dataset) > 0:
        print("\n測試讀取第一筆資料...")
        spec, target = dataset[0]
        print(f"頻譜圖形狀: {spec.shape}")
        print(f"目標形狀: {target.shape}")