import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.u_net import AudioUNet

# --- 設定 ---
MODEL_PATH = "./checkpoints_direct_mse/model_epoch_50.pth" # 確保路徑對
INPUT_AUDIO = r"C:\Users\richa\Documents\專題數據\maestro-v3.0.0\maestro-v3.0.0\2017\MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.wav"
SAMPLE_RATE = 44100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 讀取
    y, sr = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE, duration=10.0) # 只看前10秒
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    mag = np.abs(stft)
    log_spec = np.log(mag + 1e-6)
    
    # 正規化
    min_v, max_v = log_spec.min(), log_spec.max()
    norm_spec = (log_spec - min_v) / (max_v - min_v + 1e-6)
    
    # 準備輸入
    inp = norm_spec[:1024, :]
    pad = 0
    if inp.shape[1] % 16 != 0:
        pad = 16 - (inp.shape[1] % 16)
        inp = np.pad(inp, ((0,0), (0, pad)))
        
    t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        pred = model(t).cpu().numpy()[0]
    
    # --- 畫圖 ---
    plt.figure(figsize=(15, 10))
    
    # 1. 原曲
    plt.subplot(3, 1, 1)
    librosa.display.specshow(inp, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input (Mixture)')
    
    # 2. 預測的 Melody
    plt.subplot(3, 1, 2)
    # 去除 padding
    p_mel = pred[0]
    if pad > 0: p_mel = p_mel[:, :-pad]
    librosa.display.specshow(p_mel, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Predicted Melody')
    
    # 3. 預測的 Accomp
    plt.subplot(3, 1, 3)
    p_acc = pred[1]
    if pad > 0: p_acc = p_acc[:, :-pad]
    librosa.display.specshow(p_acc, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Predicted Accompaniment')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()