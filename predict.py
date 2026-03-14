import torch
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt  # ✨ 新增
from utils.u_net import AudioUNet

# --- 設定維持不變 ---
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
MODEL_PATH = "./checkpoints_dynamic_8s/best_model.pth"
INPUT_AUDIO = r"C:\Users\cebit\Desktop\專題生成\classified_dataset\mix_audio_flac\Classical_Classical Era_Muzio Clementi_Sonatina-1-2_mixed.flac"
OUTPUT_DIR = "./results"
SAMPLE_RATE = 44100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = AudioUNet(n_channels=1, n_classes=2).to(device)

    # 載入權重邏輯 (維持不變)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ 成功載入模型: {MODEL_PATH}")
    else:
        print("❌ 找不到權重檔")
        return

    model.eval()

    # 3. 讀取音訊並轉換
    y, _ = librosa.load(INPUT_AUDIO, sr=SAMPLE_RATE)
    stft_mix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mag_mix = np.abs(stft_mix)
    phase_mix = np.angle(stft_mix)

    # 4. 推論準備 (與 Padding)
    input_tensor = torch.from_numpy(mag_mix).unsqueeze(0).unsqueeze(0).to(device)
    pad_h = 31 - ((input_tensor.shape[2] - 1) % 32)
    pad_w = 31 - ((input_tensor.shape[3] - 1) % 32)
    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h))

    # 5. 執行分離
    print("正在執行 2D 頻譜分離運算...")
    with torch.no_grad():
        pred_mags = model(input_tensor)

    # 移除 Padding 並轉回 Numpy [2, Freq, Time]
    pred_mags = pred_mags[:, :, :mag_mix.shape[0], :mag_mix.shape[1]].cpu().numpy()[0]

    # ✨ 6. [新增功能] 產生頻譜預覽圖 (前 10 秒)
    print("正在生成頻譜預覽圖...")
    preview_seconds = 10
    # 計算 10 秒對應的 Frame 數量
    num_frames = int((preview_seconds * SAMPLE_RATE) / HOP_LENGTH)

    # 擷取前 num_frames 幀
    mag_mix_slice = mag_mix[:, :num_frames]
    pred_mel_slice = pred_mags[0, :, :num_frames]

    plt.figure(figsize=(15, 10))

    # 子圖 1: 原始混合頻譜
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(mag_mix_slice, ref=np.max),
                             sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
    plt.title('Original Mix Spectrogram (Classical)')
    plt.colorbar(format='%+2.0f dB')

    # 子圖 2: 分離後的旋律頻譜
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(pred_mel_slice, ref=np.max),
                             sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='log')
    plt.title('Separated Melody Spectrogram (Top-note Focus)')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "spectrogram_preview.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ 頻譜預覽圖已儲存: {plot_path}")

    # 7. 儲存音訊檔 (維持不變)
    names = ['melody', 'accompaniment']
    for i, name in enumerate(names):
        reconstructed_stft = pred_mags[i] * np.exp(1j * phase_mix)
        y_out = librosa.istft(reconstructed_stft, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

        # 音量歸一化
        max_amp = np.max(np.abs(y_out))
        if max_amp > 1e-6:
            y_out = y_out * (0.9 / max_amp)

        save_path = os.path.join(OUTPUT_DIR, f"result_{name}.wav")
        sf.write(save_path, y_out, SAMPLE_RATE)
        print(f"✅ 已儲存音訊: {save_path}")

    print("✨ 分離任務全數完成！")


if __name__ == "__main__":
    main()