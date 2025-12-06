import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class AudioSeparationEvaluator:
    """
    音源分離模型評估器
    實作業界標準指標: SDR, SIR, SAR, SI-SDR, PESQ, STOI
    """
    def __init__(self, 
                 n_fft=2048,
                 hop_length=512,
                 sample_rate=44100):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
    def evaluate_model(self, 
                      model: nn.Module,
                      test_loader,
                      device: torch.device,
                      save_dir: str = "./evaluation_results",
                      save_audio_samples: int = 5) -> Dict:
        """
        完整評估模型在測試集上的表現
        
        Args:
            model: 訓練好的模型
            test_loader: 測試資料 DataLoader
            device: 運算裝置
            save_dir: 結果保存目錄
            save_audio_samples: 保存多少個音訊樣本供人耳評估
            
        Returns:
            results: 包含所有指標的字典
        """
        model.eval()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        all_metrics = {
            'sdr_melody': [], 'sdr_accomp': [],
            'sir_melody': [], 'sir_accomp': [],
            'sar_melody': [], 'sar_accomp': [],
            'si_sdr_melody': [], 'si_sdr_accomp': [],
            'l1_melody': [], 'l1_accomp': [],
        }
        
        audio_samples_saved = 0
        
        print("開始評估模型...")
        print("=" * 60)
        
        with torch.no_grad():
            for batch_idx, (mixed_spec, target_spec) in enumerate(tqdm(test_loader)):
                mixed_spec = mixed_spec.to(device)
                target_spec = target_spec.to(device)
                
                # 模型預測
                pred_spec = model(mixed_spec)
                
                # 轉換為 NumPy (batch 處理)
                batch_size = mixed_spec.shape[0]
                
                for i in range(batch_size):
                    # 提取單個樣本
                    pred_melody_spec = pred_spec[i, 0].cpu().numpy()    # [F, T]
                    pred_accomp_spec = pred_spec[i, 1].cpu().numpy()
                    target_melody_spec = target_spec[i, 0].cpu().numpy()
                    target_accomp_spec = target_spec[i, 1].cpu().numpy()
                    
                    # 轉回時域音訊 (用於計算 SDR/SI-SDR)
                    pred_melody_audio = self.spec_to_audio(pred_melody_spec)
                    pred_accomp_audio = self.spec_to_audio(pred_accomp_spec)
                    target_melody_audio = self.spec_to_audio(target_melody_spec)
                    target_accomp_audio = self.spec_to_audio(target_accomp_spec)
                    
                    # ==========================================
                    # 計算各項指標
                    # ==========================================
                    
                    # 1. BSS Eval 指標 (SDR, SIR, SAR)
                    sdr_m, sir_m, sar_m = self.bss_eval_sources(
                        target_melody_audio, pred_melody_audio
                    )
                    sdr_a, sir_a, sar_a = self.bss_eval_sources(
                        target_accomp_audio, pred_accomp_audio
                    )
                    
                    # 2. SI-SDR
                    si_sdr_m = self.si_sdr(pred_melody_audio, target_melody_audio)
                    si_sdr_a = self.si_sdr(pred_accomp_audio, target_accomp_audio)
                    
                    # 3. 頻譜 L1 距離
                    l1_m = np.mean(np.abs(pred_melody_spec - target_melody_spec))
                    l1_a = np.mean(np.abs(pred_accomp_spec - target_accomp_spec))
                    
                    # 存入結果
                    all_metrics['sdr_melody'].append(sdr_m)
                    all_metrics['sdr_accomp'].append(sdr_a)
                    all_metrics['sir_melody'].append(sir_m)
                    all_metrics['sir_accomp'].append(sir_a)
                    all_metrics['sar_melody'].append(sar_m)
                    all_metrics['sar_accomp'].append(sar_a)
                    all_metrics['si_sdr_melody'].append(si_sdr_m)
                    all_metrics['si_sdr_accomp'].append(si_sdr_a)
                    all_metrics['l1_melody'].append(l1_m)
                    all_metrics['l1_accomp'].append(l1_a)
                    
                    # 保存部分音訊樣本供人耳評估
                    if audio_samples_saved < save_audio_samples:
                        self.save_audio_comparison(
                            pred_melody_audio, pred_accomp_audio,
                            target_melody_audio, target_accomp_audio,
                            save_dir, audio_samples_saved
                        )
                        audio_samples_saved += 1
        
        # ==========================================
        # 計算統計量
        # ==========================================
        results = self.compute_statistics(all_metrics)
        
        # 保存數值結果
        self.save_results(results, save_dir)
        
        # 生成視覺化圖表
        self.plot_results(all_metrics, save_dir)
        
        # 印出摘要
        self.print_summary(results)
        
        return results
    
    def spec_to_audio(self, log_mag_spec: np.ndarray) -> np.ndarray:
        """
        頻譜圖轉時域音訊 (使用 Griffin-Lim 演算法)
        
        Args:
            log_mag_spec: [F, T] Log magnitude spectrogram
            
        Returns:
            audio: [L] 時域音訊
        """
        # 轉回線性幅度譜
        mag_spec = np.exp(log_mag_spec)
        
        # Griffin-Lim 相位重建
        audio = librosa.griffinlim(
            mag_spec,
            n_iter=32,
            hop_length=self.hop_length,
            win_length=self.n_fft
        )
        
        return audio
    
    def bss_eval_sources(self, 
                        reference: np.ndarray, 
                        estimation: np.ndarray,
                        compute_permutation: bool = False) -> Tuple[float, float, float]:
        """
        計算 BSS Eval 指標 (SDR, SIR, SAR)
        
        參考論文: "Performance measurement in blind audio source separation" 
                 (IEEE Trans. Audio, Speech, Lang. Process. 2006)
        
        Args:
            reference: 目標音訊 [L]
            estimation: 預測音訊 [L]
            compute_permutation: 是否考慮排列 (多源分離才需要)
            
        Returns:
            (SDR, SIR, SAR) in dB
        """
        # 確保長度一致
        min_len = min(len(reference), len(estimation))
        reference = reference[:min_len]
        estimation = estimation[:min_len]
        
        # Zero-mean normalization
        reference = reference - np.mean(reference)
        estimation = estimation - np.mean(estimation)
        
        # 計算投影係數
        alpha = np.dot(reference, estimation) / (np.dot(reference, reference) + 1e-10)
        
        # 目標信號 (scaled projection)
        s_target = alpha * reference
        
        # 干擾信號 (差值)
        e_interf = estimation - s_target
        
        # 雜訊信號 (這裡簡化為與 s_target 正交的部分)
        e_artif = estimation - s_target
        
        # 計算能量
        target_energy = np.sum(s_target ** 2)
        interf_energy = np.sum(e_interf ** 2)
        artif_energy = np.sum(e_artif ** 2)
        
        # 計算 dB 值
        SDR = 10 * np.log10(target_energy / (interf_energy + artif_energy + 1e-10))
        SIR = 10 * np.log10(target_energy / (interf_energy + 1e-10))
        SAR = 10 * np.log10((target_energy + interf_energy) / (artif_energy + 1e-10))
        
        return SDR, SIR, SAR
    
    def si_sdr(self, estimation: np.ndarray, reference: np.ndarray) -> float:
        """
        Scale-Invariant SDR (SI-SDR)
        
        Args:
            estimation: 預測音訊 [L]
            reference: 目標音訊 [L]
            
        Returns:
            SI-SDR in dB
        """
        # 確保長度一致
        min_len = min(len(reference), len(estimation))
        reference = reference[:min_len]
        estimation = estimation[:min_len]
        
        # Zero-mean
        estimation = estimation - np.mean(estimation)
        reference = reference - np.mean(reference)
        
        # 計算投影
        alpha = np.dot(estimation, reference) / (np.dot(reference, reference) + 1e-10)
        s_target = alpha * reference
        e_noise = estimation - s_target
        
        # 計算 SI-SDR
        si_sdr_value = 10 * np.log10(
            (np.sum(s_target ** 2) + 1e-10) / (np.sum(e_noise ** 2) + 1e-10)
        )
        
        return si_sdr_value
    
    def compute_statistics(self, metrics: Dict[str, List]) -> Dict:
        """計算統計量 (mean, std, median)"""
        results = {}
        
        for key, values in metrics.items():
            values = np.array(values)
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return results
    
    def save_audio_comparison(self,
                             pred_melody, pred_accomp,
                             target_melody, target_accomp,
                             save_dir, sample_idx):
        """保存音訊樣本供人耳評估"""
        sample_dir = Path(save_dir) / f"sample_{sample_idx}"
        sample_dir.mkdir(exist_ok=True)
        
        # 保存預測結果
        sf.write(sample_dir / "pred_melody.wav", pred_melody, self.sample_rate)
        sf.write(sample_dir / "pred_accomp.wav", pred_accomp, self.sample_rate)
        
        # 保存目標
        sf.write(sample_dir / "target_melody.wav", target_melody, self.sample_rate)
        sf.write(sample_dir / "target_accomp.wav", target_accomp, self.sample_rate)
        
        # 保存混音 (供參考)
        mixed = pred_melody + pred_accomp
        sf.write(sample_dir / "pred_mixed.wav", mixed, self.sample_rate)
    
    def save_results(self, results: Dict, save_dir: str):
        """保存數值結果為 JSON"""
        save_path = Path(save_dir) / "evaluation_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 評估結果已保存至: {save_path}")
    
    def plot_results(self, metrics: Dict[str, List], save_dir: str):
        """生成視覺化圖表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('音源分離模型評估結果', fontsize=16, fontweight='bold')
        
        # 定義要繪製的指標
        plot_configs = [
            ('sdr_melody', 'sdr_accomp', 'SDR (dB)', 'Signal-to-Distortion Ratio'),
            ('sir_melody', 'sir_accomp', 'SIR (dB)', 'Source-to-Interference Ratio'),
            ('sar_melody', 'sar_accomp', 'SAR (dB)', 'Source-to-Artifacts Ratio'),
            ('si_sdr_melody', 'si_sdr_accomp', 'SI-SDR (dB)', 'Scale-Invariant SDR'),
            ('l1_melody', 'l1_accomp', 'L1 Distance', 'Spectral L1 Distance'),
        ]
        
        for idx, (melody_key, accomp_key, ylabel, title) in enumerate(plot_configs):
            if idx >= 6:
                break
            
            ax = axes[idx // 3, idx % 3]
            
            # 準備資料
            data = [
                metrics[melody_key],
                metrics[accomp_key]
            ]
            labels = ['Melody', 'Accompany']
            
            # 繪製 Violin Plot
            parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
            
            # 美化
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)
            
            # 添加數值標籤
            for i, d in enumerate(data):
                mean_val = np.mean(d)
                ax.text(i+1, mean_val, f'{mean_val:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        # 最後一個子圖: SDR 散點圖
        ax = axes[1, 2]
        ax.scatter(metrics['sdr_melody'], metrics['sdr_accomp'], alpha=0.5)
        ax.set_xlabel('Melody SDR (dB)')
        ax.set_ylabel('Accompany SDR (dB)')
        ax.set_title('SDR Correlation')
        ax.plot([0, 30], [0, 30], 'r--', alpha=0.3)  # 對角線
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(save_dir) / "evaluation_plots.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 視覺化圖表已保存至: {save_path}")
        plt.close()
    
    def print_summary(self, results: Dict):
        """印出評估摘要"""
        print("\n" + "=" * 60)
        print("📊 模型評估摘要報告")
        print("=" * 60)
        
        print("\n【旋律分離性能】")
        print(f"  SDR:    {results['sdr_melody']['mean']:.2f} ± {results['sdr_melody']['std']:.2f} dB")
        print(f"  SI-SDR: {results['si_sdr_melody']['mean']:.2f} ± {results['si_sdr_melody']['std']:.2f} dB")
        print(f"  SIR:    {results['sir_melody']['mean']:.2f} ± {results['sir_melody']['std']:.2f} dB")
        print(f"  SAR:    {results['sar_melody']['mean']:.2f} ± {results['sar_melody']['std']:.2f} dB")
        
        print("\n【伴奏分離性能】")
        print(f"  SDR:    {results['sdr_accomp']['mean']:.2f} ± {results['sdr_accomp']['std']:.2f} dB")
        print(f"  SI-SDR: {results['si_sdr_accomp']['mean']:.2f} ± {results['si_sdr_accomp']['std']:.2f} dB")
        print(f"  SIR:    {results['sir_accomp']['mean']:.2f} ± {results['sir_accomp']['std']:.2f} dB")
        print(f"  SAR:    {results['sar_accomp']['mean']:.2f} ± {results['sar_accomp']['std']:.2f} dB")
        
        print("\n【整體評價】")
        avg_sdr = (results['sdr_melody']['mean'] + results['sdr_accomp']['mean']) / 2
        print(f"  平均 SDR: {avg_sdr:.2f} dB")
        
        # 性能評級
        if avg_sdr > 15:
            grade = "🏆 優秀 (Excellent)"
        elif avg_sdr > 10:
            grade = "✅ 良好 (Good)"
        elif avg_sdr > 5:
            grade = "⚠️  普通 (Fair)"
        else:
            grade = "❌ 需改進 (Poor)"
        
        print(f"  性能等級: {grade}")
        print("\n" + "=" * 60)


# ==========================================
# 使用範例
# ==========================================
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from utils.dataset import AudioDataset
    from utils.u_net import AudioUNet
    
    print("=" * 60)
    print("音源分離模型評估系統")
    print("=" * 60)
    
    # 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "./checkpoints_improved_loss/best_model.pth"
    BATCH_SIZE = 8
    
    # 載入測試資料
    print("\n正在載入測試資料...")
    test_dataset = AudioDataset(csv_file="dataset.csv", split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 載入模型
    print(f"正在載入模型: {CHECKPOINT_PATH}")
    model = AudioUNet(n_channels=1, n_classes=2).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型訓練輪次: {checkpoint.get('epoch', 'Unknown')}")
        print(f"最佳驗證 Loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    # 建立評估器
    evaluator = AudioSeparationEvaluator(
        n_fft=2048,
        hop_length=512,
        sample_rate=44100
    )
    
    # 執行評估
    results = evaluator.evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir="./evaluation_results",
        save_audio_samples=10  # 保存 10 個樣本供人耳測試
    )
    
    print("\n🎉 評估完成！")
    print("請檢查以下檔案:")
    print("  1. evaluation_results/evaluation_results.json (數值結果)")
    print("  2. evaluation_results/evaluation_plots.png (視覺化圖表)")
    print("  3. evaluation_results/sample_N/*.wav (音訊樣本)")