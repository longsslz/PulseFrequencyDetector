# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from src.config import Config
from .generator import generate_synthetic_pulse, add_noise_to_signal, add_missing_pulses
from .preprocessor import normalize_signal


class PulseDataset(Dataset):
    """用于训练模型的脉冲数据集"""

    def __init__(self, config, num_samples=None, synthetic=True, real_data_path=None):
        """
        初始化数据集

        参数:
            config (Config): 配置对象
            num_samples (int): 样本数量（如果为None，则使用配置值）
            synthetic (bool): 是否使用合成数据
            real_data_path (str): 真实数据文件路径（如果使用真实数据）
        """
        self.config = config
        self.synthetic = synthetic
        self.real_data_path = real_data_path

        # 确定样本数量
        self.num_samples = num_samples or config.DATA['num_samples']
        self.seq_length = config.DATA['seq_length']
        self.sampling_rate = config.DATA['sampling_rate']
        self.max_freq = config.DATA['max_freq']

        # 生成或加载数据
        if synthetic:
            self.data, self.labels = self._generate_synthetic_data()
        else:
            self.data, self.labels = self._load_real_data()

    def _generate_synthetic_data(self):
        """生成合成数据"""
        X = []
        y = []

        for _ in range(self.num_samples):
            # 随机频率
            freq = np.random.uniform(0.5, self.max_freq)

            # 生成脉冲信号
            pulse_signal = generate_synthetic_pulse(
                freq,
                self.seq_length,
                self.sampling_rate,
                duty_cycle=np.random.uniform(0.1, 0.5)
            )

            # 添加噪声
            noisy_signal = add_noise_to_signal(
                pulse_signal,
                noise_level=self.config.DATA['noise_level']
            )

            # 添加缺失脉冲
            if np.random.rand() < self.config.DATA['missing_prob']:
                noisy_signal = add_missing_pulses(
                    noisy_signal,
                    missing_prob=self.config.DATA['missing_prob']
                )

            # 归一化信号
            processed_signal = normalize_signal(noisy_signal)

            X.append(processed_signal)
            y.append(freq)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _load_real_data(self):
        """加载真实数据（占位符实现）"""
        # 在实际应用中，这里会加载真实采集的脉冲信号数据
        # 由于我们主要使用合成数据，这里返回空数组
        return np.array([]), np.array([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回形状为 (seq_length, 1) 的序列和标量频率值
        return torch.tensor(self.data[idx].reshape(-1, 1)), torch.tensor(self.labels[idx], dtype=torch.float32)


class RealTimePulseDataset(Dataset):
    """用于模拟实时处理的脉冲数据集"""

    def __init__(self, config, duration=60, freq_variation=True):
        """
        初始化实时数据集

        参数:
            config (Config): 配置对象
            duration (int): 数据持续时间（秒）
            freq_variation (bool): 是否随时间变化频率
        """
        self.config = config
        self.duration = duration
        self.sampling_rate = config.REALTIME['sampling_rate']
        self.num_samples = duration * self.sampling_rate
        self.freq_variation = freq_variation

        # 生成数据
        self.timestamps, self.signal, self.actual_freq = self._generate_data()

    def _generate_data(self):
        """生成随时间变化的脉冲信号"""
        # 时间数组
        t = np.linspace(0, self.duration, self.num_samples)

        # 创建随时间变化的频率
        if self.freq_variation:
            # 基础频率 + 正弦变化
            base_freq = np.random.uniform(2, 10)
            freq_variation = np.sin(np.linspace(0, 4 * np.pi, self.num_samples)) * 3 + base_freq
        else:
            # 固定频率
            fixed_freq = np.random.uniform(2, 20)
            freq_variation = np.full(self.num_samples, fixed_freq)

        # 生成脉冲信号
        pulse_signal = np.zeros(self.num_samples)
        for i in range(1, self.num_samples):
            dt = t[i] - t[i - 1]
            phase_increment = 2 * np.pi * freq_variation[i] * dt
            pulse_signal[i] = 1.0 if (pulse_signal[i - 1] + phase_increment) % (2 * np.pi) < np.pi else 0.0

        # 添加噪声
        noisy_signal = add_noise_to_signal(
            pulse_signal,
            noise_level=self.config.DATA['noise_level']
        )

        # 添加脉冲丢失
        noisy_signal = add_missing_pulses(
            noisy_signal,
            missing_prob=self.config.DATA['missing_prob']
        )

        # 滤波
        filtered_signal = filter_signal(
            noisy_signal,
            filter_type='lowpass',
            cutoff_freq=0.1
        )

        return t, filtered_signal, freq_variation

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 对于实时处理，我们只返回当前时间点的样本
        return torch.tensor(self.signal[idx], dtype=torch.float32)

    def get_full_data(self):
        """获取完整数据集用于分析"""
        return {
            'timestamps': self.timestamps,
            'signal': self.signal,
            'actual_frequency': self.actual_freq
        }