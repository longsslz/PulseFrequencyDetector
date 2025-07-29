# src/data/dataset.py
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.generator import (
    generate_synthetic_pulse,
    add_noise_to_signal,
    add_missing_pulses,
    filter_signal,
)
from src.data.preprocessor import normalize_signal, load_and_process_tdms, sliding_window_processing


def _load_real_data_info(real_data_path, logger):
    """加载真实数据文件信息"""
    from src.data.preprocessor import process_tdms_folder
    # 返回格式: [(文件路径, 频率, 样本数), ...]
    return process_tdms_folder(real_data_path, window_size=3, save_processed=True, logger = logger)

class PulseDataset(Dataset):
    """用于训练模型的脉冲数据集"""

    def __init__(self, config, num_samples=None, synthetic=True, real_data_path=None, logger = None):
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
        self.logger = logger

        self.task_type = config.TRAINING['task_type']

        # 生成或加载数据
        if synthetic:
            self.data, self.labels = self._generate_synthetic_data()
        else:
            # 加载文件信息并预加载所有数据
            self._preload_real_data()

    def _preload_real_data(self):
        """预加载所有真实数据并转换为张量"""
        self.file_info = _load_real_data_info(self.real_data_path, self.logger)
        self.all_samples = []
        self.all_labels = []

        # 遍历所有文件加载数据
        for file_idx, (file_path, frequency, num_samples) in enumerate(self.file_info):
            # 加载文件数据
            processed_path = os.path.join(
                os.path.dirname(file_path),
                f"processed_{frequency}.npz"
            )

            if os.path.exists(processed_path):
                with np.load(processed_path) as data:
                    samples = data['samples']
            else:
                waveform_data = load_and_process_tdms(file_path, self.logger)
                samples = sliding_window_processing(waveform_data, window_size=3)
                np.savez_compressed(processed_path, samples=samples)

            # 转换为张量并调整形状
            samples_tensor = torch.tensor(samples, dtype=torch.float32)
            # 调整形状: [样本数, 序列长度, 1]
            samples_tensor = samples_tensor.view(len(samples), -1, 1)

            # 创建对应频率标签
            if self.task_type == 'classification':
                # 分类任务 - 将频率转换为类别
                thresholds = self.config.MODEL['classifier_thresholds']
                class_idx = int(np.digitize(frequency, thresholds))
                labels_tensor = torch.full((len(samples),), class_idx, dtype=torch.long)
            else:
                # 回归任务
                labels_tensor = torch.full((len(samples),), frequency, dtype=torch.float32)

            self.all_samples.append(samples_tensor)
            self.all_labels.append(labels_tensor)

        # 合并所有数据
        self.data = torch.cat(self.all_samples, dim=0)
        self.labels = torch.cat(self.all_labels, dim=0)
        self.total_samples = len(self.data)

        # 清理临时数据
        del self.all_samples, self.all_labels

    def _generate_synthetic_data(self):
        """生成合成数据并直接转换为张量"""
        X = torch.zeros(self.num_samples, self.seq_length, 1)
        y = torch.zeros(self.num_samples)

        for i in range(self.num_samples):
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

            # 存储为张量
            X[i, :, 0] = torch.tensor(processed_signal, dtype=torch.float32)
            if self.task_type == 'classification':
                # 分类任务 - 将频率转换为类别
                thresholds = self.config.MODEL['classifier_thresholds']
                class_idx = int(np.digitize(freq, thresholds))
                y[i] = class_idx
            else:
                # 回归任务
                y[i] = freq

        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if self.task_type == 'classification':
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

    def __len__(self):
        return self.num_samples if self.synthetic else self.total_samples

    def __getitem__(self, idx):
        # 直接返回预处理的张量
        return self.data[idx], self.labels[idx]


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