# tests/test_data.py
import pytest
import numpy as np
import torch

from src.data.dataset import PulseDataset
from src.data.generator import (
    generate_synthetic_pulse,
    add_noise_to_signal,
    add_missing_pulses,
    filter_signal
)
from src.data.preprocessor import (
    normalize_signal,
    sliding_window,
    convert_timestamps_to_binary,
    calculate_frequency_label
)


class TestDataGenerator:
    def test_generate_synthetic_pulse(self):
        """测试合成脉冲生成"""
        freq = 10
        num_samples = 1000
        sampling_rate = 100
        signal = generate_synthetic_pulse(freq, num_samples, sampling_rate)

        assert len(signal) == num_samples
        assert np.max(signal) == 1.0
        assert np.min(signal) == 0.0

        # 检查频率是否正确
        pulse_indices = np.where(signal > 0.5)[0]
        intervals = np.diff(pulse_indices) / sampling_rate
        estimated_freq = 1.0 / np.mean(intervals)
        assert pytest.approx(estimated_freq, rel=0.1) == freq

    def test_add_noise_to_signal(self):
        """测试添加噪声"""
        clean_signal = np.zeros(100)
        clean_signal[::10] = 1.0  # 10Hz脉冲

        noisy_signal = add_noise_to_signal(clean_signal, noise_level=0.2)

        assert noisy_signal.shape == clean_signal.shape
        assert not np.array_equal(noisy_signal, clean_signal)
        assert np.max(noisy_signal) <= 1.0
        assert np.min(noisy_signal) >= 0.0

    def test_filter_signal(self):
        """测试信号滤波"""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        filtered = filter_signal(signal, 'lowpass', cutoff_freq=10, sampling_rate=1000)

        assert filtered.shape == signal.shape
        # 高频分量应该被衰减
        assert np.max(np.abs(filtered)) < np.max(np.abs(signal))


class TestDataset:
    def test_pulse_dataset(self, config):
        """测试数据集类"""
        dataset = PulseDataset(config, num_samples=10)

        assert len(dataset) == 10

        # 检查样本格式
        signal, freq = dataset[0]
        assert signal.shape == (config.DATA['seq_length'], 1)
        assert isinstance(freq, torch.Tensor)
        assert 0.5 <= freq.item() <= config.DATA['max_freq']


class TestPreprocessor:
    def test_normalize_signal(self):
        """测试信号归一化"""
        signal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        normalized = normalize_signal(signal)

        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        assert pytest.approx(normalized[2]) == 0.5

    def test_sliding_window(self):
        """测试滑动窗口"""
        data = np.arange(10)
        windows = sliding_window(data, window_size=5, step_size=2)

        assert len(windows) == 3
        assert np.array_equal(windows[0], np.array([0, 1, 2, 3, 4]))
        assert np.array_equal(windows[1], np.array([2, 3, 4, 5, 6]))
        assert np.array_equal(windows[2], np.array([4, 5, 6, 7, 8]))

    def test_timestamps_to_binary(self):
        """测试时间戳转换"""
        timestamps = [0.1, 0.3, 0.5]
        binary_seq, min_t, max_t = convert_timestamps_to_binary(
            timestamps, sampling_rate=10, duration=1.0
        )

        assert len(binary_seq) == 10
        assert binary_seq[1] == 1.0  # 0.1s -> 第1个样本
        assert binary_seq[3] == 1.0  # 0.3s -> 第3个样本
        assert binary_seq[5] == 1.0  # 0.5s -> 第5个样本
        assert min_t == 0.0
        assert max_t == 1.0

    def test_calculate_frequency_label(self):
        """测试频率计算"""
        timestamps = [0.1, 0.3, 0.5, 0.7]

        # 平均间隔方法
        freq1 = calculate_frequency_label(timestamps, method='mean_interval')
        assert pytest.approx(freq1) == 5.0  # 0.2s间隔 -> 5Hz

        # 总数方法
        freq2 = calculate_frequency_label(timestamps, method='total_count')
        assert pytest.approx(freq2) == 5.0  # 4脉冲/0.8s = 5Hz