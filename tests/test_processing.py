# tests/test_processing.py
import time

import pytest
import numpy as np
from src.processing import (
    RealTimePulseProcessor,
    AsyncPulseProcessor
)
from src.processing.utils import (
    compute_instant_frequency,
    signal_quality_check,
    detect_pulse_peaks
)


class TestRealTimeProcessor:
    def test_processor_initialization(self, config):
        """测试处理器初始化"""
        processor = RealTimePulseProcessor(config=config)

        assert processor is not None
        assert processor.buffer.shape == (1, config.REALTIME['window_size'], 1)
        assert processor.current_index == 0
        assert not processor.window_full

    def test_data_processing(self, config, synthetic_pulse_data):
        """测试数据处理"""
        processor = RealTimePulseProcessor(config=config)
        signal = synthetic_pulse_data['noisy_signal']

        # 处理前半个窗口
        for i in range(config.REALTIME['window_size'] // 2):
            result = processor.add_data_point(signal[i])
            assert result is None

        # 处理完整窗口
        for i in range(config.REALTIME['window_size'] // 2, config.REALTIME['window_size']):
            result = processor.add_data_point(signal[i])

        assert result is not None
        assert 4.5 <= result <= 5.5  # 应该在真实频率附近

    def test_reset(self, config):
        """测试处理器重置"""
        processor = RealTimePulseProcessor(config=config)

        # 添加一些数据
        for i in range(10):
            processor.add_data_point(0.5)

        # 重置
        processor.reset()

        assert processor.current_index == 0
        assert not processor.window_full
        assert len(processor.frequency_history) == 0


class TestAsyncProcessor:
    def test_async_processing(self, config, synthetic_pulse_data):
        """测试异步处理"""
        processor = AsyncPulseProcessor(config=config)
        processor.start()

        signal = synthetic_pulse_data['noisy_signal']

        # 发送数据
        for sample in signal[:200]:  # 只发送前200个样本测试
            processor.add_data_point(sample)
            time.sleep(0.001)  # 模拟实时延迟

        # 获取结果
        results = processor.get_results()
        assert len(results) > 0

        # 检查频率估计
        freqs = [r['frequency'] for r in results]
        assert 4.5 <= np.mean(freqs) <= 5.5

        processor.stop()


class TestProcessingUtils:
    def test_compute_instant_frequency(self, synthetic_pulse_data):
        """测试瞬时频率计算"""
        signal = synthetic_pulse_data['clean_signal']
        sampling_rate = synthetic_pulse_data['sampling_rate']

        freq = compute_instant_frequency(signal, sampling_rate)
        assert pytest.approx(freq, rel=0.1) == synthetic_pulse_data['true_frequency']

    def test_signal_quality_check(self, synthetic_pulse_data):
        """测试信号质量评估"""
        clean_signal = synthetic_pulse_data['clean_signal']
        noisy_signal = synthetic_pulse_data['noisy_signal']
        sampling_rate = synthetic_pulse_data['sampling_rate']

        clean_quality = signal_quality_check(clean_signal, sampling_rate)
        noisy_quality = signal_quality_check(noisy_signal, sampling_rate)

        assert 0 <= clean_quality <= 1
        assert 0 <= noisy_quality <= 1
        assert clean_quality > noisy_quality

    def test_detect_pulse_peaks(self):
        """测试脉冲峰值检测"""
        signal = np.zeros(100)
        signal[::10] = 1.0  # 每10个样本一个脉冲

        peaks = detect_pulse_peaks(signal)
        assert len(peaks) == 10
        assert np.array_equal(peaks, np.arange(0, 100, 10))