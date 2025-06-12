# tests/conftest.py
import pytest
import numpy as np
import torch
from src.config import Config


@pytest.fixture
def config():
    """提供配置对象fixture"""
    return Config()


@pytest.fixture
def synthetic_pulse_data():
    """生成合成脉冲数据fixture"""
    np.random.seed(42)
    sampling_rate = 100
    duration = 1.0  # 秒
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)

    # 生成5Hz脉冲信号
    freq = 5.0
    pulse_signal = np.zeros(num_samples)
    for i in range(1, num_samples):
        dt = t[i] - t[i - 1]
        phase_increment = 2 * np.pi * freq * dt
        pulse_signal[i] = 1.0 if (pulse_signal[i - 1] + phase_increment) % (2 * np.pi) < np.pi else 0.0

    # 添加噪声
    noisy_signal = pulse_signal + np.random.normal(0, 0.1, num_samples)
    noisy_signal = np.clip(noisy_signal, 0, 1)

    return {
        'clean_signal': pulse_signal,
        'noisy_signal': noisy_signal,
        'sampling_rate': sampling_rate,
        'true_frequency': freq
    }


@pytest.fixture
def sample_model(config):
    """提供测试模型fixture"""
    from src.model import create_model
    model = create_model('standard', config)
    return model