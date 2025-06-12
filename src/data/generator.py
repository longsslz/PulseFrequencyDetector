# src/data/generator.py
import numpy as np
from scipy.signal import butter, filtfilt


def generate_synthetic_pulse(frequency, num_samples, sampling_rate, duty_cycle=0.2):
    """
    生成合成脉冲信号

    参数:
        frequency (float): 脉冲频率 (Hz)
        num_samples (int): 样本数量
        sampling_rate (float): 采样率 (Hz)
        duty_cycle (float): 脉冲占空比 (0-1)

    返回:
        np.ndarray: 生成的脉冲信号
    """
    # 创建时间序列
    t = np.linspace(0, num_samples / sampling_rate, num_samples)

    # 生成方波脉冲信号
    pulse_signal = np.zeros(num_samples)

    # 计算脉冲周期和宽度
    period = 1.0 / frequency
    pulse_width = period * duty_cycle

    # 生成脉冲
    for i in range(num_samples):
        # 计算当前时间在周期中的位置
        time_in_period = t[i] % period

        # 如果时间在脉冲宽度内，则为高电平
        if time_in_period < pulse_width:
            pulse_signal[i] = 1.0

    return pulse_signal


def generate_frequency_varying_pulse(base_freq, num_samples, sampling_rate, variation_amp=5, variation_freq=0.1):
    """
    生成频率变化的脉冲信号

    参数:
        base_freq (float): 基础频率 (Hz)
        num_samples (int): 样本数量
        sampling_rate (float): 采样率 (Hz)
        variation_amp (float): 频率变化幅度 (Hz)
        variation_freq (float): 频率变化频率 (Hz)

    返回:
        (np.ndarray, np.ndarray): (生成的信号, 实际频率数组)
    """
    t = np.linspace(0, num_samples / sampling_rate, num_samples)

    # 创建频率变化
    freq_variation = variation_amp * np.sin(2 * np.pi * variation_freq * t) + base_freq
    freq_variation = np.clip(freq_variation, 0.5, sampling_rate / 2)  # 限制在合理范围内

    # 生成脉冲信号
    pulse_signal = np.zeros(num_samples)
    phase = 0

    for i in range(num_samples):
        # 在每个时间点，频率可能不同
        if i == 0:
            dt = 1 / sampling_rate
        else:
            dt = t[i] - t[i - 1]

        # 更新相位
        phase += 2 * np.pi * freq_variation[i] * dt

        # 当相位超过2π时生成脉冲
        if phase >= 2 * np.pi:
            pulse_signal[i] = 1.0
            phase -= 2 * np.pi  # 重置相位

    return pulse_signal, freq_variation


def add_noise_to_signal(signal, noise_level=0.2, noise_type='gaussian'):
    """
    向信号添加噪声

    参数:
        signal (np.ndarray): 输入信号
        noise_level (float): 噪声水平 (0-1)
        noise_type (str): 噪声类型 ('gaussian', 'impulse', 'uniform')

    返回:
        np.ndarray: 带噪声的信号
    """
    noisy_signal = signal.copy()

    if noise_type == 'gaussian':
        # 高斯噪声
        noise = np.random.normal(0, noise_level, len(signal))
        noisy_signal += noise

    elif noise_type == 'impulse':
        # 脉冲噪声
        noise_mask = np.random.rand(len(signal)) < noise_level
        impulse_values = np.random.choice([-1, 1], size=np.sum(noise_mask)) * np.random.uniform(0.5, 1.0,
                                                                                                np.sum(noise_mask))
        noisy_signal[noise_mask] += impulse_values

    elif noise_type == 'uniform':
        # 均匀噪声
        noise = np.random.uniform(-noise_level, noise_level, len(signal))
        noisy_signal += noise

    # 确保信号在0-1范围内
    noisy_signal = np.clip(noisy_signal, 0, 1)

    return noisy_signal


def add_missing_pulses(signal, missing_prob=0.05):
    """
    随机添加缺失脉冲

    参数:
        signal (np.ndarray): 输入信号
        missing_prob (float): 脉冲缺失概率

    返回:
        np.ndarray: 带缺失脉冲的信号
    """
    # 创建缺失掩码（只影响脉冲位置）
    pulse_positions = np.where(signal > 0.5)[0]
    missing_mask = np.random.rand(len(pulse_positions)) < missing_prob
    missing_indices = pulse_positions[missing_mask]

    # 将脉冲位置设置为0
    modified_signal = signal.copy()
    modified_signal[missing_indices] = 0

    return modified_signal


def filter_signal(signal, filter_type='lowpass', cutoff_freq=0.1, sampling_rate=100):
    """
    对信号进行滤波

    参数:
        signal (np.ndarray): 输入信号
        filter_type (str): 滤波器类型 ('lowpass', 'highpass', 'bandpass')
        cutoff_freq (float or tuple): 截止频率 (Hz)
        sampling_rate (float): 采样率 (Hz)

    返回:
        np.ndarray: 滤波后的信号
    """
    # 计算归一化截止频率
    nyquist = 0.5 * sampling_rate

    if filter_type == 'lowpass':
        # 低通滤波
        b, a = butter(3, cutoff_freq / nyquist, btype='low')
    elif filter_type == 'highpass':
        # 高通滤波
        b, a = butter(3, cutoff_freq / nyquist, btype='high')
    elif filter_type == 'bandpass':
        # 带通滤波
        b, a = butter(3, [cutoff_freq[0] / nyquist, cutoff_freq[1] / nyquist], btype='band')
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    # 应用滤波器
    filtered_signal = filtfilt(b, a, signal)

    # 确保信号在0-1范围内
    filtered_signal = np.clip(filtered_signal, 0, 1)

    return filtered_signal