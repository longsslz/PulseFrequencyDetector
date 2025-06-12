# src/processing/utils.py
import numpy as np
from scipy import signal, fft


def sliding_window(data, window_size, step_size=1):
    """
    将数据分割为滑动窗口

    参数:
        data (np.ndarray): 输入数据 (1D)
        window_size (int): 窗口大小
        step_size (int): 步长

    返回:
        list: 窗口列表
    """
    windows = []
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        windows.append(data[start_idx:end_idx])
    return windows


def compute_instant_frequency(pulse_signal, sampling_rate, method='autocorrelation'):
    """
    计算瞬时脉冲频率

    参数:
        pulse_signal (np.ndarray): 脉冲信号 (1D)
        sampling_rate (float): 采样率 (Hz)
        method (str): 计算方法 ('autocorrelation', 'fft', 'zero_crossing')

    返回:
        float: 估计频率 (Hz)
    """
    if len(pulse_signal) < 2:
        return 0.0

    if method == 'autocorrelation':
        # 自相关方法
        corr = np.correlate(pulse_signal, pulse_signal, mode='full')
        corr = corr[len(corr) // 2:]  # 只取正延迟

        # 找到第一个峰值后的第一个零点
        peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
        if len(peaks) == 0:
            return 0.0

        first_peak = peaks[0]
        zero_crossings = np.where(np.diff(np.sign(corr[first_peak:])))[0]

        if len(zero_crossings) == 0:
            return 0.0

        period = zero_crossings[0] / sampling_rate
        return 1.0 / period if period > 0 else 0.0

    elif method == 'fft':
        # FFT方法
        n = len(pulse_signal)
        yf = fft.fft(pulse_signal)
        xf = fft.fftfreq(n, 1 / sampling_rate)[:n // 2]

        # 找到主频
        idx = np.argmax(np.abs(yf[:n // 2]))
        return abs(xf[idx])

    elif method == 'zero_crossing':
        # 过零检测方法
        zero_crossings = np.where(np.diff(np.sign(pulse_signal - 0.5)))[0]
        if len(zero_crossings) < 2:
            return 0.0

        intervals = np.diff(zero_crossings) / sampling_rate
        avg_interval = np.mean(intervals)
        return 1.0 / (2 * avg_interval)  # 每个周期两次过零

    else:
        raise ValueError(f"Unknown frequency computation method: {method}")


def signal_quality_check(signal_data, sampling_rate, min_freq=0.5, max_freq=50):
    """
    评估信号质量

    参数:
        signal_data (np.ndarray): 信号数据
        sampling_rate (float): 采样率 (Hz)
        min_freq (float): 最小有效频率 (Hz)
        max_freq (float): 最大有效频率 (Hz)

    返回:
        float: 信号质量分数 (0-1)
    """
    if len(signal_data) < 10:
        return 0.0

    # 1. 计算信号能量
    energy = np.mean(signal_data ** 2)

    # 2. 计算有效频段能量比
    n = len(signal_data)
    yf = fft.fft(signal_data)
    xf = fft.fftfreq(n, 1 / sampling_rate)[:n // 2]

    # 有效频段掩码
    mask = (xf >= min_freq) & (xf <= max_freq)
    useful_energy = np.sum(np.abs(yf[:n // 2][mask]) ** 2)
    total_energy = np.sum(np.abs(yf[:n // 2]) ** 2)
    spectral_ratio = useful_energy / total_energy if total_energy > 0 else 0

    # 3. 计算峰均比 (PAPR)
    peak = np.max(signal_data)
    rms = np.sqrt(np.mean(signal_data ** 2))
    papr = peak / rms if rms > 0 else 0

    # 4. 综合质量分数
    quality = 0.4 * energy + 0.4 * spectral_ratio + 0.2 * (1 / (1 + np.exp(-(papr - 2))))

    return np.clip(quality, 0, 1)


def filter_signal_realtime(signal_data, filter_type='lowpass', cutoff=0.2, sampling_rate=100):
    """
    实时信号滤波

    参数:
        signal_data (np.ndarray): 输入信号
        filter_type (str): 滤波器类型 ('lowpass', 'highpass', 'bandpass')
        cutoff (float or tuple): 截止频率
        sampling_rate (float): 采样率 (Hz)

    返回:
        np.ndarray: 滤波后的信号
    """
    nyquist = 0.5 * sampling_rate

    if filter_type == 'lowpass':
        b, a = signal.butter(3, cutoff / nyquist, btype='low')
    elif filter_type == 'highpass':
        b, a = signal.butter(3, cutoff / nyquist, btype='high')
    elif filter_type == 'bandpass':
        b, a = signal.butter(3, [cutoff[0] / nyquist, cutoff[1] / nyquist], btype='band')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # 使用零相位滤波
    filtered = signal.filtfilt(b, a, signal_data)

    return np.clip(filtered, 0, 1)


def resample_signal(signal_data, original_rate, target_rate):
    """
    信号重采样

    参数:
        signal_data (np.ndarray): 输入信号
        original_rate (float): 原始采样率 (Hz)
        target_rate (float): 目标采样率 (Hz)

    返回:
        np.ndarray: 重采样后的信号
    """
    if original_rate == target_rate:
        return signal_data.copy()

    num_samples = int(len(signal_data) * target_rate / original_rate)
    return signal.resample(signal_data, num_samples)


def detect_pulse_peaks(signal_data, threshold=0.5, min_distance=10):
    """
    检测脉冲峰值

    参数:
        signal_data (np.ndarray): 输入信号
        threshold (float): 检测阈值
        min_distance (int): 最小峰间距离 (样本数)

    返回:
        np.ndarray: 峰值位置索引
    """
    peaks, _ = signal.find_peaks(
        signal_data,
        height=threshold,
        distance=min_distance
    )
    return peaks


def calculate_pulse_stats(pulse_signal, sampling_rate):
    """
    计算脉冲统计信息

    参数:
        pulse_signal (np.ndarray): 脉冲信号
        sampling_rate (float): 采样率 (Hz)

    返回:
        dict: 包含脉冲统计信息的字典
    """
    stats = {
        'mean_amplitude': np.mean(pulse_signal),
        'max_amplitude': np.max(pulse_signal),
        'min_amplitude': np.min(pulse_signal),
        'pulse_count': 0,
        'mean_frequency': 0,
        'duty_cycle': 0
    }

    # 检测脉冲
    peaks = detect_pulse_peaks(pulse_signal)
    stats['pulse_count'] = len(peaks)

    if len(peaks) > 1:
        # 计算频率
        intervals = np.diff(peaks) / sampling_rate
        stats['mean_frequency'] = 1.0 / np.mean(intervals)

        # 计算占空比 (粗略估计)
        pulse_widths = []
        for peak in peaks:
            left = max(0, peak - 1)
            while left > 0 and pulse_signal[left] > 0.5:
                left -= 1

            right = min(len(pulse_signal) - 1, peak + 1)
            while right < len(pulse_signal) - 1 and pulse_signal[right] > 0.5:
                right += 1

            pulse_widths.append(right - left)

        stats['duty_cycle'] = np.mean(pulse_widths) / (1 / stats['mean_frequency'] * sampling_rate)

    return stats