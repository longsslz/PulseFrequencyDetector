# src/data/preprocessor.py
import numpy as np
import torch


def normalize_signal(signal, method='minmax'):
    """
    归一化信号

    参数:
        signal (np.ndarray): 输入信号
        method (str): 归一化方法 ('minmax', 'zscore', 'unit_range')

    返回:
        np.ndarray: 归一化后的信号
    """
    if method == 'minmax':
        # Min-Max归一化 [0, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val > 1e-6:
            normalized = (signal - min_val) / (max_val - min_val)
        else:
            normalized = signal - min_val
    elif method == 'zscore':
        # Z-score归一化 (均值0，标准差1)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val > 1e-6:
            normalized = (signal - mean_val) / std_val
        else:
            normalized = signal - mean_val
    elif method == 'unit_range':
        # 单位范围归一化 [-1, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val > 1e-6:
            normalized = 2 * ((signal - min_val) / (max_val - min_val)) - 1
        else:
            normalized = signal - min_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def sliding_window(signal, window_size, step_size=1):
    """
    将信号分割为滑动窗口

    参数:
        signal (np.ndarray): 输入信号 (1D)
        window_size (int): 窗口大小
        step_size (int): 步长

    返回:
        list: 窗口列表
    """
    windows = []
    for start_idx in range(0, len(signal) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = signal[start_idx:end_idx]
        windows.append(window)

    return windows


def convert_timestamps_to_binary(timestamps, sampling_rate, duration=None, min_time=None, max_time=None):
    """
    将时间戳列表转换为二进制序列

    参数:
        timestamps (list): 脉冲时间戳列表 (秒)
        sampling_rate (float): 采样率 (Hz)
        duration (float): 总持续时间 (秒)
        min_time (float): 最小时间戳
        max_time (float): 最大时间戳

    返回:
        (np.ndarray, float, float): (二进制序列, 实际最小时间, 实际最大时间)
    """
    if not timestamps:
        if duration is not None:
            num_samples = int(duration * sampling_rate)
            return np.zeros(num_samples), 0, duration
        else:
            raise ValueError("Either timestamps or duration must be provided")

    # 确定时间范围
    actual_min_time = min_time if min_time is not None else min(timestamps)
    actual_max_time = max_time if max_time is not None else max(timestamps)

    # 如果提供了duration，使用它来确定结束时间
    if duration is not None:
        actual_max_time = actual_min_time + duration

    total_duration = actual_max_time - actual_min_time
    num_samples = int(total_duration * sampling_rate)

    # 创建时间网格
    t_grid = np.linspace(actual_min_time, actual_max_time, num_samples)

    # 生成二进制序列
    binary_seq = np.zeros(num_samples)

    # 找到每个时间戳最近的索引
    for ts in timestamps:
        if actual_min_time <= ts <= actual_max_time:
            idx = int((ts - actual_min_time) / total_duration * (num_samples - 1))
            binary_seq[idx] = 1.0

    return binary_seq, actual_min_time, actual_max_time


def calculate_frequency_label(timestamps, method='mean_interval'):
    """
    计算时间戳序列的频率标签

    参数:
        timestamps (list): 时间戳列表
        method (str): 计算方法 ('mean_interval', 'total_count')

    返回:
        float: 估计频率 (Hz)
    """
    if len(timestamps) < 2:
        return 0.0

    if method == 'mean_interval':
        # 基于平均间隔计算频率
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        return 1.0 / mean_interval if mean_interval > 1e-6 else 0.0

    elif method == 'total_count':
        # 基于总计数计算频率
        total_time = max(timestamps) - min(timestamps)
        num_pulses = len(timestamps)
        return num_pulses / total_time if total_time > 1e-6 else 0.0

    else:
        raise ValueError(f"Unknown frequency calculation method: {method}")


def create_tensor_from_signal(signal, seq_length=None, step_size=None, device='cpu'):
    """
    从信号创建PyTorch张量，可选择分割为序列

    参数:
        signal (np.ndarray): 输入信号
        seq_length (int): 序列长度 (如果为None，则返回完整信号)
        step_size (int): 步长 (仅当seq_length不为None时使用)
        device (str): 目标设备 ('cpu', 'cuda')

    返回:
        torch.Tensor: 输出张量
    """
    if seq_length is None:
        # 返回完整信号作为单个序列
        tensor = torch.tensor(signal, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

    # 分割为序列
    sequences = []
    for start_idx in range(0, len(signal) - seq_length + 1, step_size or 1):
        end_idx = start_idx + seq_length
        seq = signal[start_idx:end_idx]
        sequences.append(seq)

    # 转换为张量
    tensor = torch.tensor(np.array(sequences), dtype=torch.float32, device=device)
    return tensor.unsqueeze(-1)  # (batch, seq_length, 1)