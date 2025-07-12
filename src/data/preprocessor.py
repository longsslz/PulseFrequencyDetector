# src/data/preprocessor.py
import os

import numpy as np
import torch
from nptdms import TdmsFile

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


def load_and_process_tdms(file_path):
    """
    从TDMS文件中提取所有组的波形数据并转换为NumPy数组

    参数:
        file_path (str): TDMS文件路径

    返回:
        numpy.ndarray: 三维波形数据，形状为(组数, 通道数, 数据点数)
    """
    # 读取TDMS文件
    tdms_file = TdmsFile.read(file_path)
    # 存储所有组的时间数据和幅值数据
    time_arrays = []
    amplitude_arrays = []

    # 获取所有组
    groups = tdms_file.groups()
    num_groups = len(groups)

    # 检查第一个组以确定通道数和数据点数
    first_group = groups[0]

    num_channels = len(first_group.channels())
    num_points = len(first_group.channels()[0].data)

    # 创建三维数组 (组数, 通道数, 数据点数)
    data_3d = np.zeros((num_groups, num_channels, num_points))

    # 填充数据
    for group_idx, group in enumerate(groups):
        for channel_idx, channel in enumerate(group.channels()):
            data_3d[group_idx, channel_idx, :] = channel.data

    return data_3d




def sliding_window_processing(waveform_data, window_size=3):
    """
    使用滑动窗口处理波形数据，生成样本

    参数:
        waveform_data (numpy.ndarray): 三维波形数据，形状为(组数, 通道数, 数据点数)
        window_size (int): 滑动窗口大小

    返回:
        numpy.ndarray: 样本数组，形状为(样本数, 窗口大小 × 数据点数)
    """
    num_groups = waveform_data.shape[0]

    # 检查是否有足够的数据
    if num_groups < window_size:
        raise ValueError(f"数据组数({num_groups})小于窗口大小({window_size})")

    # 计算样本数量
    num_samples = num_groups - window_size + 1

    # 初始化样本和标签数组
    samples = np.zeros((num_samples, window_size * waveform_data.shape[2]))
    #labels = np.zeros(num_samples) if label is None else np.zeros(num_samples, dtype=object)

    # 滑动窗口处理
    for i in range(num_samples):
        # 获取当前窗口内的波形组
        window_groups = waveform_data[i:i + window_size].copy()

        # 获取第一个波形的起始时间作为参考点
        first_time_start = window_groups[0, 0, 0]

        # 提取时间数据并转换为相对时间
        time_data = np.zeros((window_size, waveform_data.shape[2]))

        # 将所有波形的时间转换为相对于第一个波形的相对时间
        for j in range(window_size):

            # 转换为相对于第一个波形的相对时间
            time_data[j, :] = window_groups[j, 0, :] - first_time_start

        # 存储样本（只包含时间数据）
        samples[i] = time_data.flatten()


    # 归一化处理
    min_val = np.min(samples, axis=1, keepdims=True)
    max_val = np.max(samples, axis=1, keepdims=True)
    samples = (samples - min_val) / (max_val - min_val)
    return samples


def process_tdms_folder(folder_path, window_size=3):
    """
    批量处理文件夹中的所有TDMS文件

    参数:
        folder_path (str): 包含TDMS文件的文件夹路径
        window_size (int): 滑动窗口大小

    返回:
        tuple: (样本数组, 标签数组)
    """
    all_samples = []
    all_labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tdms'):
            file_path = os.path.join(folder_path, filename)

            try:
                # 从文件名提取频率标签
                frequency = float(os.path.splitext(filename)[0])

                # 加载和处理文件
                waveform_data = load_and_process_tdms(file_path)
                # 使用滑动窗口处理数据
                samples = sliding_window_processing(waveform_data, window_size)

                # 为每个样本创建标签（相同的频率）
                labels = np.full(samples.shape[0], frequency)

                # 添加到总数据集
                all_samples.append(samples)
                all_labels.append(labels)

                print(f"成功处理文件: {filename}, 频率: {frequency}Hz")
                print(f"生成样本数: {samples.shape[0]}, 时间向量长度: {samples.shape[1]}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    # 合并所有样本和标签
    if all_samples:
        samples_array = np.concatenate(all_samples, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        return samples_array, labels_array
    else:
        return np.array([]), np.array([])