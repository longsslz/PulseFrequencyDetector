# src/data/preprocessor.py
import os
# 在文件开头添加
from functools import lru_cache

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


@lru_cache(maxsize=32)
def load_and_process_tdms(file_path, logger):
    """
    加载并处理单个TDMS文件，处理空数据通道并排除没有数据的组

    参数:
        file_path (str): TDMS文件路径
        logger: 日志记录器对象

    返回:
        numpy.ndarray: 三维波形数据，形状为(有效组数, 通道数, 数据点数)
    """
    try:
        # 读取TDMS文件
        tdms_file = TdmsFile.read(file_path)

        # 获取所有组
        groups = tdms_file.groups()
        valid_groups = []  # 存储有有效数据的组

        # 第一次遍历：筛选有有效数据的组
        for group in groups:
            has_valid_data = False
            for channel in group.channels():
                if len(channel.data) > 0:
                    has_valid_data = True
                    break

            if has_valid_data:
                valid_groups.append(group)

        num_valid_groups = len(valid_groups)

        if num_valid_groups == 0:
            logger.warning(f"文件 {file_path} 中没有找到任何有效数据组")
            return np.array([])

        # 确定最大通道数和最大数据点数
        max_channels = 0
        max_points = 0

        for group in valid_groups:
            channels = group.channels()
            num_channels = len(channels)
            max_channels = max(max_channels, num_channels)

            for channel in channels:
                if len(channel.data) > 0:
                    max_points = max(max_points, len(channel.data))

        if max_channels == 0 or max_points == 0:
            logger.warning(f"文件 {file_path} 中没有找到有效数据")
            return np.array([])

        # 创建三维数组 (有效组数, 最大通道数, 最大数据点数)
        data_3d = np.zeros((num_valid_groups, max_channels, max_points))

        # 填充数据
        for group_idx, group in enumerate(valid_groups):
            channels = group.channels()

            for channel_idx, channel in enumerate(channels):
                if channel_idx >= max_channels:
                    break

                channel_data = channel.data
                if len(channel_data) > 0:
                    # 只填充有效数据部分
                    end_idx = min(len(channel_data), max_points)
                    data_3d[group_idx, channel_idx, :end_idx] = channel_data[:end_idx]
                else:
                    logger.debug(f"组 {group.name} 通道 {channel.name} 无数据 - 跳过")

        return data_3d
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return np.array([])


def sliding_window_processing(waveform_data, window_size=3):
    """
    使用滑动窗口处理波形数据，生成样本

    参数:
        waveform_data (numpy.ndarray): 三维波形数据，形状为(组数, 通道数, 数据点数)
        window_size (int): 滑动窗口大小

    返回:
        numpy.ndarray: 样本数组，形状为(样本数, 窗口大小 × 数据点数)
    """
    if waveform_data.size == 0:
        return np.array([])

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
    # min_val = np.min(samples, axis=1, keepdims=True)
    # max_val = np.max(samples, axis=1, keepdims=True)
    # samples = (samples - min_val) / (max_val - min_val)
    samples = normalize_signal(samples)
    return samples


def process_tdms_folder(folder_path, window_size=3, save_processed=False, logger = None):
    """
     批量处理文件夹中的所有TDMS文件，返回文件路径列表而不是完整数据

    参数:
        folder_path (str): 包含TDMS文件的文件夹路径
        window_size (int): 滑动窗口大小
        save_processed (bool): 是否保存处理后的数据

    返回:
        list: 文件信息列表，每个元素为(文件路径, 频率, 样本数)
    """

    file_info = []
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")
        return file_info

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.tdms'):
            file_path = os.path.join(folder_path, filename)

            try:
                # 从文件名提取频率标签
                frequency = float(os.path.splitext(filename)[0])

                # 处理后的数据保存路径
                processed_path = os.path.join(folder_path, f"processed_{frequency}.npz")
                # 如果已存在处理好的数据，直接使用
                if os.path.exists(processed_path):
                    # 加载样本数量信息
                    with np.load(processed_path) as data:
                        num_samples = data['samples'].shape[0]
                    print(f"使用预处理的文件: {filename}, 频率: {frequency}Hz, 样本数: {num_samples}")
                else:
                    # 加载和处理文件
                    waveform_data = load_and_process_tdms(file_path, logger)
                    # 使用滑动窗口处理数据
                    samples = sliding_window_processing(waveform_data, window_size)

                    if samples.size == 0:
                        logger.warning(f"文件 {filename} 滑动窗口处理失败，跳过")
                        continue

                    num_samples = samples.shape[0]

                    if save_processed:
                        # 保存处理后的数据
                        np.savez_compressed(processed_path, samples=samples)
                        print(f"保存处理后的文件: {processed_path}, 样本数: {num_samples}")

                    # 及时释放内存
                    del waveform_data, samples

                file_info.append((file_path, frequency, num_samples))
                logger.info(f"处理文件: {filename}, 频率: {frequency}Hz, 样本数: {num_samples}")

            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {str(e)}")

    return file_info