# src/processing/realtime.py
import time
import threading
import numpy as np
import torch
from collections import deque
from src.config import Config
from src.model import create_model, load_model_onnx


class RealTimePulseProcessor:
    """实时脉冲信号处理器"""

    def __init__(self, model_path=None, config=None, device='auto', use_onnx=False):
        """
        初始化实时处理器

        参数:
            model_path (str): 模型文件路径
            config (Config): 配置对象
            device (str): 计算设备 ('cpu', 'cuda', 'auto')
            use_onnx (bool): 是否使用ONNX模型
        """
        self.config = config or Config()
        self.device = self._get_device(device)
        self.use_onnx = use_onnx

        # 初始化模型
        self.model = self._load_model(model_path)

        # 初始化处理状态
        self._init_processing_state()

        # 性能统计
        self.processing_times = deque(maxlen=100)
        self.sample_counter = 0
        self.start_time = time.time()

    def _get_device(self, device):
        """确定使用的设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_model(self, model_path):
        """加载模型"""
        if self.use_onnx:
            # 加载ONNX模型
            if model_path is None:
                model_path = self.config.PATHS['model_dir'] + self.config.PATHS['onnx_model']

            return load_model_onnx(model_path)
        else:
            # 加载PyTorch模型
            if model_path is None:
                model_path = self.config.PATHS['model_dir'] + self.config.PATHS['quantized_model']

            model = create_model('standard', self.config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model

    def _init_processing_state(self):
        """初始化处理状态"""
        self.window_size = self.config.REALTIME['window_size']
        self.sampling_rate = self.config.REALTIME['sampling_rate']
        self.max_freq = self.config.REALTIME['max_freq']

        # 创建处理缓冲区
        if self.use_onnx:
            self.buffer = np.zeros((1, self.window_size, 1), dtype=np.float32)
        else:
            self.buffer = torch.zeros((1, self.window_size, 1),
                                      dtype=torch.float32,
                                      device=self.device)

        self.current_index = 0
        self.window_full = False

        # 历史记录
        self.frequency_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.signal_quality = deque(maxlen=100)

    def add_data_point(self, data_point):
        """
        添加一个新的数据点到处理缓冲区

        参数:
            data_point (float): 新的数据点值 (0-1范围)

        返回:
            float or None: 如果窗口已满则返回频率估计，否则返回None
        """
        start_time = time.time()

        # 更新缓冲区
        if self.current_index < self.window_size:
            self.buffer[0, self.current_index, 0] = data_point
            self.current_index += 1
            if self.current_index == self.window_size:
                self.window_full = True
        else:
            # 滑动窗口：移除最旧的点，添加新点
            if self.use_onnx:
                self.buffer = np.roll(self.buffer, shift=-1, axis=1)
                self.buffer[0, -1, 0] = data_point
            else:
                self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
                self.buffer[0, -1, 0] = data_point

        # 更新计数器
        self.sample_counter += 1

        # 如果窗口已满，进行推理
        if self.window_full:
            freq_estimate = self._process_window()

            # 记录处理时间
            process_time = time.time() - start_time
            self.processing_times.append(process_time)

            return freq_estimate

        return None

    def _process_window(self):
        """处理当前窗口并返回频率估计"""
        current_time = time.time()

        # 执行推理
        if self.use_onnx:
            ort_inputs = {self.model.get_inputs()[0].name: self.buffer}
            ort_outs = self.model.run(None, ort_inputs)
            freq_norm = ort_outs[0][0][0]
        else:
            with torch.no_grad():
                output = self.model(self.buffer)
                freq_norm = output.item()

        # 限制在合理范围内
        freq = max(0, min(freq_norm * self.max_freq, self.max_freq))

        # 记录结果
        self.frequency_history.append(freq)
        self.timestamps.append(current_time)

        # 计算信号质量
        if not self.use_onnx:
            window_data = self.buffer.cpu().numpy().flatten()
            quality = signal_quality_check(window_data, self.sampling_rate)
            self.signal_quality.append(quality)

        return freq

    def get_current_frequency(self, smoothing_window=10):
        """
        获取当前估计的频率

        参数:
            smoothing_window (int): 平滑窗口大小

        返回:
            float: 平滑后的频率估计
        """
        if len(self.frequency_history) == 0:
            return 0.0

        # 使用移动平均
        recent = list(self.frequency_history)[-smoothing_window:]
        return sum(recent) / len(recent)

    def get_processing_stats(self):
        """获取处理统计信息"""
        if len(self.processing_times) == 0:
            return {
                'avg_time': 0,
                'max_time': 0,
                'samples_per_second': 0
            }

        avg_time = sum(self.processing_times) / len(self.processing_times)
        max_time = max(self.processing_times)

        elapsed = time.time() - self.start_time
        samples_per_second = self.sample_counter / elapsed if elapsed > 0 else 0

        return {
            'avg_time': avg_time,
            'max_time': max_time,
            'samples_per_second': samples_per_second
        }

    def reset(self):
        """重置处理器状态"""
        self._init_processing_state()
        self.processing_times.clear()
        self.sample_counter = 0
        self.start_time = time.time()


class AsyncPulseProcessor:
    """异步脉冲处理器（使用独立线程处理）"""

    def __init__(self, model_path=None, config=None, device='auto'):
        self.config = config or Config()
        self.device = device

        # 创建主处理器
        self.processor = RealTimePulseProcessor(model_path, config, device)

        # 异步处理状态
        self.input_queue = deque()
        self.output_queue = deque()
        self.lock = threading.Lock()
        self.running = False
        self.processing_thread = None

    def start(self):
        """启动异步处理线程"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop(self):
        """停止异步处理线程"""
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join()

    def add_data_point(self, data_point):
        """添加数据点到处理队列"""
        with self.lock:
            self.input_queue.append(data_point)

    def get_results(self):
        """获取处理结果"""
        results = []
        with self.lock:
            while len(self.output_queue) > 0:
                results.append(self.output_queue.popleft())
        return results

    def _processing_loop(self):
        """处理循环（在独立线程中运行）"""
        while self.running:
            # 获取输入数据
            input_data = None
            with self.lock:
                if len(self.input_queue) > 0:
                    input_data = self.input_queue.popleft()

            if input_data is not None:
                # 处理数据
                result = self.processor.add_data_point(input_data)

                # 存储结果
                if result is not None:
                    with self.lock:
                        self.output_queue.append({
                            'timestamp': time.time(),
                            'frequency': result,
                            'quality': self.processor.signal_quality[-1] if self.processor.signal_quality else 0
                        })
            else:
                time.sleep(0.001)  # 避免忙等待

    def get_current_frequency(self, smoothing_window=10):
        """获取当前频率估计"""
        return self.processor.get_current_frequency(smoothing_window)

    def reset(self):
        """重置处理器状态"""
        with self.lock:
            self.input_queue.clear()
            self.output_queue.clear()
        self.processor.reset()


class MultiChannelProcessor:
    """多通道脉冲处理器"""

    def __init__(self, num_channels, model_path=None, config=None, device='auto'):
        self.config = config or Config()
        self.num_channels = num_channels
        self.device = device

        # 为每个通道创建处理器
        self.channels = [
            RealTimePulseProcessor(model_path, config, device)
            for _ in range(num_channels)
        ]

        # 交叉通道分析状态
        self.cross_channel_delay = [0] * num_channels
        self.last_update_time = time.time()

    def add_data_point(self, channel_idx, data_point):
        """
        添加数据点到指定通道

        参数:
            channel_idx (int): 通道索引 (0到num_channels-1)
            data_point (float): 数据点值

        返回:
            float or None: 频率估计（如果窗口已满）
        """
        if channel_idx < 0 or channel_idx >= self.num_channels:
            raise ValueError(f"Invalid channel index: {channel_idx}")

        return self.channels[channel_idx].add_data_point(data_point)

    def get_channel_frequency(self, channel_idx, smoothing_window=10):
        """
        获取指定通道的频率估计

        参数:
            channel_idx (int): 通道索引
            smoothing_window (int): 平滑窗口大小

        返回:
            float: 频率估计
        """
        return self.channels[channel_idx].get_current_frequency(smoothing_window)

    def get_consensus_frequency(self, min_channels=2):
        """
        获取多通道共识频率（多数通道同意的频率）

        参数:
            min_channels (int): 需要达成共识的最小通道数

        返回:
            float or None: 共识频率（如果没有足够通道达成共识则返回None）
        """
        # 收集所有通道的当前频率估计
        freqs = []
        for i in range(self.num_channels):
            freq = self.channels[i].get_current_frequency()
            if freq > 0:  # 忽略无效频率
                freqs.append(freq)

        if len(freqs) < min_channels:
            return None

        # 计算频率直方图
        hist, bin_edges = np.histogram(freqs, bins=10)
        max_bin = np.argmax(hist)

        # 检查是否达到最小通道数共识
        if hist[max_bin] >= min_channels:
            # 返回该区间的平均频率
            return (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2

        return None

    def analyze_cross_channel_delay(self):
        """
        分析通道间的延迟

        返回:
            list: 每个通道相对于第一个通道的延迟（秒）
        """
        # 需要至少两个通道
        if self.num_channels < 2:
            return [0] * self.num_channels

        # 获取所有通道的信号质量
        qualities = []
        for chan in self.channels:
            if chan.signal_quality:
                qualities.append(chan.signal_quality[-1])
            else:
                qualities.append(0)

        # 找到主通道（信号质量最好的）
        main_channel = np.argmax(qualities)

        # 计算其他通道相对于主通道的延迟
        delays = [0] * self.num_channels
        main_freq = self.channels[main_channel].get_current_frequency()

        if main_freq == 0:
            return delays

        # 假设信号是周期性的，计算相位差
        for i in range(self.num_channels):
            if i == main_channel:
                continue

            chan_freq = self.channels[i].get_current_frequency()
            if chan_freq == 0:
                continue

            # 简单的相位差计算（实际应用中可能需要更复杂的算法）
            phase_diff = abs(main_freq - chan_freq)
            delays[i] = phase_diff / (2 * np.pi * main_freq)

        return delays

    def reset_channel(self, channel_idx):
        """重置指定通道"""
        self.channels[channel_idx].reset()

    def reset_all(self):
        """重置所有通道"""
        for chan in self.channels:
            chan.reset()