# tests/test_realtime.py
import unittest
import numpy as np
import torch
from src.processing.realtime import RealTimePulseProcessor
from src.config import Config


class TestRealTimeProcessor(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.processor = RealTimePulseProcessor(device='cpu')

    def test_initial_state(self):
        self.assertEqual(self.processor.current_index, 0)
        self.assertFalse(self.processor.window_full)
        self.assertEqual(len(self.processor.frequency_history), 0)

    def test_buffer_filling(self):
        # 添加数据直到窗口填满
        for i in range(self.config.REALTIME['window_size'] - 1):
            self.processor.add_data_point(0.5)
            self.assertEqual(self.processor.current_index, i + 1)
            self.assertFalse(self.processor.window_full)

        # 添加最后一个点
        result = self.processor.add_data_point(0.5)
        self.assertEqual(self.processor.current_index, self.config.REALTIME['window_size'])
        self.assertTrue(self.processor.window_full)
        self.assertIsNotNone(result)

    def test_sliding_window(self):
        # 填满窗口
        for _ in range(self.config.REALTIME['window_size'] + 10):
            self.processor.add_data_point(0.5)

        # 检查缓冲区大小不变
        self.assertEqual(self.processor.buffer.shape[1], self.config.REALTIME['window_size'])

    def test_frequency_prediction(self):
        # 创建脉冲信号 (10Hz)
        sampling_rate = self.config.REALTIME['sampling_rate']
        period = sampling_rate // 10  # 10Hz

        # 填满缓冲区
        for i in range(self.config.REALTIME['window_size'] + 20):
            # 生成10Hz脉冲信号
            value = 1.0 if i % period == 0 else 0.0
            self.processor.add_data_point(value)

        # 检查频率估计
        current_freq = self.processor.get_current_frequency()
        self.assertGreaterEqual(current_freq, 9.0)
        self.assertLessEqual(current_freq, 11.0)

    def test_reset(self):
        # 添加一些数据
        for i in range(50):
            self.processor.add_data_point(0.5)

        # 重置处理器
        self.processor.reset()

        # 验证状态重置
        self.assertEqual(self.processor.current_index, 0)
        self.assertFalse(self.processor.window_full)
        self.assertEqual(len(self.processor.frequency_history), 0)
        self.assertTrue(torch.all(self.processor.buffer == 0))


if __name__ == '__main__':
    unittest.main()