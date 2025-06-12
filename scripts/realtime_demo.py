#!/usr/bin/env python3
"""
实时脉冲频率检测演示脚本
用法: python realtime_demo.py [--model MODEL_PATH] [--duration DURATION]
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from src.config import Config
from src.processing import RealTimePulseProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='实时脉冲频率检测演示')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径(默认使用配置中的量化模型)')
    parser.add_argument('--duration', type=float, default=30,
                        help='演示时长(秒)')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='采样率(Hz)')
    parser.add_argument('--base_freq', type=float, default=5,
                        help='基础频率(Hz)')
    parser.add_argument('--variation', type=float, default=3,
                        help='频率变化幅度(Hz)')
    return parser.parse_args()


def generate_test_signal(duration, sampling_rate, base_freq, variation):
    """生成测试信号"""
    t = np.linspace(0, duration, int(duration * sampling_rate))
    freq_variation = variation * np.sin(2 * np.pi * 0.1 * t) + base_freq
    signal = square(2 * np.pi * freq_variation * t, duty=0.2)
    return t, signal, freq_variation


def main():
    args = parse_args()
    config = Config()

    # 初始化处理器
    processor = RealTimePulseProcessor(
        model_path=args.model,
        config=config,
        device='auto'
    )

    # 生成测试信号
    t, signal, true_freq = generate_test_signal(
        args.duration, args.sampling_rate,
        args.base_freq, args.variation
    )

    # 添加噪声
    noise = np.random.normal(0, 0.3, len(signal))
    noisy_signal = np.clip(signal + noise, 0, 1)

    # 准备绘图
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('实时脉冲频率检测演示')

    # 实时处理循环
    estimated_freqs = []
    timestamps = []

    start_time = time.time()
    for i, sample in enumerate(noisy_signal):
        # 模拟实时采样延迟
        elapsed = time.time() - start_time
        expected_time = i / args.sampling_rate
        sleep_time = expected_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # 处理数据点
        freq_est = processor.add_data_point(sample)

        if freq_est is not None:
            estimated_freqs.append(processor.get_current_frequency())
            timestamps.append(time.time() - start_time)

            # 更新绘图
            if i % args.sampling_rate == 0:  # 每秒更新一次
                ax1.clear()
                ax2.clear()

                # 绘制信号
                ax1.plot(t[:i + 1], noisy_signal[:i + 1], label='信号')
                ax1.set_xlabel('时间 (s)')
                ax1.set_ylabel('幅值')
                ax1.set_title('输入信号')
                ax1.grid(True)

                # 绘制频率
                ax2.plot(t[:i + 1], true_freq[:i + 1], 'b-', label='实际频率')
                if timestamps:
                    ax2.plot(timestamps, estimated_freqs, 'r-', label='估计频率')
                ax2.set_xlabel('时间 (s)')
                ax2.set_ylabel('频率 (Hz)')
                ax2.set_title('频率估计')
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                plt.pause(0.01)

    # 保持绘图显示
    plt.ioff()
    plt.show()

    # 打印统计信息
    stats = processor.get_processing_stats()
    print("\n处理统计:")
    print(f"平均处理时间: {stats['avg_time'] * 1000:.2f} ms/样本")
    print(f"最大处理时间: {stats['max_time'] * 1000:.2f} ms")
    print(f"吞吐量: {stats['samples_per_second']:.1f} 样本/秒")


if __name__ == '__main__':
    main()