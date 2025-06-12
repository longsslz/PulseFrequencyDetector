#!/usr/bin/env python3
"""
脉冲频率离线推理脚本
用法: python inference.py --input INPUT_FILE [--model MODEL_PATH]
"""

import argparse
import numpy as np
import torch
from src.config import Config
from src.data.dataset import PulseDataset
from src.model import create_model
from src.processing.utils import sliding_window


def parse_args():
    parser = argparse.ArgumentParser(description='脉冲频率离线推理')
    parser.add_argument('--input', type=str, required=True,
                        help='输入信号文件(.npy或.csv)')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径(默认使用配置中的最佳模型)')
    parser.add_argument('--window_size', type=int, default=256,
                        help='处理窗口大小')
    parser.add_argument('--step_size', type=int, default=64,
                        help='滑动窗口步长')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu, cuda)')
    return parser.parse_args()


def load_signal(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError("不支持的文件格式，请使用.npy或.csv")


def main():
    args = parse_args()
    config = Config()

    # 加载模型
    device = torch.device(args.device)
    model = create_model('standard', config).to(device)

    model_path = args.model or (config.PATHS['model_dir'] + config.PATHS['best_model'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载信号
    signal = load_signal(args.input)
    if len(signal.shape) > 1:
        signal = signal.squeeze()

    # 分割窗口
    windows = sliding_window(signal, args.window_size, args.step_size)

    # 推理
    frequencies = []
    with torch.no_grad():
        for window in windows:
            # 转换为tensor并添加批次和通道维度
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            # 预测
            pred = model(x).item()
            freq = max(0, min(pred * config.DATA['max_freq'], config.DATA['max_freq']))
            frequencies.append(freq)

    # 输出结果
    print("\n推理结果:")
    print(f"信号长度: {len(signal)} 样本")
    print(f"处理窗口数: {len(windows)}")
    print(f"平均频率: {np.mean(frequencies):.2f} Hz")
    print(f"最小频率: {np.min(frequencies):.2f} Hz")
    print(f"最大频率: {np.max(frequencies):.2f} Hz")

    # 保存结果
    output_file = args.input.rsplit('.', 1)[0] + '_result.npy'
    np.save(output_file, np.array(frequencies))
    print(f"\n频率结果已保存到: {output_file}")


if __name__ == '__main__':
    main()