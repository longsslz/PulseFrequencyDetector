#!/usr/bin/env python3
"""
模型性能测试脚本
用法: python benchmark.py [--model MODEL_TYPE] [--batch_size BATCH_SIZE]
"""

import argparse
import time
import torch
import numpy as np
from src.config import Config
from src.model import create_model, model_size_in_mb, count_parameters
from src.training.utils import benchmark_model


def parse_args():
    parser = argparse.ArgumentParser(description='模型性能测试')
    parser.add_argument('--model', type=str, default='standard',
                        choices=['standard', 'lite', 'deep', 'tcn'],
                        help='模型类型 (standard, lite, deep, tcn)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=256,
                        help='序列长度')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='测试运行次数')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (cpu, cuda)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(
        args.device)

    # 初始化模型
    model = create_model(args.model, config).to(device)

    # 加载预训练权重
    model_path = config.PATHS['model_dir'] + config.PATHS['best_model']
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建测试输入
    input_tensor = torch.randn(
        args.batch_size, args.seq_length, config.MODEL['input_size']
    ).to(device)

    # 打印模型信息
    print("\n模型信息:")
    print(f"类型: {args.model}")
    print(f"参数量: {count_parameters(model):,}")
    print(f"模型大小: {model_size_in_mb(model):.2f} MB")
    print(f"输入形状: {tuple(input_tensor.shape)}")

    # 基准测试
    print("\n运行基准测试...")
    results = benchmark_model(model, input_tensor, args.num_runs)

    # 打印结果
    print("\n性能结果:")
    print(f"平均推理时间: {results['average_time_ms']:.2f} ms")
    print(f"标准差: {results['std_dev_ms']:.2f} ms")
    print(f"帧率(FPS): {results['fps']:.1f}")

    # 内存使用情况
    if device.type == 'cuda':
        print("\nGPU内存使用:")
        print(f"分配内存: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"缓存内存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


if __name__ == '__main__':
    main()