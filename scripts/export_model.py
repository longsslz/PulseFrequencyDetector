#!/usr/bin/env python3
"""
模型导出脚本 (ONNX/TorchScript)
用法: python export_model.py [--format FORMAT] [--model MODEL_TYPE]
"""

import argparse
import torch
from src.config import Config
from src.model import create_model
from src.model.model_utils import save_model_onnx


def parse_args():
    parser = argparse.ArgumentParser(description='模型导出脚本')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript'],
                        help='导出格式 (onnx, torchscript)')
    parser.add_argument('--model', type=str, default='standard',
                        choices=['standard', 'lite', 'deep', 'tcn'],
                        help='模型类型 (standard, lite, deep, tcn)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu, cuda)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # 初始化模型
    device = torch.device(args.device)
    model = create_model(args.model, config).to(device)

    # 加载预训练权重
    model_path = config.PATHS['model_dir'] + config.PATHS['best_model']
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 定义输入尺寸
    input_size = (1, config.REALTIME['window_size'], config.MODEL['input_size'])

    # 导出模型
    output_path = args.output or f"{config.PATHS['model_dir']}{args.model}_pulse_detector.{args.format}"

    if args.format == 'onnx':
        print(f"导出ONNX模型到 {output_path}...")
        save_model_onnx(model, output_path, input_size, device)
    elif args.format == 'torchscript':
        print(f"导出TorchScript模型到 {output_path}...")
        dummy_input = torch.randn(input_size, device=device)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)

    print(f"成功导出 {args.model} 模型为 {args.format} 格式")


if __name__ == '__main__':
    main()