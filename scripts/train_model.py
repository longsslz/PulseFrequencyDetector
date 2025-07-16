#!/usr/bin/env python3
"""
脉冲频率检测模型训练脚本
用法: python train_model.py [--model_type MODEL_TYPE] [--epochs EPOCHS]
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import torch
from torch.utils.data import DataLoader
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.config import Config
    from src.data.dataset import PulseDataset
    from src.model import create_model
    from src.training.trainer import Trainer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都已实现")
    sys.exit(1)


def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='训练脉冲频率检测模型')
    parser.add_argument('--model_type', type=str, default='deep',
                        choices=['standard', 'lite', 'deep', 'tcn'],
                        help='模型类型 (standard, lite, deep, tcn)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'auto'],
                        help='计算设备 (cpu, cuda, auto)')
    parser.add_argument('--data_path', type=str, default='../src/data/2025-7-15',
                        help='数据路径')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器工作进程数')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每N个epoch保存一次模型')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg):
    """获取计算设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用 GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("使用 CPU")
    else:
        device = torch.device(device_arg)
        print(f"使用指定设备: {device}")

    return device


def create_data_loaders(config, args, logger):
    """创建数据加载器"""
    logger.info("准备数据集...")
    # 确保使用绝对路径
    data_path = os.path.abspath(args.data_path)
    try:
        dataset = PulseDataset(config,None,False, data_path, logger)
        logger.info(f"数据集大小: {len(dataset)}")

        # 数据集分割
        val_size = int(args.val_split * len(dataset))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        logger.info(f"训练集大小: {train_size}, 验证集大小: {val_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.TRAINING['batch_size'],
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.TRAINING['batch_size'],
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            pin_memory=True
        )

        return train_loader, val_loader, dataset

    except Exception as e:
        logger.error(f"创建数据加载器时出错: {e}")
        raise


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置日志
    logger = setup_logging()
    logger.info("开始训练脉冲频率检测模型")
    logger.info(f"训练参数: {vars(args)}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 加载配置
        config = Config()

        # 使用命令行参数覆盖配置
        config.TRAINING['epochs'] = args.epochs
        config.TRAINING['batch_size'] = args.batch_size
        config.TRAINING['learning_rate'] = args.learning_rate

        # 创建数据加载器
        train_loader, val_loader, dataset = create_data_loaders(config, args, logger)

        # 获取设备
        device = get_device(args.device)

        # 初始化模型
        logger.info(f"初始化 {args.model_type} 模型...")
        model = create_model(args.model_type, config).to(device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数数量: {total_params:,}")

        # 训练配置
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAINING['learning_rate'],
            weight_decay=1e-5
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        criterion = torch.nn.MSELoss()

        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            scheduler=scheduler,
        )

        # # 从检查点恢复（如果指定）
        # if args.resume:
        #     logger.info(f"从检查点恢复训练: {args.resume}")
        #     trainer.load_checkpoint(args.resume)

        # 开始训练
        logger.info("开始训练...")
        best_val_loss = trainer.train()

        # 保存最终模型
        logger.info("保存最终模型...")
        trainer.save_best_model()

        # 量化模型并保存
        logger.info("量化模型...")
        try:
            trainer.quantize_and_save()
        except Exception as e:
            logger.warning(f"模型量化失败: {e}")

        # 评估最终模型
        logger.info("评估最终模型...")
        trainer.evaluate_final_model()

        logger.info("训练完成!")
        #logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        #logger.info(f"最终评估指标: {final_metrics}")

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()