# src/training/utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.quantization import quantize_dynamic
import seaborn as sns

# from src.training import r_squared, mean_absolute_error
from .metrics import (
    mean_absolute_error,
    r_squared
)

def save_model(model, path, state_dict=None):
    """保存模型到文件"""
    if state_dict is None:
        state_dict = model.state_dict()

    torch.save(state_dict, path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu'):
    """从文件加载模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def quantize_model(model):
    """量化模型以减少大小和提高推理速度"""
    # 动态量化
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.GRU, torch.nn.Conv1d},
        dtype=torch.qint8
    )
    print("Model quantized successfully")
    return quantized_model




def plot_training_history(train_loss, val_loss, train_metric, val_metric,
                          metric_name='MAE (Hz)', save_path=None):
    """
    绘制训练历史图表

    参数:
        train_loss (list): 训练损失历史
        val_loss (list): 验证损失历史
        train_metric (list): 训练指标历史
        val_metric (list): 验证指标历史
        metric_name (str): 指标名称（默认为'MAE (Hz)'）
        save_path (str): 图像保存路径（可选）
    """

    plt.figure(figsize=(12, 10))

    # 损失历史
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)

    # 指标历史
    plt.subplot(2, 1, 2)
    plt.plot(train_metric, label=f'Training {metric_name}')
    plt.plot(val_metric, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name} History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def plot_predictions_vs_actual(y_true, y_pred, save_path=None):
    """绘制预测值 vs 实际值"""
    # 计算线性回归线
    coefficients = np.polyfit(y_true, y_pred, 1)
    regression_line = np.poly1d(coefficients)

    # 计算完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    perfect_line = np.linspace(min_val, max_val, 100)

    plt.figure(figsize=(10, 8))

    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')

    # 回归线
    plt.plot(perfect_line, perfect_line, 'r--', label='Perfect Prediction')
    plt.plot(perfect_line, regression_line(perfect_line), 'g-', label='Regression Line')

    plt.xlabel('Actual Frequency (Hz)')
    plt.ylabel('Predicted Frequency (Hz)')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.grid(True)

    # 添加R²值
    r2 = r_squared(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加MAE值
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(0.05, 0.90, f'MAE = {mae:.4f} Hz', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 设置相同的坐标轴范围
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path)
        print(f"Predictions vs actual plot saved to {save_path}")
    else:
        plt.show()


def create_learning_rate_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler_type = config.TRAINING.get('scheduler', 'plateau')

    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.TRAINING.get('scheduler_patience', 3),
            verbose=True
        )

    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.TRAINING.get('step_size', 10),
            gamma=config.TRAINING.get('gamma', 0.1)
        )

    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.TRAINING.get('t_max', 10),
            eta_min=config.TRAINING.get('eta_min', 1e-6)
        )

    else:
        print(f"Unknown scheduler type: {scheduler_type}. Using no scheduler.")
        return None


def plot_confusion_matrix(targets, predictions, class_names, save_path=None):
    """
    绘制混淆矩阵（分类任务）

    参数:
        targets (array): 实际类别数组
        predictions (array): 预测类别数组
        class_names (list): 类别名称列表
        save_path (str): 图像保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()

def calculate_model_size(model):
    """计算模型大小（参数数量和文件大小）"""
    param_count = sum(p.numel() for p in model.parameters())

    # 保存临时模型以计算文件大小
    temp_path = "temp_model.pth"
    torch.save(model.state_dict(), temp_path)
    file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
    os.remove(temp_path)

    return {
        'parameters': param_count,
        'size_mb': file_size
    }


def print_model_summary(model):
    """打印模型摘要信息"""
    print("Model Summary:")
    print("=" * 50)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name: <40} | {param.size()} | {param_count} parameters")

    print("=" * 50)
    print(f"Total trainable parameters: {total_params}")
    print(f"Model size: {calculate_model_size(model)['size_mb']:.2f} MB")
    print("=" * 50)