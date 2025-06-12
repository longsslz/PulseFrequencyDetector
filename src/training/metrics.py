# src/training/metrics.py
import numpy as np
import torch

def mean_absolute_error(y_true, y_pred):
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """计算均方误差 (MSE)"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    """计算决定系数 (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def explained_variance_score(y_true, y_pred):
    """计算解释方差分数"""
    diff_true = y_true - np.mean(y_true)
    diff_pred = y_pred - np.mean(y_pred)
    numerator = np.sum(diff_true * diff_pred)
    denominator = np.sqrt(np.sum(diff_true ** 2) * np.sum(diff_pred ** 2))
    return (numerator / denominator) ** 2 if denominator != 0 else 0.0

def calculate_metrics(y_true, y_pred):
    """计算多个回归指标"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'evs': explained_variance_score(y_true, y_pred)
    }
    return metrics

def torch_mae(y_true, y_pred):
    """PyTorch实现的MAE"""
    return torch.mean(torch.abs(y_true - y_pred))

def torch_mse(y_true, y_pred):
    """PyTorch实现的MSE"""
    return torch.mean((y_true - y_pred) ** 2)