# src/training/trainer.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.training.metrics import mean_absolute_error, r_squared
from src.training.utils import save_model, quantize_model, plot_training_history, plot_predictions_vs_actual


class Trainer:
    """神经网络训练器类，封装训练循环和验证逻辑"""

    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 config, device='cuda', scheduler=None):
        """
        初始化训练器

        参数:
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            optimizer (optim.Optimizer): 优化器
            criterion (nn.Module): 损失函数
            config (Config): 配置对象
            device (str): 训练设备 ('cuda' 或 'cpu')
            scheduler (optim.lr_scheduler): 学习率调度器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = torch.device(device)
        self.scheduler = scheduler

        # 训练状态
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_mae_history = []
        self.val_mae_history = []
        self.epoch_times = []
        self.start_time = None

        # 确保模型在正确设备上
        self.model.to(self.device)

        # 创建模型目录
        os.makedirs(self.config.PATHS['model_dir'], exist_ok=True)

    def train_epoch(self):
        """训练单个epoch"""
        self.model.train()
        epoch_loss = 0.0
        all_targets = []
        all_predictions = []

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 移动数据到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()

            # 计算损失
            loss = self.criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 记录损失
            epoch_loss += loss.item() * inputs.size(0)

            # 收集预测和目标用于指标计算
            all_predictions.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            # 打印进度
            if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}")

        # 计算平均损失
        epoch_loss /= len(self.train_loader.dataset)

        # 计算指标
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        epoch_mae = mean_absolute_error(all_targets, all_predictions)

        return epoch_loss, epoch_mae

    def validate(self):
        """在验证集上评估模型"""
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # 移动数据到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(inputs).squeeze()

                # 计算损失
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # 收集预测和目标
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 计算平均损失
        val_loss /= len(self.val_loader.dataset)

        # 计算指标
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        val_mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r_squared(all_targets, all_predictions)

        return val_loss, val_mae, r2, all_targets, all_predictions

    def train(self, epochs=None):
        """执行完整的训练过程"""
        epochs = epochs or self.config.TRAINING['epochs']
        self.start_time = time.time()

        print(f"Starting training for {epochs} epochs on {self.device}...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 训练一个epoch
            train_loss, train_mae = self.train_epoch()

            # 验证
            val_loss, val_mae, r2, val_targets, val_preds = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录训练历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_mae_history.append(train_mae)
            self.val_mae_history.append(val_mae)
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # 打印epoch结果
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.6f}, MAE: {train_mae:.4f}")
            print(f"  Val Loss: {val_loss:.6f}, MAE: {val_mae:.4f}, R²: {r2:.4f} \n")

            # 检查是否是最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
                print(f"  New best model! Val loss: {val_loss:.6f}")

        # 训练完成
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")

        # 保存训练历史图
        plot_training_history(
            self.train_loss_history,
            self.val_loss_history,
            self.train_mae_history,
            self.val_mae_history,
            save_path=os.path.join(self.config.PATHS['model_dir'], 'training_history.png')
        )

        return self.best_model_state

    def save_best_model(self):
        """保存最佳模型"""
        if not hasattr(self, 'best_model_state'):
            raise RuntimeError("No best model state found. Did you run training?")

        model_path = os.path.join(self.config.PATHS['model_dir'], self.config.PATHS['best_model'])
        save_model(self.model, model_path, self.best_model_state)
        print(f"Saved best model to {model_path}")

    def quantize_and_save(self):
        """量化模型并保存"""
        if not hasattr(self, 'best_model_state'):
            raise RuntimeError("No best model state found. Did you run training?")

        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)

        # 量化模型
        quantized_model = quantize_model(self.model)

        # 保存量化模型
        model_path = os.path.join(self.config.PATHS['model_dir'], self.config.PATHS['quantized_model'])
        torch.save(quantized_model.state_dict(), model_path)
        print(f"Saved quantized model to {model_path}")

        return quantized_model

    def evaluate_final_model(self):
        """评估最终模型在验证集上的性能"""
        # 使用最佳模型状态
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()

        # 在验证集上评估
        val_loss, val_mae, r2, val_targets, val_preds = self.validate()

        print("\nFinal Model Evaluation:")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation MAE: {val_mae:.4f} Hz")
        print(f"R² Score: {r2:.4f}")



        # 绘制预测 vs 实际值
        plot_predictions_vs_actual(
            val_targets,
            val_preds,
            save_path=os.path.join(self.config.PATHS['model_dir'], 'predictions_vs_actual.png')
        )

        return val_loss, val_mae, r2

    def plot_loss_history(self):
        """绘制损失历史"""
        plt.figure(figsize=(12, 6))

        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # 绘制MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.train_mae_history, label='Train MAE')
        plt.plot(self.val_mae_history, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE (Hz)')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.PATHS['model_dir'], 'loss_history.png'))
        plt.show()