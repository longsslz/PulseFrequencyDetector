# src/training/trainer.py
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import torch.amp as amp
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.training.metrics import mean_absolute_error, r_squared
from src.training.utils import save_model, quantize_model, plot_training_history, plot_predictions_vs_actual, \
    plot_confusion_matrix
from sklearn.metrics import accuracy_score, mean_absolute_error

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

        # 新增分类任务支持
        self.task_type = config.TRAINING.get('task_type', 'regression')

        # 训练历史记录
        self.train_loss_history = []
        self.val_loss_history = []

        # 根据任务类型初始化不同的指标历史
        if self.task_type == 'regression':
            self.train_mae_history = []
            self.val_mae_history = []
            self.val_r2_history = []
        else:
            self.train_acc_history = []
            self.val_acc_history = []

        # 训练状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
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
            outputs = self.model(inputs)

            # 计算损失
            # 修改损失计算逻辑
            if self.task_type == 'classification':
                # 分类任务 - 输出形状应为 (batch, num_classes)
                loss = self.criterion(outputs, targets.long().squeeze())

                # 获取预测类别
                _, preds = torch.max(outputs, 1)
                all_predictions.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            else:
                # 回归任务
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)
                all_predictions.append(outputs.detach().cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 记录损失
            epoch_loss += loss.item() * inputs.size(0)

            # 打印进度
            if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}")

        # 计算平均损失
        epoch_loss /= len(self.train_loader.dataset)

        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        # 计算指标
        if self.task_type == 'classification':
            # 分类任务指标
            epoch_metric  = accuracy_score(all_targets, all_predictions)
        else:
            # 回归任务指标
            epoch_metric  = mean_absolute_error(all_targets, all_predictions)

        return epoch_loss, epoch_metric

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
                outputs = self.model(inputs)

                if self.task_type == 'classification':
                    # 分类任务
                    targets = targets.long().squeeze()
                    loss = self.criterion(outputs, targets)
                    # 获取预测类别
                    _, preds = torch.max(outputs, 1)
                else:
                    # 回归任务
                    outputs = outputs.squeeze()
                    loss = self.criterion(outputs, targets)
                    preds = outputs
                # 记录损失
                val_loss += loss.item() * inputs.size(0)

                # 收集预测和目标值
                all_predictions.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 计算平均损失
        val_loss /= len(self.val_loader.dataset)

        # 合并所有批次的结果
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        # 计算指标
        if self.task_type == 'classification':
            # 分类任务指标
            val_metric = accuracy_score(all_targets, all_predictions)
            return val_loss, val_metric
        else:
            # 回归任务指标
            val_mae = mean_absolute_error(all_targets, all_predictions)
            val_r2 = r_squared(all_targets, all_predictions)
            return val_loss, val_mae, val_r2



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
            train_loss, train_metric = self.train_epoch()

            # 验证
            if self.task_type == 'regression':
                val_loss, val_mae, val_r2 = self.validate()
            else:
                val_loss, val_metric = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录训练历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.task_type == 'regression':
                self.train_mae_history.append(train_metric)
                self.val_mae_history.append(val_mae)
                self.val_r2_history.append(val_r2)
            else:
                self.train_acc_history.append(train_metric)
                self.val_acc_history.append(val_metric)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # 打印epoch结果
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.6f}")

            if self.task_type == 'regression':
                print(f"  Train MAE: {train_metric:.4f}")
                print(f"  Val Loss: {val_loss:.6f}, MAE: {val_metric:.4f}, R²: {val_r2:.4f}")
            else:
                print(f"  Train Accuracy: {train_metric:.4f}")
                print(f"  Val Loss: {val_loss:.6f}, Accuracy: {val_metric:.4f}")

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
        if self.task_type == 'regression':
            train_metric = self.train_mae_history
            val_metric = self.val_mae_history
            metric_name = 'MAE (Hz)'
        else:
            train_metric = self.train_acc_history
            val_metric = self.val_acc_history
            metric_name = 'Accuracy'

        plot_training_history(
            self.train_loss_history,
            self.val_loss_history,
            train_metric,
            val_metric,
            metric_name=metric_name,
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
        if self.task_type == 'regression':
            # 回归任务
            val_loss, val_mae, val_r2 = self.validate()
            all_targets = []
            all_preds = []

            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs).squeeze()
                    all_targets.append(targets.cpu().numpy())
                    all_preds.append(outputs.cpu().numpy())

            # 合并所有预测和目标值
            val_targets = np.concatenate(all_targets)
            val_preds = np.concatenate(all_preds)

            print("\nFinal Model Evaluation (Regression):")
            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Validation MAE: {val_mae:.4f} Hz")
            print(f"R² Score: {val_r2:.4f}")

            # 绘制预测 vs 实际值
            plot_predictions_vs_actual(
                val_targets,
                val_preds,
                save_path=os.path.join(self.config.PATHS['model_dir'], 'predictions_vs_actual.png')
            )

            return val_loss, val_mae, val_r2
        else:
            # 分类任务
            val_loss, val_acc = self.validate()
            all_targets = []
            all_preds = []

            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_targets.append(targets.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())

            # 合并所有预测和目标值
            val_targets = np.concatenate(all_targets)
            val_preds = np.concatenate(all_preds)

            print("\nFinal Model Evaluation (Classification):")
            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Validation Accuracy: {val_acc:.4f}")

            # 绘制混淆矩阵
            plot_confusion_matrix(
                val_targets,
                val_preds,
                class_names=self.config.MODEL.get('class_names',
                                                  [str(i) for i in range(self.config.MODEL['num_classes'])]),
                save_path=os.path.join(self.config.PATHS['model_dir'], 'confusion_matrix.png')
            )

            return val_loss, val_acc

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


class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 config, device='cuda', scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # 任务类型
        self.task_type = config.TRAINING.get('task_type', 'regression')

        # 内存优化配置
        self.use_amp = config.TRAINING.get('use_amp', True)
        self.gradient_accumulation_steps = config.TRAINING.get('gradient_accumulation_steps', 4)
        self.batch_size = config.TRAINING.get('batch_size', 32)

        # AMP 初始化
        if self.use_amp:
            self.scaler = amp.GradScaler(enabled=True)
            print(f"启用 AMP (自动混合精度)")

        # 训练历史记录
        self.train_loss_history = []
        self.val_loss_history = []

        # 根据任务类型初始化不同的指标历史
        if self.task_type == 'regression':
            self.train_mae_history = []
            self.val_mae_history = []
            self.val_r2_history = []
        else:
            self.train_acc_history = []
            self.val_acc_history = []

        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        self.epoch_times = []

    def train_epoch(self):
        """训练单个epoch"""
        self.model.train()
        epoch_loss = 0.0
        all_targets = []
        all_predictions = []
        accumulation_counter = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 移动数据到设备
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 使用 AMP 上下文
            with amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                # 前向传播
                outputs = self.model(inputs)

                # 计算损失
                if self.task_type == 'classification':
                    # 分类任务
                    targets = targets.long().squeeze()
                    loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
                    _, preds = torch.max(outputs, 1)
                else:
                    # 回归任务
                    outputs = outputs.squeeze()
                    loss = self.criterion(outputs, targets) / self.gradient_accumulation_steps
                    preds = outputs

            # 缩放损失并反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 记录损失
            epoch_loss += loss.item() * inputs.size(0) * self.gradient_accumulation_steps

            # 收集预测和目标值
            all_targets.append(targets.cpu().detach().numpy())
            all_predictions.append(preds.cpu().detach().numpy())

            # 梯度累积
            accumulation_counter += 1
            if accumulation_counter % self.gradient_accumulation_steps == 0:
                # 更新参数
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # 清空梯度
                self.optimizer.zero_grad()
                accumulation_counter = 0

            # 及时释放内存
            del inputs, targets, outputs, loss
            torch.cuda.empty_cache()

        # 处理剩余梯度
        if accumulation_counter > 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        # 计算平均损失
        epoch_loss /= len(self.train_loader.dataset)

        # 计算指标
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        if self.task_type == 'classification':
            # 分类任务指标
            epoch_metric = accuracy_score(all_targets, all_predictions)
        else:
            # 回归任务指标
            epoch_metric = mean_absolute_error(all_targets, all_predictions)

        return epoch_loss, epoch_metric

    def validate(self):
        """在验证集上评估模型"""
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # 移动数据到设备
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 使用 AMP 上下文
                with amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    # 前向传播
                    outputs = self.model(inputs)

                    # 计算损失
                    if self.task_type == 'classification':
                        # 分类任务
                        targets = targets.long().squeeze()
                        loss = self.criterion(outputs, targets)
                        _, preds = torch.max(outputs, 1)
                    else:
                        # 回归任务
                        outputs = outputs.squeeze()
                        loss = self.criterion(outputs, targets)
                        preds = outputs

                # 记录损失
                val_loss += loss.item() * inputs.size(0)

                # 收集预测和目标值
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())

                # 及时释放内存
                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()

        # 计算平均损失
        val_loss /= len(self.val_loader.dataset)

        # 计算指标
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        if self.task_type == 'classification':
            # 分类任务指标
            val_metric = accuracy_score(all_targets, all_predictions)
            return val_loss, val_metric
        else:
            # 回归任务指标
            val_mae = mean_absolute_error(all_targets, all_predictions)
            val_r2 = r_squared(all_targets, all_predictions)
            return val_loss, val_mae, val_r2

    def train(self, epochs=None):
        """执行完整的训练过程"""
        epochs = epochs or self.config.TRAINING['epochs']
        self.start_time = time.time()

        print(f"Starting training for {epochs} epochs on {self.device}...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Using AMP: {self.use_amp}, Gradient Accumulation: {self.gradient_accumulation_steps}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 训练一个epoch
            train_loss, train_metric = self.train_epoch()

            # 验证
            if self.task_type == 'regression':
                val_loss, val_mae, val_r2 = self.validate()
            else:
                val_loss, val_metric = self.validate()

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录训练历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.task_type == 'regression':
                self.train_mae_history.append(train_metric)
                self.val_mae_history.append(val_mae)
                self.val_r2_history.append(val_r2)
            else:
                self.train_acc_history.append(train_metric)
                self.val_acc_history.append(val_metric)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # 打印epoch结果
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.6f}")

            if self.task_type == 'regression':
                print(f"  Train MAE: {train_metric:.4f}")
                print(f"  Val Loss: {val_loss:.6f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            else:
                print(f"  Train Accuracy: {train_metric:.4f}")
                print(f"  Val Loss: {val_loss:.6f}, Accuracy: {val_metric:.4f}")

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
        if self.task_type == 'regression':
            train_metric = self.train_mae_history
            val_metric = self.val_mae_history
            metric_name = 'MAE (Hz)'
        else:
            train_metric = self.train_acc_history
            val_metric = self.val_acc_history
            metric_name = 'Accuracy'

        plot_training_history(
            self.train_loss_history,
            self.val_loss_history,
            train_metric,
            val_metric,
            metric_name=metric_name,
            save_path=os.path.join(self.config.PATHS['model_dir'], 'training_history.png')
        )

        return self.best_model_state
