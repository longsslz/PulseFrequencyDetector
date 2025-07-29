# src/config.py

class Config:
    # 数据生成配置
    DATA = {
        'num_samples': 10000,
        'seq_length': 256,
        'sampling_rate': 100,
        'max_freq': 50,
        'noise_level': 0.2,
        'missing_prob': 0.05
    }

    # 模型配置
    MODEL = {
        'input_size': 1,
        'conv_channels': [8, 16],
        'gru_hidden_size': 32,
        'fc_layers': [32, 16],
        'output_size': 1,
        # 新增分类任务配置
        'num_classes': 7,  # 类别数量（根据实际频率范围设置）
        'classifier_thresholds': [5.25, 5.75, 6.25, 6.75, 7.25, 7.75]  # 频率分类阈值（Hz）
    }

    # 训练配置
    TRAINING = {
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        'train_split': 0.8,
        'early_stopping_patience': 5,
        'task_type': 'regression',  # 默认回归任务，可设置为' classification '
        'gradient_accumulation_steps': 4,  # 梯度累积步数
        'use_amp': True,  # 启用自动混合精度
    }

    # 实时处理配置
    REALTIME = {
        'window_size': 256,
        'sampling_rate': 100,
        'max_freq': 50,
        'smoothing_window': 10,
        'quantized_model': True
    }

    # 路径配置
    PATHS = {
        'model_dir': '../models/',
        'best_model': 'best_pulse_detector.pth',
        'quantized_model': 'pulse_detector_quantized.pth'
    }