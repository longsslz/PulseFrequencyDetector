# src/model/architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class BasePulseDetector(nn.Module):
    """脉冲频率检测器的基类"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.MODEL['input_size']
        self.output_size = config.MODEL['output_size']

    def forward(self, x):
        raise NotImplementedError

    def get_optimizer(self, lr=None):
        """获取默认优化器"""
        lr = lr or self.config.TRAINING['learning_rate']
        return torch.optim.Adam(self.parameters(), lr=lr)

    def get_criterion(self):
        """获取默认损失函数"""
        return nn.MSELoss()


class PulseFrequencyDetector(BasePulseDetector):
    """标准脉冲频率检测器模型 (CNN + GRU)"""

    def __init__(self, config):
        super().__init__(config)

        # 1D卷积层
        conv_channels = config.MODEL['conv_channels']
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_size

        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels

        # GRU层
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=config.MODEL['gru_hidden_size'],
            batch_first=True
        )

        # 全连接层
        fc_layers = []
        in_features = config.MODEL['gru_hidden_size']

        for out_features in config.MODEL['fc_layers']:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, self.output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch, input_size, seq_len)

        # 通过卷积层
        for conv in self.conv_layers:
            x = conv(x)

        # 转换为GRU需要的形状 (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # 通过GRU层
        _, h_n = self.gru(x)  # h_n形状: (1, batch, hidden_size)

        # 通过全连接层
        output = self.fc(h_n.squeeze(0))

        return output


class LitePulseDetector(BasePulseDetector):
    """轻量级脉冲检测器 (更小的参数量)"""

    def __init__(self, config):
        super().__init__(config)

        # 1D卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_size, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # GRU层
        self.gru = nn.GRU(
            input_size=16,
            hidden_size=16,
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_size)
        )

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch, input_size, seq_len)

        # 通过卷积层
        x = self.conv1(x)
        x = self.conv2(x)

        # 转换为GRU需要的形状 (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # 通过GRU层
        _, h_n = self.gru(x)  # h_n形状: (1, batch, hidden_size)

        # 通过全连接层
        output = self.fc(h_n.squeeze(0))

        return output


class DeepPulseDetector(BasePulseDetector):
    """更深的脉冲检测器模型 (更高精度)"""

    def __init__(self, config):
        super().__init__(config)

        # 深度卷积网络
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_size
        conv_channels = [32, 32, 64, 128]

        for out_channels in conv_channels:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(0.2)
                )
            )
            in_channels = out_channels

        # 双向GRU层
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=config.MODEL['gru_hidden_size'],
            batch_first=True,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(config.MODEL['gru_hidden_size'] * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 全连接层
        fc_layers = []
        in_features = config.MODEL['gru_hidden_size'] * 2  # 双向

        for out_features in [256, 128, 64]:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.3))
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, self.output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch, input_size, seq_len)

        # 通过卷积层
        for conv in self.conv_blocks:
            x = conv(x)

        # 转换为GRU需要的形状 (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # 通过双向GRU层
        gru_out, _ = self.gru(x)  # gru_out形状: (batch, seq_len, hidden_size*2)

        # 注意力机制
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_size*2)

        # 通过全连接层
        output = self.fc(context_vector)

        return output


class TemporalConvNetDetector(BasePulseDetector):
    """基于Temporal Convolutional Network的脉冲检测器"""

    def __init__(self, config):
        super().__init__(config)

        # 时间卷积网络
        self.tcn = nn.Sequential(
            nn.Conv1d(self.input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch, input_size, seq_len)

        # 通过TCN
        x = self.tcn(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 通过全连接层
        output = self.fc(x)

        return output


def create_model(model_name, config):
    """模型工厂函数"""
    models = {
        'standard': PulseFrequencyDetector,
        'lite': LitePulseDetector,
        'deep': DeepPulseDetector,
        'tcn': TemporalConvNetDetector
    }

    if model_name not in models: 
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(models.keys())}")

    return models[model_name](config)