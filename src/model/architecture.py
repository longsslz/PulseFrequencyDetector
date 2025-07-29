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


class PulseFrequencyClassifier(BasePulseDetector):
    """脉冲频率分类器模型 (CNN + BiGRU + 注意力机制)"""

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.MODEL['num_classes']  # 新增分类类别数配置

        # 卷积特征提取层
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_size
        conv_channels = [32, 64, 128]

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
            bidirectional=True,
            num_layers=2
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(config.MODEL['gru_hidden_size'] * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.MODEL['gru_hidden_size'] * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)

        # 卷积特征提取
        for conv in self.conv_blocks:
            x = conv(x)

        # 准备序列特征
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)

        # 双向GRU处理
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)

        # 注意力机制
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden_size*2)

        # 分类预测
        logits = self.classifier(context)
        return logits


class EnhancedPulseClassifier(BasePulseDetector):
    """增强版脉冲频率分类器模型 (残差卷积 + BiGRU + 多头注意力)"""

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.MODEL['num_classes']
        self.embed_size = config.MODEL.get('embed_size', 64)

        # 嵌入层处理不同特征的尺度差异
        self.embedding = nn.Linear(self.input_size, self.embed_size)

        # 改进的卷积模块：残差块+空洞卷积
        self.conv_blocks = nn.ModuleList()
        conv_channels = [64, 128, 256]
        dilation_rates = [1, 2, 4]  # 空洞卷积扩大感受野

        in_channels = self.embed_size
        for i, (out_channels, dilation) in enumerate(zip(conv_channels, dilation_rates)):
            # 主干卷积路径
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.2)
            )

            # 残差连接捷径
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

            self.conv_blocks.append(nn.ModuleDict({
                'main': conv_block,
                'shortcut': self.shortcut
            }))
            in_channels = out_channels

        # 双向GRU层
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=config.MODEL['gru_hidden_size'],
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )
        gru_out_size = config.MODEL['gru_hidden_size'] * 2  # 双向

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=gru_out_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_size),  # 稳定训练
            nn.Linear(gru_out_size, 256),
            nn.SiLU(),  # 更平滑的激活函数
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)

        # 特征嵌入
        x = self.embedding(x)  # (batch, seq_len, embed_size)

        # 卷积特征提取
        x = x.permute(0, 2, 1)  # (batch, embed_size, seq_len)

        for block in self.conv_blocks:
            # 残差连接
            residual = block['shortcut'](x)

            # 主干路径
            main_out = block['main'](x)

            # 残差连接 (确保形状匹配)
            if residual.shape != main_out.shape:
                # 调整残差连接形状
                residual = F.interpolate(residual, size=main_out.shape[2:])

            # 合并输出
            x = main_out + residual

        x = x.permute(0, 2, 1)  # (batch, seq_len, features)

        # 双向GRU处理
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)

        # 多头注意力机制
        attn_out, _ = self.multihead_attn(
            gru_out, gru_out, gru_out  # self-attention
        )  # (batch, seq_len, hidden_size*2)

        # 全局平均池化
        context = torch.mean(attn_out, dim=1)  # (batch, hidden_size*2)

        # 分类预测
        logits = self.classifier(context)
        return logits


class LightPulseClassifier(BasePulseDetector):
    """轻量级脉冲频率分类器模型 (优化内存使用)"""

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.MODEL['num_classes']
        self.embed_size = config.MODEL.get('embed_size', 32)  # 减小嵌入维度

        # 特征嵌入层
        self.embedding = nn.Linear(self.input_size, self.embed_size)

        # 轻量卷积模块
        self.conv_blocks = nn.Sequential(
            # 第1卷积层
            nn.Conv1d(self.embed_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 第2卷积层
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # 第3卷积层 (深度可分离卷积)
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 轻量GRU层
        self.gru = nn.GRU(
            input_size=128,  # 减小输入维度
            hidden_size=64,  # 减小隐藏层大小
            batch_first=True,
            bidirectional=True
        )

        # 轻量注意力机制
        self.attention = nn.Sequential(
            nn.Linear(128, 32),  # 减小维度
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        # 轻量分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 输入形状: (batch, seq_len, input_size)

        # 特征嵌入
        x = self.embedding(x)  # (batch, seq_len, embed_size)

        # 卷积特征提取
        x = x.permute(0, 2, 1)  # (batch, embed_size, seq_len)
        x = self.conv_blocks(x)

        # 准备序列特征
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)

        # GRU处理
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size*2)

        # 注意力机制
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden_size*2)

        # 分类预测
        logits = self.classifier(context)
        return logits

# 更新模型工厂
def create_model(model_name, config):
    """模型工厂函数"""
    models = {
        'standard': PulseFrequencyDetector,
        'lite': LitePulseDetector,
        'deep': DeepPulseDetector,
        'tcn': TemporalConvNetDetector,
        'classifier': PulseFrequencyClassifier,
        'enhanced_classifier': EnhancedPulseClassifier,  # 添加新模型
        'light_classifier':LightPulseClassifier,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(models.keys())}")

    return models[model_name](config)