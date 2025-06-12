```markdown
pulse_frequency_detector/
│
├── models/                   # 存储训练好的模型
│   ├── best_pulse_detector.pth
│   └── pulse_detector_quantized.pth
│
├── src/                      # 源代码目录
│   ├── data/                 # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py        # 数据集类
│   │   ├── generator.py      # 合成数据生成
│   │   └── preprocessor.py   # 预处理工具
│   │
│   ├── model/                # 模型相关
│   │   ├── __init__.py
│   │   ├── architecture.py   # 模型架构定义
│   │   └── quantize.py       # 模型量化工具
│   │
│   ├── processing/           # 实时处理模块
│   │   ├── __init__.py
│   │   ├── realtime.py       # 实时处理器类
│   │   └── utils.py          # 处理工具函数
│   │
│   ├── training/             # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练器类
│   │   └── metrics.py        # 评估指标
│   │
│   ├── config.py             # 全局配置
│   └── constants.py          # 常量定义
│
├── scripts/                  # 可执行脚本
│   ├── train_model.py        # 训练模型脚本
│   ├── realtime_demo.py      # 实时演示脚本
│   └── benchmark.py          # 性能测试脚本
│
├── tests/                    # 单元测试
│   ├── test_data_generation.py
│   ├── test_model.py
│   └── test_realtime.py
│
├── requirements.txt          # Python依赖
├── README.md                 # 项目文档
└── .gitignore
```