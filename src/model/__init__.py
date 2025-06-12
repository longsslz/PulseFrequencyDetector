# src/model/__init__.py

from .architecture import (
    PulseFrequencyDetector,
    LitePulseDetector,
    DeepPulseDetector,
    TemporalConvNetDetector,
    create_model
)
from .quantize import (
    quantize_model_dynamic,
    quantize_model_static,
    prepare_model_for_quantization
)
from .model_utils import (
    count_parameters,
    model_size_in_mb,
    print_model_summary,
    save_model_onnx,
    load_model_onnx
)

# 可选：定义 __all__ 列表，明确指定可以被导入的内容
__all__ = [
    # 模型架构
    'PulseFrequencyDetector',
    'LitePulseDetector',
    'DeepPulseDetector',
    'TemporalConvNetDetector',
    'create_model',  # 添加这行！

    # 量化工具
    'quantize_model_dynamic',
    'quantize_model_static',
    'prepare_model_for_quantization',

    # 模型工具
    'count_parameters',
    'model_size_in_mb',
    'print_model_summary',
    'save_model_onnx',
    'load_model_onnx'
]