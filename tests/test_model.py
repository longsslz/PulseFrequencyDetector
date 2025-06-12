# tests/test_model.py
import pytest
import torch
from src.model import (
    create_model,
    quantize_model_dynamic,
    count_parameters,
    model_size_in_mb
)


class TestModelArchitecture:
    @pytest.mark.parametrize("model_type", ['standard', 'lite', 'deep', 'tcn'])
    def test_model_creation(self, config, model_type):
        """测试模型创建"""
        model = create_model(model_type, config)

        assert model is not None
        assert count_parameters(model) > 0

        # 测试前向传播
        x = torch.randn(1, config.DATA['seq_length'], 1)
        output = model(x)

        assert output.shape == (1, 1)
        assert 0 <= output.item() <= config.DATA['max_freq']

    def test_model_quantization(self, sample_model):
        """测试模型量化"""
        original_size = model_size_in_mb(sample_model)

        quantized_model = quantize_model_dynamic(sample_model)
        quantized_size = model_size_in_mb(quantized_model)

        assert quantized_size < original_size

        # 测试量化模型前向传播
        x = torch.randn(1, 256, 1)
        output = quantized_model(x)
        assert output.shape == (1, 1)


class TestModelUtils:
    def test_count_parameters(self, sample_model):
        """测试参数计数"""
        num_params = count_parameters(sample_model)
        assert num_params > 1000  # 确保不是零或很小的数

    def test_model_size(self, sample_model):
        """测试模型大小计算"""
        size_mb = model_size_in_mb(sample_model)
        assert size_mb > 0.1  # 确保合理的模型大小