# src/model/quantize.py
import torch
import torch.quantization
from torch.quantization import quantize_dynamic, quantize_qat


def quantize_model_dynamic(model, qconfig_spec=None, dtype=torch.qint8):
    """
    动态量化模型

    参数:
        model (nn.Module): 要量化的模型
        qconfig_spec (set or dict): 要量化的层类型
        dtype: 量化数据类型 (torch.qint8 或 torch.float16)

    返回:
        nn.Module: 量化后的模型
    """
    if qconfig_spec is None:
        qconfig_spec = {torch.nn.Linear, torch.nn.GRU, torch.nn.Conv1d}

    quantized_model = quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=dtype
    )

    return quantized_model


def prepare_model_for_quantization(model):
    """
    准备模型用于静态量化

    参数:
        model (nn.Module): 要准备的模型

    返回:
        nn.Module: 准备好的模型
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 融合模型中的层
    if hasattr(model, 'conv_layers'):
        for conv in model.conv_layers:
            torch.quantization.fuse_modules(conv, [['0', '1', '2']], inplace=True)

    prepared_model = torch.quantization.prepare(model)
    return prepared_model


def quantize_model_static(prepared_model, calibration_data):
    """
    静态量化模型

    参数:
        prepared_model (nn.Module): 准备好的模型
        calibration_data (DataLoader): 用于校准的数据

    返回:
        nn.Module: 量化后的模型
    """
    # 使用校准数据校准模型
    with torch.no_grad():
        for data, _ in calibration_data:
            prepared_model(data)

    # 转换模型
    quantized_model = torch.quantization.convert(prepared_model)

    return quantized_model


def quantize_model_qat(model, train_loader, num_batches=10):
    """
    量化感知训练

    参数:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        num_batches (int): 用于训练的批次数量

    返回:
        nn.Module: 量化感知训练后的模型
    """
    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # 融合模型中的层
    if hasattr(model, 'conv_layers'):
        for conv in model.conv_layers:
            torch.quantization.fuse_modules(conv, [['0', '1', '2']], inplace=True)

    # 准备和训练
    model.train()
    prepared_model = torch.quantization.prepare_qat(model)

    # 进行少量训练
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(prepared_model.parameters(), lr=0.001)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        optimizer.zero_grad()
        output = prepared_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 转换为量化模型
    quantized_model = torch.quantization.convert(prepared_model)

    return quantized_model


def print_quantization_info(model):
    """打印模型的量化信息"""
    print("Quantization Information:")
    print("=" * 50)

    for name, module in model.named_modules():
        if isinstance(module, torch.quantization.QuantWrapper):
            print(f"Layer: {name}")
            print(f"  Original: {module.original_module}")
            print(f"  Quantized: {module.quantized_module}")
            print("-" * 40)

    print("=" * 50)