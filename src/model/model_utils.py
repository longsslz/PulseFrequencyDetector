# src/model/model_utils.py
import torch
import torch.onnx
from torchsummary import summary


def count_parameters(model):
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model):
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def print_model_summary(model, input_size=None):
    """打印模型摘要信息"""
    print("Model Summary:")
    print("=" * 50)

    # 使用torchsummary打印摘要
    if input_size is not None:
        try:
            summary(model, input_size=input_size)
        except:
            print("Could not generate full summary with input size")

    # 打印参数统计
    total_params = count_parameters(model)
    size_mb = model_size_in_mb(model)

    print("\nParameter Statistics:")
    print(f"Total trainable parameters: {total_params}")
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 50)


def save_model_onnx(model, path, input_size, device='cpu'):
    """
    将模型保存为ONNX格式

    参数:
        model (nn.Module): 要保存的模型
        path (str): 保存路径
        input_size (tuple): 输入尺寸 (batch, seq_len, features)
        device (str): 设备 ('cpu' 或 'cuda')
    """
    # 确保模型在正确设备上
    model.to(device)
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(*input_size, device=device)

    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model saved as ONNX to {path}")


def load_model_onnx(path, providers=None):
    """
    加载ONNX模型

    参数:
        path (str): ONNX模型路径
        providers (list): 执行提供者列表

    返回:
        onnx.ModelProto: 加载的ONNX模型
    """
    import onnxruntime as ort

    if providers is None:
        providers = ['CPUExecutionProvider']

    # 创建ONNX运行时会话
    ort_session = ort.InferenceSession(path, providers=providers)

    print(f"Loaded ONNX model from {path}")
    return ort_session


def compare_models(model1, model2, input_tensor, rtol=1e-5, atol=1e-8):
    """
    比较两个模型的输出

    参数:
        model1 (nn.Module): 第一个模型
        model2 (nn.Module): 第二个模型
        input_tensor (torch.Tensor): 输入张量
        rtol (float): 相对容差
        atol (float): 绝对容差

    返回:
        bool: 如果输出在容差范围内相同则为True
    """
    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)

    return torch.allclose(output1, output2, rtol=rtol, atol=atol)


def benchmark_model(model, input_tensor, num_runs=100, warmup=10):
    """
    基准测试模型推理时间

    参数:
        model (nn.Module): 要测试的模型
        input_tensor (torch.Tensor): 输入张量
        num_runs (int): 运行次数
        warmup (int): 预热次数

    返回:
        dict: 包含平均时间和标准差的字典
    """
    model.eval()
    times = []

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 基准测试
    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(input_tensor)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {
        'average_time_ms': avg_time,
        'std_dev_ms': std_dev,
        'fps': 1000 / avg_time if avg_time > 0 else float('inf')
    }