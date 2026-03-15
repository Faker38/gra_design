import torch

print("--- 环境验证报告 ---")
# 1. 验证版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. 验证 CUDA（这个返回 True 才是真的成了）
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

# 3. 验证显卡能否进行张量计算
if cuda_available:
    print(f"当前显卡型号: {torch.cuda.get_device_name(0)}")
    # 尝试在显卡上创建一个张量并计算
    x = torch.ones(1, device="cuda")
    print(f"显卡张量计算验证: {'成功' if x.item() == 1 else '失败'}")