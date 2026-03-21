import torch
from torch.utils.data import DataLoader
from model_dann import AIS_DANN_Model
from arpl_loss import ARPLLoss
from sklearn.metrics import roc_auc_score
import numpy as np

# --- 1. 加载：从“全家桶”读取大脑和罗盘 ---
print("🕒 正在加载 DANN 全量模型包...")
checkpoint_data = torch.load(r"E:\gratuate_design\processed_pt\dec_target.pt")
test_data = checkpoint_data['data']

# 加载我们新存的 full_package
ckpt = torch.load('best_dann_full_package.pth') 

# 初始化并加载模型
model = AIS_DANN_Model(num_classes=32).cuda()
model.load_state_dict(ckpt['model']) # 👈 从包里取模型参数
model.eval()

# 初始化并加载 ARPL 坐标点
criterion = ARPLLoss(num_classes=32, feat_dim=128).cuda()
criterion.load_state_dict(ckpt['criterion']) # 👈 关键：加载训练好的互惠点
criterion.eval()

# --- 2. 跨域测试逻辑 (保持不变，但使用正确的 criterion) ---
closed_correct, closed_total = 0, 0
all_scores, all_labels = [], []

print("🚀 正在对 12 月实测数据进行‘复仇测试’...")

with torch.no_grad():
    for data, label in test_data:
        data = data.unsqueeze(0).cuda()
        feat, _ = model(data, alpha=0.0) 
        
        # 使用加载好的、属于 5 月份知识体系的互惠点计算距离
        dists = torch.cdist(feat, criterion.reciprocal_points)
        min_dist, pred = torch.min(dists, dim=1)
        
        all_scores.append(min_dist.item())
        
        if label >= 0: # 老朋友 (18个 ID)
            if pred.item() == label:
                closed_correct += 1
            closed_total += 1
            all_labels.append(0) 
        else: # 陌生人 (81个 ID)
            all_labels.append(1)

# --- 3. 结果输出 ---
acc_final = closed_correct / closed_total if closed_total > 0 else 0
auroc_final = roc_auc_score(all_labels, all_scores)

print("\n" + "★"*40)
print("🏆 DANN 领域自适应：正式实战报告")
print("★"*40)
print(f"📈 闭集迁移准确率: {acc_final:.4f}")
print(f"🛡️ 开集拒识 AUROC: {auroc_final:.4f}")
print("★"*40)