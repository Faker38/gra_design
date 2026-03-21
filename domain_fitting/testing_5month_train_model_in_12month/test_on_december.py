import torch
from torch.utils.data import DataLoader
from model import AIS_Ablation_Model
from arpl_loss import ARPLLoss
from sklearn.metrics import roc_auc_score
import numpy as np

# --- 1. 加载模型与 12 月数据 ---
print("🕒 正在加载 12 月跨域测试包...")
checkpoint_data = torch.load(r"E:\gratuate_design\processed_pt\dec_target.pt")
test_data = checkpoint_data['data'] 

# 加载训练好的 5 月模型权重
ckpt_model = torch.load('best_may_baseline.pth')
num_classes = len(ckpt_model['mapping'])

model = AIS_Ablation_Model(num_classes=num_classes).cuda()
model.load_state_dict(ckpt_model['model'])
criterion = ARPLLoss(num_classes=num_classes, feat_dim=128).cuda()
criterion.load_state_dict(ckpt_model['criterion'])
model.eval()

# --- 2. 分类统计 (18 个重合 ID vs 81 个新增 ID) ---
closed_correct, closed_total = 0, 0
all_scores, all_labels = [], []

print("🚀 开始跨域性能评估...")
with torch.no_grad():
    for data, label in test_data:
        data = data.unsqueeze(0).cuda()
        feat = model(data)
        
        # 计算到互惠点的最小距离作为“异常得分”
        dists = torch.cdist(feat, criterion.reciprocal_points)
        min_dist, _ = torch.min(dists, dim=1)
        all_scores.append(min_dist.item())
        
        # 判定是否为重合 ID (Label >= 0)
        if label >= 0:
            _, pred = torch.min(dists, dim=1) # 距离越小越接近该类
            if pred.item() == label:
                closed_correct += 1
            closed_total += 1
            all_labels.append(0) # 已知类标签为 0
        else:
            all_labels.append(1) # 未知类标签为 1

# --- 3. 打印残酷的现实 ---
acc_closed = closed_correct / closed_total if closed_total > 0 else 0
auroc = roc_auc_score(all_labels, all_scores)

print("\n" + "="*40)
print("📊 12月数据跨域测试报告")
print("="*40)
print(f"📉 闭集迁移准确率 (18个重合ID): {acc_closed:.4f}")
print(f"🛡️ 开集拒识能力 (AUROC): {auroc:.4f}")
print("-" * 40)
print("💡 结论分析：")
if acc_closed < 0.7:
    print("⚠️ 警告：环境偏移严重！模型在 12 月完全认不出老朋友了，急需领域自适应 (DA)！")
else:
    print("✅ 表现尚可：模型具备一定的跨环境泛化能力。")