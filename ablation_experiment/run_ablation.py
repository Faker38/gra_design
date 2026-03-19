import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import AIS_Ablation_Model
from utils import get_dataloader
from arpl_loss import ARPLLoss
from sklearn.metrics import roc_auc_score

# 配置升级
KNOWN_PATH = r"E:\gratuate_design\data_known"
UNKNOWN_PATH = r"E:\gratuate_design\data_unknown"
EPOCHS = 100 # 👈 增加到 100 轮以保证充分收敛
LR = 5e-4

def calculate_auroc(model, criterion, k_loader, u_loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for loader, label_val in [(k_loader, 0), (u_loader, 1)]:
            for data, _ in loader:
                feat = model(data.cuda())
                # 计算到互惠点的最小距离
                dists = torch.cdist(feat, criterion.reciprocal_points)
                min_dists, _ = torch.min(dists, dim=1)
                scores.extend(min_dists.cpu().numpy())
                labels.extend([label_val] * data.size(0))
    return roc_auc_score(labels, scores)

def run_experiment():
    train_loader = get_dataloader(KNOWN_PATH)
    u_loader = get_dataloader(UNKNOWN_PATH)
    
    # 既然是冲刺高分，我们只跑 Full_Model，或者按需保留其他组
    configs = [
        {"name": "Full_Model_Pro", "resnet": True, "trans": True},
        {"name": "No_Transformer_Pro", "resnet": True, "trans": False},
        {"name": "No_ResNet_Pro", "resnet": False, "trans": True}
    ]

    results = []
    for cfg in configs:
        print(f"\n🚀 启动 PRO 版训练: {cfg['name']}")
        model = AIS_Ablation_Model(use_resnet=cfg['resnet'], use_transformer=cfg['trans']).cuda()
        criterion = ARPLLoss(num_classes=42, feat_dim=128).cuda()
        
        optimizer = optim.AdamW([
            {'params': model.parameters()},
            {'params': criterion.parameters(), 'lr': LR}
        ], lr=LR, weight_decay=1e-2)
        
        # 引入余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()
                
                # --- 核心：引入训练噪声 (模拟 SNR 波动，增强鲁棒性) ---
                noise = torch.randn_like(data) * 0.02 # 约 34dB SNR
                data = data + noise
                
                optimizer.zero_grad()
                feat = model(data)
                loss, logits = criterion(feat, target)
                loss.backward()
                optimizer.step()
                
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            scheduler.step() # 更新学习率
            acc = correct / total
            if acc > best_acc: best_acc = acc
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        auroc = calculate_auroc(model, criterion, train_loader, u_loader)
        results.append({"Experiment": cfg['name'], "Best_ACC": best_acc, "AUROC": auroc})
        print(f"⭐ {cfg['name']} 冲刺完成! AUROC: {auroc:.4f}")

    pd.DataFrame(results).to_csv("pro_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()