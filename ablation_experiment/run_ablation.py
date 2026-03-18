import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import AIS_Ablation_Model
from utils import get_dataloader
from arpl_loss import ARPLLoss # 建议直接使用你之前的 arpl_loss.py
from sklearn.metrics import roc_auc_score

# 配置
KNOWN_PATH = r"E:\gratuate_design\data_known"
UNKNOWN_PATH = r"E:\gratuate_design\data_unknown"
EPOCHS = 50
LR = 5e-4

def calculate_auroc(model, criterion, k_loader, u_loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for loader, label_val in [(k_loader, 0), (u_loader, 1)]:
            for data, _ in loader:
                feat = model(data.cuda())
                # 使用到互惠点的最小距离作为异常分数
                dists = torch.cdist(feat, criterion.reciprocal_points)
                min_dists, _ = torch.min(dists, dim=1)
                scores.extend(min_dists.cpu().numpy())
                labels.extend([label_val] * data.size(0))
    return roc_auc_score(labels, scores)

def run_experiment():
    train_loader = get_dataloader(KNOWN_PATH)
    u_loader = get_dataloader(UNKNOWN_PATH)
    
    configs = [
        {"name": "Full_Model", "resnet": True, "trans": True},
        {"name": "No_Transformer", "resnet": True, "trans": False},
        {"name": "No_ResNet", "resnet": False, "trans": True}
    ]

    results = []
    for cfg in configs:
        print(f"\n🚀 启动消融组: {cfg['name']}")
        model = AIS_Ablation_Model(use_resnet=cfg['resnet'], use_transformer=cfg['trans']).cuda()
        criterion = ARPLLoss(num_classes=42, feat_dim=128).cuda()
        
        # 必须同步优化互惠点
        optimizer = optim.AdamW([
            {'params': model.parameters()},
            {'params': criterion.parameters(), 'lr': LR}
        ], lr=LR, weight_decay=1e-2)
        
        best_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                feat = model(data)
                loss, logits = criterion(feat, target)
                loss.backward()
                optimizer.step()
                
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            acc = correct / total
            if acc > best_acc: best_acc = acc
            print(f"Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.4f}")

        # 计算开集指标
        auroc = calculate_auroc(model, criterion, train_loader, u_loader)
        results.append({"Experiment": cfg['name'], "Best_ACC": best_acc, "AUROC": auroc})
        print(f"⭐ {cfg['name']} 完成! AUROC: {auroc:.4f}")

    pd.DataFrame(results).to_csv("final_ablation_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()