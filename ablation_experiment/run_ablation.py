import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import AIS_Ablation_Model
# 导入咱们改好的工具函数
from utils import get_dataloader, train_one_epoch, validate, calculate_auroc

KNOWN_PATH = r"E:\gratuate_design\data_known"
UNKNOWN_PATH = r"E:\gratuate_design\data_unknown"

def run_experiment():
    # 1. 准备数据 (这里会触发 AIS_Mat_Dataset 的加载)
    print("正在初始化数据加载器...")
    train_loader = get_dataloader(KNOWN_PATH, batch_size=32)
    known_test_loader = get_dataloader(KNOWN_PATH, batch_size=32)
    unknown_test_loader = get_dataloader(UNKNOWN_PATH, batch_size=32)

    configs = [
        {"name": "Full_Model", "resnet": True, "trans": True},
        {"name": "No_Transformer", "resnet": True, "trans": False},
        {"name": "No_ResNet", "resnet": False, "trans": True}
    ]

    results = []
    for cfg in configs:
        print(f"\n🚀 消融组: {cfg['name']}")
        model = AIS_Ablation_Model(use_resnet=cfg['resnet'], use_transformer=cfg['trans']).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(30):
            loss = train_one_epoch(model, optimizer, criterion, train_loader)
            acc = validate(model, known_test_loader) # 验证集暂用已知类测试集
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"best_{cfg['name']}.pth")
            print(f"Epoch {epoch+1}/30 - Loss: {loss:.4f}, Acc: {acc:.4f}")

        # 2. 关键：评估该架构的开集 AUROC
        # 注意：这里的 calculate_auroc 内部也要使用咱们的 AIS_Mat_Dataset 逻辑
        auroc = calculate_auroc(model, f"best_{cfg['name']}.pth", known_test_loader, unknown_test_loader)
        
        results.append({
            "Experiment": cfg['name'],
            "Best_ACC": best_acc,
            "AUROC": auroc
        })

    # 3. 输出 CSV
    pd.DataFrame(results).to_csv("ablation_results_report.csv", index=False)
    print("\n✅ 消融实验全部跑完！请查看 ablation_results_report.csv")

if __name__ == "__main__":
    run_experiment()