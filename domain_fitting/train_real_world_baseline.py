import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import AIS_Ablation_Model
from arpl_loss import ARPLLoss
import os

# --- 1. 加载加工好的 pt 数据 ---
print("🕒 正在从‘速冻包’加载 5 月源域数据...")
checkpoint = torch.load(r"E:\gratuate_design\processed_pt\may_source.pt")
full_data = checkpoint['data'] # 格式为 [(tensor, label), ...]
id_mapping = checkpoint['mapping']
num_classes = len(id_mapping) # 应该是 32

# 简单划分训练集和验证集 (90% 训练, 10% 验证)
train_size = int(0.9 * len(full_data))
train_ds = [full_data[i] for i in range(train_size)]
val_ds = [full_data[i] for i in range(train_size, len(full_data))]

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# --- 2. 初始化模型与损失函数 ---
# 既然消融实验证明 Full_Model 最强，咱们直接上全量版
# 这里传入的 num_classes 会覆盖 model.py 里的默认值 42
model = AIS_Ablation_Model(num_classes=num_classes).cuda() 
# 同时告诉损失函数，现在要划定 32 个“互惠点”
criterion = ARPLLoss(num_classes=num_classes, feat_dim=128).cuda()

optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': criterion.parameters(), 'lr': 5e-4}
], lr=5e-4, weight_decay=1e-2)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# --- 3. 训练循环 ---
print(f"🚀 开始在 5 月实测数据上训练 (类别数: {num_classes})...")
best_acc = 0

for epoch in range(100):
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        # 别忘了加入你 0.90 高分的功臣：训练噪声
        noise = torch.randn_like(data) * 0.02
        data = data + noise
        
        optimizer.zero_grad()
        feat = model(data)
        loss, logits = criterion(feat, target)
        loss.backward()
        optimizer.step()
        
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    scheduler.step()
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model': model.state_dict(),
            'criterion': criterion.state_dict(),
            'mapping': id_mapping
        }, 'best_may_baseline.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 | May_Acc: {acc:.4f} | Best: {best_acc:.4f}")

print(f"✅ 5月基准模型训练完成！最高准确率: {best_acc:.4f}")