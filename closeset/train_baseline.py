import torch
import torch.nn as nn
import torch.optim as optim
from data_engine import get_dataloader
from model_lib import AISTransformerModel

# --- 配置区 ---
DATA_PATH = r"E:\gratuate_design\data" # 确保路径正确
BATCH_SIZE = 128  # 3050 显存跑这个模型完全没问题
LR = 5e-4        # 稍微调低初始学习率，更稳
EPOCHS = 50

DEVICE = torch.device("cuda")
model = AISTransformerModel(num_classes=52).to(DEVICE)
train_loader = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)

# 换用 AdamW 优化器，处理 Transformer 的效果比 Adam 好
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# 引入动态学习率调度器
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, 
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS
)

# 混合精度缩放器
scaler = torch.amp.GradScaler('cuda')

def train():
    print(f"🔥 启动优化版训练！Batch Size: {BATCH_SIZE}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用新版 autocast 语法
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # 学习率随步长更新
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 每 50 个 batch 打印一次，减少屏幕 I/O 开销
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] "
                      f"| Loss: {loss.item():.3f} | Acc: {100.*correct/total:.1f}%")
        
        avg_acc = 100.*correct/total
        print(f"✅ Epoch {epoch+1} 结束 | 平均 Loss: {total_loss/len(train_loader):.4f} | 准确率: {avg_acc:.2f}%")
        
        # 只要准确率突破 50%，就开始保存，防止意外
        if avg_acc > 50:
            torch.save(model.state_dict(), f"ais_best_model.pth")

if __name__ == "__main__":
    train()