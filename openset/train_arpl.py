import torch
import torch.optim as optim
from arpl_model import ARPLFeatureExtractor
from arpl_loss import ARPLLoss
from arpl_data import get_arpl_loader

# 配置
DATA_PATH = r"E:\gratuate_design\data" # 修改为你的路径
FEAT_DIM = 128
NUM_CLASSES = 42
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4

device = torch.device("cuda")
loader = get_arpl_loader(DATA_PATH, BATCH_SIZE)

# 初始化
model = ARPLFeatureExtractor(feat_dim=FEAT_DIM).to(device)
criterion = ARPLLoss(num_classes=NUM_CLASSES, feat_dim=FEAT_DIM).to(device)

# 注意：ARPL 的互惠点也是需要优化的参数
optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': criterion.parameters(), 'lr': LR}
], lr=LR, weight_decay=1e-2)

scaler = torch.amp.GradScaler('cuda')

def train():
    print("🔥 开始 ARPL 开集识别训练模式...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                feat = model(data)
                loss, logits = criterion(feat, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, pred = logits.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

        acc = 100.*correct/total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.3f} | Train Acc: {acc:.2f}%")
        
        # 保存：ARPL 需要保存模型和 criterion(含互惠点)
        if acc > 90:
            torch.save({
                'model': model.state_dict(),
                'criterion': criterion.state_dict()
            }, f"arpl_checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()