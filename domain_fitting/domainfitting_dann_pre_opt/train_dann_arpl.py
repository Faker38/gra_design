import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_dann import AIS_DANN_Model
from arpl_loss import ARPLLoss
import numpy as np
import matplotlib.pyplot as plt # 👈 新增：绘图库

# --- 1. 加载双域物资包 ---
source_ckpt = torch.load(r"E:\gratuate_design\processed_pt\may_source.pt")
target_ckpt = torch.load(r"E:\gratuate_design\processed_pt\dec_target.pt")

source_loader = DataLoader(source_ckpt['data'], batch_size=32, shuffle=True, drop_last=True)
target_loader = DataLoader(target_ckpt['data'], batch_size=32, shuffle=True, drop_last=True)

num_classes = len(source_ckpt['mapping']) 

# --- 2. 初始化 DANN 架构 ---
model = AIS_DANN_Model(num_classes=num_classes).cuda()
criterion_osr = ARPLLoss(num_classes=num_classes, feat_dim=128).cuda()
criterion_domain = torch.nn.CrossEntropyLoss().cuda()

optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': criterion_osr.parameters(), 'lr': 1e-4}
], lr=1e-4, weight_decay=1e-2)

# --- 3. 新增：日志容器 ---
history = {
    'epoch': [],
    'osr_loss': [],
    'domain_loss': [],
    'alpha': []
}

# --- 4. 对抗训练循环 ---
print(f"🚀 DANN + ARPL 联合训练启动！类别数: {num_classes}")

for epoch in range(100):
    model.train()
    p = float(epoch) / 100
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    len_dataloader = min(len(source_loader), len(target_loader))
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_loader)

    epoch_osr_loss = 0
    epoch_dom_loss = 0

    for i in range(len_dataloader):
        # A. 源域训练
        s_img, s_label = next(data_source_iter)
        s_img, s_label = s_img.cuda(), s_label.cuda()
        
        # 修正：动态获取 batch size
        domain_label_s = torch.zeros(s_img.size(0)).long().cuda() 
        
        feat_s, domain_s = model(s_img, alpha=alpha)
        loss_osr, _ = criterion_osr(feat_s, s_label)
        loss_domain_s = criterion_domain(domain_s, domain_label_s)

        # B. 目标域训练
        t_img, _ = next(data_target_iter)
        t_img = t_img.cuda()
        domain_label_t = torch.ones(t_img.size(0)).long().cuda()
        
        _, domain_t = model(t_img, alpha=alpha)
        loss_domain_t = criterion_domain(domain_t, domain_label_t)

        # C. 总损失
        loss = loss_osr + (loss_domain_s + loss_domain_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_osr_loss += loss_osr.item()
        epoch_dom_loss += (loss_domain_s + loss_domain_t).item()

    # 记录本轮平均损失
    avg_osr = epoch_osr_loss / len_dataloader
    avg_dom = epoch_dom_loss / len_dataloader
    history['epoch'].append(epoch + 1)
    history['osr_loss'].append(avg_osr)
    history['domain_loss'].append(avg_dom)
    history['alpha'].append(alpha)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/100 | OSR: {avg_osr:.4f} | Dom: {avg_dom:.4f} | Alpha: {alpha:.3f}")

# --- 5. 存储权重与绘图 ---
torch.save({
    'model': model.state_dict(),
    'criterion': criterion_osr.state_dict(), # 👈 必须存下这 32 个“坐标点”
    'mapping': source_ckpt['mapping']
}, 'best_dann_full_package.pth')
print("✅ 完整模型包（含互惠点坐标）已保存！")

def save_and_plot(history):
    # 保存文本日志
    with open('training_history.txt', 'w') as f:
        f.write("epoch,osr_loss,domain_loss,alpha\n")
        for i in range(len(history['epoch'])):
            f.write(f"{history['epoch'][i]},{history['osr_loss'][i]},{history['domain_loss'][i]},{history['alpha'][i]}\n")
    
    # 绘制趋势图
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['epoch'], history['osr_loss'], 'b-', label='OSR_Loss (Ship ID)')
    plt.ylabel('Loss')
    plt.title('Classification Task (认船任务)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['epoch'], history['domain_loss'], 'r-', label='Domain_Loss (Environment)')
    plt.plot(history['epoch'], history['alpha'], 'g--', label='Alpha (Adversarial Strength)', alpha=0.5)
    plt.ylabel('Value')
    plt.title('Adversarial Task & Alpha (环境对抗与强度)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dann_training_plot.png')
    print("📈 趋势图已保存为 dann_training_plot.png")
    plt.show()

save_and_plot(history)
print("✅ DANN 对抗模型训练完成！")