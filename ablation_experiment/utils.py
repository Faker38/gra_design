import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# --- 1. 数据集定义：处理 .mat 复数信号 ---
class AIS_Mat_Dataset(Dataset):
    def __init__(self, root_dir, length=1024):
        self.samples = []
        self.labels = []
        
        # 获取所有 .mat 文件并排序，保证 ID 映射固定
        file_list = [f for f in os.listdir(root_dir) if f.endswith('.mat')]
        file_list.sort()

        print(f"🕒 正在从 {root_dir} 加载数据...")
        for idx, file_name in enumerate(file_list):
            file_path = os.path.join(root_dir, file_name)
            
            # 加载 mat (这里使用 simplify_cells 处理较新版本的 mat)
            try:
                mat_dict = scio.loadmat(file_path)
                raw_data = mat_dict['data'] # 对应你图片里的变量名 'data'
            except Exception as e:
                print(f"❌ 读取文件 {file_name} 出错: {e}")
                continue
            
            # 遍历每一行样本
            num_samples = raw_data.shape[0]
            for i in range(num_samples):
                # 1. 提取复数信号并截断/补齐到指定长度
                sig = raw_data[i, :length]
                if len(sig) < length: # 如果长度不足则补零
                    sig = np.pad(sig, (0, length - len(sig)), 'constant')
                
                # 2. 拆分实部(I)和虚部(Q)
                real_part = np.real(sig)
                imag_part = np.imag(sig)
                
                # 3. 堆叠成 (2, 1024) 形状
                combined = np.stack([real_part, imag_part], axis=0)
                
                self.samples.append(combined.astype(np.float32))
                self.labels.append(idx)
        
        print(f"✅ 数据加载完成！总计 ID 数: {len(file_list)}, 总样本数: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])

# --- 2. 数据加载器封装 ---
def get_dataloader(path, batch_size=32, shuffle=True):
    dataset = AIS_Mat_Dataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- 3. 训练函数 ---
def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        _, logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

# --- 4. 验证函数 (闭集准确率) ---
def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            _, logits = model(data)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

# --- 5. 开集评估函数 (AUROC) ---
def calculate_auroc(model, weights_path, known_loader, unknown_loader):
    # 加载该实验组表现最好的模型权重
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    scores = []
    labels = []
    
    print(f"🔍 正在计算 {weights_path} 的开集 AUROC...")
    with torch.no_grad():
        # 处理已知类 (标签为 0)
        for data, _ in known_loader:
            feat, _ = model(data.cuda())
            # 异常分数逻辑：特征向量的 L2 范数（距离原点越远，特征越强）
            # 或者可以改用你之前习惯的距离逻辑
            s = torch.norm(feat, p=2, dim=1)
            scores.extend(s.cpu().numpy())
            labels.extend([0] * data.size(0))
            
        # 处理未知类 (标签为 1)
        for data, _ in unknown_loader:
            feat, _ = model(data.cuda())
            s = torch.norm(feat, p=2, dim=1)
            scores.extend(s.cpu().numpy())
            labels.extend([1] * data.size(0))
            
    return roc_auc_score(labels, scores)