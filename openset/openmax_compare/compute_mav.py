import torch
import numpy as np
import os
from arpl_model import ARPLFeatureExtractor
from arpl_data import get_arpl_loader

# 配置
CHECKPOINT_PATH = "arpl_checkpoint_epoch_50.pth"
TRAIN_DATA_PATH = r"E:\gratuate_design\data_known" # 使用训练集
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 42

def compute_mavs():
    model = ARPLFeatureExtractor(feat_dim=128).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    loader = get_arpl_loader(TRAIN_DATA_PATH, batch_size=32)
    
    class_features = {i: [] for i in range(NUM_CLASSES)}
    
    print("正在提取各类别特征中心...")
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            feat = model(data).cpu().numpy()
            target = target.numpy()
            for f, t in zip(feat, target):
                class_features[t].append(f)

    mavs = []
    dists_to_mav = []
    
    for i in range(NUM_CLASSES):
        feats = np.array(class_features[i])
        # 计算均值向量 (MAV)
        mav = np.mean(feats, axis=0)
        mavs.append(mav)
        # 计算该类所有样本到 MAV 的距离，存起来后面做 Weibull 拟合
        dists = np.linalg.norm(feats - mav, axis=1)
        dists_to_mav.append(dists)
        
    # 保存结果，下一步 OpenMax 测试要用
    np.save("mavs.npy", np.array(mavs))
    np.save("dists_to_mav.npy", np.array(dists_to_mav, dtype=object))
    print("✅ MAV 和距离分布已保存！")

if __name__ == "__main__":
    compute_mavs()