import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from arpl_model import ARPLFeatureExtractor
from arpl_data import get_arpl_loader

# --- 配置 ---
CHECKPOINT_PATH = "arpl_checkpoint_epoch_50.pth"
KNOWN_PATH = r"E:\gratuate_design\data_known"
UNKNOWN_PATH = r"E:\gratuate_design\data_unknown"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_features(model, path, max_samples=200):
    """从指定路径提取特征"""
    loader = get_arpl_loader(path, batch_size=32)
    features, labels = [], []
    count = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            feat = model(data)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
            count += data.size(0)
            if count >= max_samples: break # 每个大类取一部分样本即可，画图太挤不好看
    return np.concatenate(features), np.concatenate(labels)

def run_visualize():
    # 1. 加载模型
    model = ARPLFeatureExtractor(feat_dim=128).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 2. 提取特征
    print("正在提取特征...")
    known_feat, known_label = get_features(model, KNOWN_PATH, max_samples=1000)
    unknown_feat, _ = get_features(model, UNKNOWN_PATH, max_samples=300)

    # 合并数据 (未知类全部打上统一标签 -1)
    all_feat = np.concatenate([known_feat, unknown_feat])
    all_label = np.concatenate([known_label, [-1] * len(unknown_feat)])

    # 3. t-SNE 降维
    print("正在进行 t-SNE 降维 (可能需要 1-2 分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded = tsne.fit_transform(all_feat)

    # 4. 绘图
    plt.figure(figsize=(12, 8))
    
    # 画已知类 (用不同的颜色)
    for i in range(42): # 假设 42 个已知类
        idx = (all_label == i)
        if np.any(idx):
            plt.scatter(embedded[idx, 0], embedded[idx, 1], s=15, alpha=0.6)
            
    # 画未知类 (用显眼的红色星号)
    idx_un = (all_label == -1)
    plt.scatter(embedded[idx_un, 0], embedded[idx_un, 1], s=30, c='red', marker='x', label='Unknown (Intruders)', alpha=0.8)

    plt.title("t-SNE Visualization of AIS Fingerprints (ARPL Space)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("tsne_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_visualize()