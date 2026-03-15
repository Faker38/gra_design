import torch
import numpy as np
import os
import scipy.io as sio
from arpl_model import ARPLFeatureExtractor 
from arpl_loss import ARPLLoss
from arpl_data import get_arpl_loader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# --- 配置区 ---
CHECKPOINT_PATH = "arpl_checkpoint_epoch_50.pth" 
UNKNOWN_DATA_PATH = r"E:\gratuate_design\unknown_data" 
KNOWN_DATA_PATH = r"E:\gratuate_design\data"     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIM = 128
NUM_CLASSES = 42 

def calculate_metrics(known_dists, unknown_dists):
    y_true = [0] * len(known_dists) + [1] * len(unknown_dists)
    y_scores = known_dists + unknown_dists
    
    # 1. 计算 AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # 2. 计算 EER
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # 3. 计算 85% 置信度下的阈值和误报率 (根据你之前的实验)
    threshold_85 = np.percentile(known_dists, 85)
    false_alarms = sum(d < threshold_85 for d in unknown_dists) / len(unknown_dists)
    
    print("\n" + "="*40)
    print("📊 开集识别 (OSR) 核心性能报告")
    print("="*40)
    print(f"✅ AUROC 指标: {auroc:.4f}")
    print(f"✅ EER (等错误率): {eer*100:.2f}%")
    print(f"✅ 判定阈值 (85% 置信度): {threshold_85:.4f}")
    print(f"⚠️ 对应误报率: {false_alarms*100:.2f}%")
    print("="*40)
    return threshold_85

def evaluate_osr():
    model = ARPLFeatureExtractor(feat_dim=FEAT_DIM).to(DEVICE)
    criterion = ARPLLoss(num_classes=NUM_CLASSES, feat_dim=FEAT_DIM).to(DEVICE)
    
    print(f"正在加载模型权重: {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])
    criterion.load_state_dict(checkpoint['criterion'])
    model.eval()

    def get_distances(path):
        loader = get_arpl_loader(path, batch_size=32)
        all_dists = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(DEVICE)
                feat = model(data)
                dists = torch.cdist(feat, criterion.reciprocal_points)
                min_dists, _ = torch.min(dists, dim=1)
                all_dists.extend(min_dists.cpu().numpy())
        return all_dists

    print("🕒 正在分析已知类信号指纹分布...")
    known_dists = get_distances(KNOWN_DATA_PATH)
    print("🕒 正在分析未知类信号指纹分布...")
    unknown_dists = get_distances(UNKNOWN_DATA_PATH)

    # 量化分析输出
    final_thresh = calculate_metrics(known_dists, unknown_dists)

    # 可视化绘图
    plt.figure(figsize=(10, 6))
    plt.hist(known_dists, bins=50, alpha=0.6, label='Known IDs', color='#4A90E2', density=True)
    plt.hist(unknown_dists, bins=50, alpha=0.6, label='Unknown IDs', color='#D0021B', density=True)
    
    # 画出 85% 分界线（保持一致性）
    plt.axvline(x=final_thresh, color='green', linestyle='--', label=f'Threshold ({final_thresh:.2f})')
    
    plt.title("AIS Fingerprint ARPL OSR Result", fontsize=14)
    plt.xlabel("Distance to Reciprocal Points", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    
    print("\n💡 提示：请关闭图片窗口查看最终 EER 报告。")
    plt.show()

if __name__ == "__main__":
    evaluate_osr()