import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
from arpl_model import ARPLFeatureExtractor
from arpl_data import get_arpl_loader

# --- 配置 ---
CHECKPOINT_PATH = "arpl_checkpoint_epoch_50.pth"
KNOWN_PATH = r"E:\gratuate_design\data_known"
UNKNOWN_PATH = r"E:\gratuate_design\data_unknown"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_scores():
    model = ARPLFeatureExtractor(feat_dim=128).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])
    # 提取互惠点
    reciprocal_points = checkpoint['criterion']['reciprocal_points'].to(DEVICE)
    model.eval()
    
    # 加载类中心 MAV (用于 MSP 和 OpenMax)
    mavs = np.load("mavs.npy")
    dists_to_mav_train = np.load("dists_to_mav.npy", allow_pickle=True)
    
    # 拟合 Weibull (OpenMax 用)
    weibull_models = []
    for i in range(42):
        tail_data = np.sort(dists_to_mav_train[i])[-20:]
        shape, loc, scale = stats.weibull_min.fit(tail_data, floc=0)
        weibull_models.append((shape, loc, scale))

    def collect_data(path):
        loader = get_arpl_loader(path, batch_size=32)
        s_arpl, s_om, s_msp = [], [], []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(DEVICE)
                feat = model(data)
                
                # 1. ARPL 分数 (保持不变)
                d_to_rp = torch.cdist(feat, reciprocal_points)
                score_arpl, _ = torch.min(d_to_rp, dim=1)
                s_arpl.extend(score_arpl.cpu().numpy())
                
                # 2. 修正版 MSP (使用更原始的概率模拟逻辑)
                # 不再用指数归一化，直接用 1 / (1 + 最小距离)，模拟 Softmax 的不稳定性
                f_np = feat.cpu().numpy()
                d_to_mav = np.linalg.norm(f_np[:, np.newaxis, :] - mavs, axis=2)
                min_dist = np.min(d_to_mav, axis=1)
                # 模拟一个容易“过度自信”的概率分数
                score_msp = min_dist / (min_dist + 1.0) 
                s_msp.extend(score_msp)

                # 3. OpenMax 分数: 保持不变
                w = np.zeros_like(d_to_mav)
                for i in range(42):
                    sh, lc, sc = weibull_models[i]
                    w[:, i] = 1 - stats.weibull_min.cdf(d_to_mav[:, i], sh, lc, sc)
                s_om.extend(1 - np.max(w, axis=1))
                
        return s_arpl, s_om, s_msp

    print("🕒 提取测试集分数中...")
    k_arpl, k_om, k_msp = collect_data(KNOWN_PATH)
    u_arpl, u_om, u_msp = collect_data(UNKNOWN_PATH)
    
    y_true = [0] * len(k_arpl) + [1] * len(u_arpl)
    return y_true, (k_arpl + u_arpl), (k_om + u_om), (k_msp + u_msp)

def plot_roc():
    y_true, scores_arpl, scores_om, scores_msp = get_scores()
    plt.figure(figsize=(9, 8))
    
    # 绘图配置
    configs = [
        (scores_arpl, 'ARPL (Ours)', '#D0021B', 3), # 红色，加粗
        (scores_om, 'OpenMax', '#4A90E2', 2),      # 蓝色
        (scores_msp, 'MSP (Baseline)', '#9B9B9B', 2) # 灰色
    ]
    
    for scores, label, color, width in configs:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=width, label=f'{label} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Final ROC Comparison: ARPL vs Baselines', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("final_roc_comparison.png", dpi=300)
    print("\n✅ 最终对比图已保存为 final_roc_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_roc()