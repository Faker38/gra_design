import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import numpy as np
from sklearn.metrics import roc_auc_score

class AIS_Mat_Dataset(Dataset):
    def __init__(self, root_dir, seq_len=1024):
        self.samples, self.labels = [], []
        file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.mat')])

        print(f"🕒 正在加载数据并进行能量归一化: {root_dir}")
        for idx, file_name in enumerate(file_list):
            mat = sio.loadmat(os.path.join(root_dir, file_name))
            # 自动获取变量名
            var_name = [k for k in mat.keys() if not k.startswith('__')][0]
            matrix = mat[var_name]
            
            for r in range(matrix.shape[0]):
                sig = matrix[r, :]
                # --- 能量归一化逻辑 (对齐 arpl_data.py) ---
                energy = np.sqrt(np.mean(np.abs(sig)**2))
                sig = sig / (energy + 1e-12)
                
                # 截断与补齐
                sig = sig[:seq_len] if len(sig) > seq_len else np.pad(sig, (0, seq_len-len(sig)))
                
                # 拆分 I/Q 路
                iq = np.stack([np.real(sig), np.imag(sig)], axis=0).astype(np.float32)
                self.samples.append(torch.from_numpy(iq))
                self.labels.append(idx)
        print(f"✅ 加载完成，总样本数: {len(self.labels)}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def get_dataloader(path, batch_size=64):
    ds = AIS_Mat_Dataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)