import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class ARPLDataset(Dataset):
    def __init__(self, data_root, seq_len=1024):
        self.file_list = sorted([f for f in os.listdir(data_root) if f.endswith('.mat')])
        self.all_data, self.all_labels = [], []
        class_counts = []

        print("🚀 ARPL 数据预热中...")
        for idx, f in enumerate(self.file_list):
            mat = sio.loadmat(os.path.join(data_root, f))
            var = [k for k in mat.keys() if not k.startswith('__')][0]
            matrix = mat[var]
            class_counts.append(matrix.shape[0])
            for r in range(matrix.shape[0]):
                sig = matrix[r, :]
                energy = np.sqrt(np.mean(np.abs(sig)**2))
                sig = sig / (energy + 1e-12)
                # 填充/截断
                sig = sig[:seq_len] if len(sig) > seq_len else np.pad(sig, (0, seq_len-len(sig)))
                iq = np.stack([np.real(sig), np.imag(sig)], axis=0).astype(np.float32)
                self.all_data.append(torch.from_numpy(iq))
                self.all_labels.append(idx)
        
        weights = 1.0 / np.array(class_counts)
        self.sample_weights = [weights[l] for l in self.all_labels]

    def __len__(self): return len(self.all_data)
    def __getitem__(self, idx): return self.all_data[idx], self.all_labels[idx]

def get_arpl_loader(path, batch_size=128):
    ds = ARPLDataset(path)
    sampler = WeightedRandomSampler(ds.sample_weights, len(ds.sample_weights))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, pin_memory=True)