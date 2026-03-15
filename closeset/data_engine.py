import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class AISDataset(Dataset):
    def __init__(self, data_root, seq_len=1024):
        self.data_root = data_root
        self.seq_len = seq_len
        self.file_list = sorted([f for f in os.listdir(data_root) if f.endswith('.mat')])
        
        self.all_data = []
        self.all_labels = []
        class_counts = []
        
        print(f"🚀 正在扫描 52 个 ID，预加载数据中...")
        for idx, file_name in enumerate(self.file_list):
            path = os.path.join(data_root, file_name)
            mat = sio.loadmat(path)
            var_name = [k for k in mat.keys() if not k.startswith('__')][0]
            matrix = mat[var_name]
            
            num_rows = matrix.shape[0]
            class_counts.append(num_rows)
            
            for r_idx in range(num_rows):
                raw_sig = matrix[r_idx, :]
                # 能量归一化
                energy = np.sqrt(np.mean(np.abs(raw_sig)**2))
                sig_norm = raw_sig / (energy + 1e-12)
                
                # 截断填充
                if len(sig_norm) > self.seq_len:
                    sig_norm = sig_norm[:self.seq_len]
                else:
                    sig_norm = np.pad(sig_norm, (0, self.seq_len - len(sig_norm)))
                
                # 转为 I/Q 张量 [2, 1024]
                iq_tensor = np.stack([np.real(sig_norm), np.imag(sig_norm)], axis=0).astype(np.float32)
                self.all_data.append(torch.from_numpy(iq_tensor))
                self.all_labels.append(idx)
        
        weights_per_class = 1.0 / np.array(class_counts)
        self.sample_weights = [weights_per_class[l] for l in self.all_labels]
        print(f"✅ 加载完毕！总样本: {len(self.all_data)}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

def get_dataloader(data_path, batch_size=128): # 默认改为 128
    dataset = AISDataset(data_path)
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset.sample_weights), replacement=True)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,      # Windows 下预加载数据后设为 0 更快
        pin_memory=True     # 加速数据向 GPU 拷贝
    )
# data_engine.py 文件的最后几行
if __name__ == "__main__":
    TEST_PATH = r"E:\gratuate_design\data" # <--- 这里也要换成同样的路径