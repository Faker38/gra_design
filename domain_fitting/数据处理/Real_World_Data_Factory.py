import os
import scipy.io as sio
import h5py
import torch
import numpy as np
from tqdm import tqdm

# --- 1. 路径配置 ---
MAY_DIR = r"E:\gratuate_design\domian_data\2025_5\2025_5"
DEC_DIR = r"E:\gratuate_design\domian_data\2025_12\2025_12"
SAVE_DIR = r"E:\gratuate_design\processed_pt"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_mat_universal(file_path):
    """通用读取：支持 v7.3 结构化复数合成"""
    try:
        mat = sio.loadmat(file_path)
        var_name = [k for k in mat.keys() if not k.startswith('__')][0]
        return mat[var_name].flatten()
    except (NotImplementedError, ValueError):
        with h5py.File(file_path, 'r') as f:
            var_name = list(f.keys())[0]
            data = f[var_name]
            # 修复 TypeError: 处理结构化 complex 数组
            if data.dtype.names is not None and 'real' in data.dtype.names:
                return (data['real'] + 1j * data['imag']).flatten()
            return np.array(data).flatten()

def process_signal(sig, seq_len=1024):
    """信号预处理：归一化与能量对齐"""
    sig = np.array(sig).astype(np.complex64)
    # 能量归一化
    energy = np.sqrt(np.mean(np.abs(sig)**2))
    sig = sig / (energy + 1e-12)
    # 滑动窗口对齐最强 1024 点
    if len(sig) > seq_len:
        sig_abs_sq = np.abs(sig)**2
        window_energy = np.convolve(sig_abs_sq, np.ones(seq_len), mode='valid')
        start_idx = np.argmax(window_energy)
        sig = sig[start_idx : start_idx + seq_len]
    else:
        sig = np.pad(sig, (0, seq_len - len(sig)))
    # 拆分 IQ 通道
    iq = np.stack([np.real(sig), np.imag(sig)], axis=0).astype(np.float32)
    return torch.from_numpy(iq)

def build_factory():
    # 建立 5 月 ID 映射表
    may_folders = sorted([d for d in os.listdir(MAY_DIR) if os.path.isdir(os.path.join(MAY_DIR, d))])
    id_to_idx = {mmsi: i for i, mmsi in enumerate(may_folders)}
    
    source_data, target_data = [], []
    print("🕒 正在处理 5 月数据 (源域)...")
    for mmsi, idx in id_to_idx.items():
        folder_path = os.path.join(MAY_DIR, mmsi)
        files = [f for f in os.listdir(folder_path) if f.endswith('.mat') 
                 and not ('_train' in f or '_test' in f)] # 过滤聚合文件
        for f in tqdm(files, desc=f"ID {mmsi}", leave=False):
            sig_raw = load_mat_universal(os.path.join(folder_path, f))
            source_data.append((process_signal(sig_raw), idx))
            
    print("\n🕒 正在处理 12 月数据 (目标域)...")
    dec_files = [f for f in os.listdir(DEC_DIR) if f.endswith('.mat')]
    for f in tqdm(dec_files):
        mmsi = f.split('_')[0]
        sig_raw = load_mat_universal(os.path.join(DEC_DIR, f))
        label = id_to_idx.get(mmsi, -1) # 老朋友保留标签，新面孔为 -1
        target_data.append((process_signal(sig_raw), label))
        
    print("💾 正在保存 .pt 文件...")
    torch.save({'data': source_data, 'mapping': id_to_idx}, os.path.join(SAVE_DIR, 'may_source.pt'))
    torch.save({'data': target_data}, os.path.join(SAVE_DIR, 'dec_target.pt'))
    print(f"✅ 成功！数据包已存至: {SAVE_DIR}")

if __name__ == "__main__":
    build_factory()