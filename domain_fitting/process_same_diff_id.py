import os
from collections import Counter

def audit_real_world_data(may_path, dec_path):
    print("🚀 开始扫描实测数据集（已加入聚合文件过滤机制）...")
    
    # --- 1. 审计 5 月数据 (按文件夹分类) ---
    # 逻辑：文件夹名即 ID；过滤掉 xxx_train.mat 和 xxx_test.mat
    may_id_counts = {}
    if os.path.exists(may_path):
        folders = [d for d in os.listdir(may_path) if os.path.isdir(os.path.join(may_path, d))]
        for folder in folders:
            folder_full_path = os.path.join(may_path, folder)
            # 关键过滤：排除包含 '_train' 或 '_test' 的聚合文件
            raw_mat_files = [
                f for f in os.listdir(folder_full_path) 
                if f.endswith('.mat') and not (f.lower().endswith('_train.mat') or f.lower().endswith('_test.mat'))
            ]
            if len(raw_mat_files) > 0:
                may_id_counts[folder] = len(raw_mat_files)
    
    # --- 2. 审计 12 月数据 (散乱文件名) ---
    # 逻辑：文件名第一段数字即 ID
    dec_id_list = []
    if os.path.exists(dec_path):
        files = [f for f in os.listdir(dec_path) if f.endswith('.mat')]
        for f in files:
            mmsi_id = f.split('_')[0] 
            dec_id_list.append(mmsi_id)
    
    dec_id_counts = Counter(dec_id_list)
    
    # --- 3. 计算重合与对比 ---
    may_set = set(may_id_counts.keys())
    dec_set = set(dec_id_counts.keys())
    overlap_ids = may_set.intersection(dec_set)
    dec_only = dec_set - may_set
    
    # --- 4. 打印报告 ---
    print("\n" + "="*45)
    print("📊 恒哥实测数据集审计报告 (Pro 版)")
    print("="*45)
    print(f"📅 [2025年05月] 总计 ID 数: {len(may_set)} (已剔除聚合文件)")
    print(f"📅 [2025年12月] 总计 ID 数: {len(dec_set)}")
    print("-" * 45)
    print(f"✅ 黄金重合 ID (用于领域自适应): {len(overlap_ids)} 个")
    print(f"⚠️ 12月新增 ID (用于开集识别测试): {len(dec_only)} 个")
    print("-" * 45)
    
    print("\n📌 典型重合 ID 样本量对比:")
    for i, mmsi in enumerate(list(overlap_ids)[:10]):
        print(f"ID: {mmsi:10} | 5月(原始样本): {may_id_counts[mmsi]:<4} | 12月: {dec_id_counts[mmsi]}")
    
    return list(overlap_ids), list(dec_only)

# --- 填入你的实际路径 ---
MAY_DIR = r"E:\gratuate_design\domian_data\2025_5\2025_5"
DEC_DIR = r"E:\gratuate_design\domian_data\2025_12\2025_12"

if __name__ == "__main__":
    overlap, newcomers = audit_real_world_data(MAY_DIR, DEC_DIR)