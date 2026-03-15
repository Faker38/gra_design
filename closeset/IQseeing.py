import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def signal_microscope(file_path):
    mat = sio.loadmat(file_path)
    var_name = [k for k in mat.keys() if not k.startswith('__')][0]
    matrix = mat[var_name]
    
    # 选第一个样本
    sample = matrix[0, :]
    I, Q = np.real(sample), np.imag(sample)
    
    # --- 显微镜 1：放大起始段 ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    # 只看前 100 个点，看看波形是怎么从 0 爬升的
    plt.plot(I[:100], 'b-o', label='I', markersize=3)
    plt.plot(Q[:100], 'r-o', label='Q', markersize=3)
    plt.title("Microscope: Rising Edge (First 100 pts)")
    plt.grid(True)

    # --- 显微镜 2：放大星座图的一个角 ---
    plt.subplot(1, 3, 2)
    # 能量归一化后，圆的半径应该在 1 附近
    energy = np.sqrt(np.mean(np.abs(sample)**2))
    I_n, Q_n = I/energy, Q/energy
    plt.plot(I_n, Q_n, 'g.', markersize=0.5, alpha=0.3)
    # 关键：我们强制放大坐标轴，看圆环的边缘
    plt.xlim(0.7, 1.1) 
    plt.ylim(0.0, 0.4)
    plt.title("Microscope: Constellation Edge")
    plt.grid(True)

    # --- 显微镜 3：幅度统计 (看它有多不稳) ---
    plt.subplot(1, 3, 3)
    amp = np.sqrt(I_n**2 + Q_n**2)
    plt.hist(amp, bins=50, color='purple', alpha=0.7)
    plt.title("Amplitude Distribution (Fingerprint density)")
    
    plt.tight_layout()
    plt.show()

    # 打印一些数值，证明不是空白
    print(f"信号前5个点数值: {sample[:5]}")
    print(f"信号幅度标准差 (代表噪声/指纹厚度): {np.std(amp):.6f}")

# 运行它
signal_microscope(r'1.mat')