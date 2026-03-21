import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.convs(x) + self.shortcut(x))

class AIS_Ablation_Model(nn.Module):
    def __init__(self, use_resnet=True, use_transformer=True, num_classes=42, feat_dim=128):
        super(AIS_Ablation_Model, self).__init__()
        self.use_resnet = use_resnet
        self.use_transformer = use_transformer

        # --- 1. ResNet 前端 (提取局部特征并下采样) ---
        if self.use_resnet:
            self.frontend = nn.Sequential(
                ResidualBlock1D(2, 64, stride=2),    # 1024 -> 512
                ResidualBlock1D(64, 128, stride=2),  # 512 -> 256
                ResidualBlock1D(128, feat_dim, stride=2) # 256 -> 128
            )
            curr_dim = feat_dim
        else:
            # 不用 ResNet 时，直接投影通道，但不改变 1024 的长度
            self.frontend = nn.Conv1d(2, feat_dim, kernel_size=1)
            curr_dim = feat_dim

        # --- 2. Transformer 中端 ---
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=curr_dim, nhead=8, dim_feedforward=512, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        else:
            self.transformer = nn.Identity()

        self.fc = nn.Linear(curr_dim, feat_dim)

    def forward(self, x):
        x = self.frontend(x)         # [B, feat_dim, L']
        x = x.permute(0, 2, 1)        # [B, L', feat_dim]
        
        if self.use_transformer:
            x = self.transformer(x)
        
        feat = torch.mean(x, dim=1)  # 全局平均池化
        feat = self.fc(feat)
        return feat