import torch
import torch.nn as nn

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
        return nn.functional.relu(self.convs(x) + self.shortcut(x))

class ARPLFeatureExtractor(nn.Module):
    def __init__(self, feat_dim=128):
        super(ARPLFeatureExtractor, self).__init__()
        # 前端 ResNet 压缩
        self.frontend = nn.Sequential(
            ResidualBlock1D(2, 64, stride=2),    # -> 512
            ResidualBlock1D(64, 128, stride=2),  # -> 256
            ResidualBlock1D(128, feat_dim, stride=2) # -> 128
        )
        # 中端 Transformer 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, x):
        x = self.frontend(x)         # [B, 128, 128]
        x = x.permute(0, 2, 1)        # [B, 128, 128]
        x = self.transformer(x)
        feat = torch.mean(x, dim=1)  # 全局池化得到 128 维特征
        return feat