import torch
import torch.nn as nn

class AIS_Ablation_Model(nn.Module):
    def __init__(self, use_resnet=True, use_transformer=True, num_classes=42, feat_dim=128):
        super(AIS_Ablation_Model, self).__init__()
        self.use_resnet = use_resnet
        self.use_transformer = use_transformer

        # --- 第一层：输入通道必须是 2 (对应 I 和 Q) ---
        if self.use_resnet:
            self.resnet_part = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3), # 这里必须是 2
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                # 后面接你的 ResNet 残差块
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
            current_channels = 128
        else:
            # 不用 ResNet 时，直接投影到 128 维度给 Transformer
            self.resnet_part = nn.Conv1d(2, 128, kernel_size=1) 
            current_channels = 128

        # --- 第二层：Transformer ---
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=current_channels, nhead=8)
            self.transformer_part = nn.TransformerEncoder(encoder_layer, num_layers=3)
        else:
            self.transformer_part = nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_feat = nn.Linear(current_channels, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: (batch, 2, 1024)
        x = self.resnet_part(x)
        if self.use_transformer:
            x = x.permute(2, 0, 1) # -> (L, N, E)
            x = self.transformer_part(x)
            x = x.permute(1, 2, 0) # -> (N, E, L)
        x = self.global_pool(x).flatten(1)
        feat = self.fc_feat(x)
        logits = self.classifier(feat)
        return feat, logits