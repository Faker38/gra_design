import torch
import torch.nn as nn

# 1. 残差块：负责在压缩长度的同时防止指纹特征丢失
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
        # 如果输入输出维度不一致，用 1x1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return nn.functional.relu(self.convs(x) + self.shortcut(x))

# 2. 总模型：ResNet + Transformer
class AISTransformerModel(nn.Module):
    def __init__(self, num_classes, feature_dim=128):
        super(AISTransformerModel, self).__init__()
        
        # --- 前端：1D-ResNet 压缩网络 ---
        # 输入形状: (Batch, 2, 1024)
        self.frontend = nn.Sequential(
            ResidualBlock1D(2, 64, stride=2),    # -> (64, 512)
            ResidualBlock1D(64, 128, stride=2),  # -> (128, 256)
            ResidualBlock1D(128, feature_dim, stride=2) # -> (128, 128)
        )
        
        # --- 中端：Transformer Encoder ---
        # Transformer 需要的输入形状是 (Batch, Seq_Len, Embedding_Dim)
        # 此时我们的 Seq_Len = 128, Embedding_Dim = 128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=8, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # --- 后端：分类器 ---
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. 进入 CNN 提取局部特征并压缩长度
        x = self.frontend(x)  # [B, 128, 128]
        
        # 2. 变换维度以适配 Transformer
        x = x.permute(0, 2, 1) # [B, 128, 128] -> [Batch, Seq, Feature]
        
        # 3. Transformer 全局建模
        x = self.transformer(x)
        
        # 4. 全局平均池化 (取序列的平均值作为整段信号的指纹特征)
        x = torch.mean(x, dim=1)
        
        # 5. 分类
        logits = self.classifier(x)
        return logits
if __name__ == "__main__":
    # 模拟你的数据情况：52个ID，Batch大小8，长度1024
    num_classes = 52
    model = AISTransformerModel(num_classes=num_classes).cuda()
    
    # 构造一个模拟输入
    test_input = torch.randn(8, 2, 1024).cuda()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 前向传播测试
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输出形状: {output.shape}") # 预期应为 [8, 52]
    if output.shape == (8, 52):
        print("--- 模型验证成功！显存完全能够承受 ---")