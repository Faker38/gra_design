import torch
import torch.nn as nn
from torch.autograd import Function

# --- 1. 定义梯度反转层 (GRL) ---
# 它是 DANN 的灵魂：前向传播不变，反向传播时将梯度取反并乘以 alpha
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# --- 2. 升级后的模型 ---
class AIS_DANN_Model(nn.Module):
    def __init__(self, use_resnet=True, use_transformer=True, num_classes=32, feat_dim=128):
        super(AIS_DANN_Model, self).__init__()
        # 复用你最强的 ResNet + Transformer 架构
        from model import AIS_Ablation_Model 
        base_model = AIS_Ablation_Model(use_resnet, use_transformer, num_classes, feat_dim)
        
        self.frontend = base_model.frontend
        self.transformer = base_model.transformer
        self.feature_extractor = base_model.fc # 这里的输出就是我们要做领域不变处理的特征

        # --- 新增：领域判别器 (判断是 5月还是 12月) ---
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2) # 二分类：0 代表 5月，1 代表 12月
        )

    def forward(self, x, alpha=1.0):
        # 1. 提取指纹特征
        x = self.frontend(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        feat = torch.mean(x, dim=1)
        feat = self.feature_extractor(feat)

        # 2. 梯度反转：这是欺骗环境的关键
        reverse_feat = ReverseLayerF.apply(feat, alpha)
        domain_output = self.domain_classifier(reverse_feat)

        return feat, domain_output