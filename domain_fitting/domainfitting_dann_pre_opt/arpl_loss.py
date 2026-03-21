import torch
import torch.nn as nn
import torch.nn.functional as F

class ARPLLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(ARPLLoss, self).__init__()
        self.num_classes = num_classes
        # 互惠点：每个类在空间中都有一个“对立坐标”
        self.reciprocal_points = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.reciprocal_points)
        # 距离损失权重，建议设为 0.1
        self.weight_dist = 0.1 

    def forward(self, feat, labels):
        # 计算特征到所有互惠点的欧氏距离
        dist = torch.cdist(feat, self.reciprocal_points) # [Batch, 42]
        
        # 1. 分类损失 (Logits)
        logits = -dist 
        loss_ce = F.cross_entropy(logits, labels)
        
        # 2. 距离约束损失 (Dist Loss)
        # 提取当前样本与其所属正确类别的互惠点之间的距离
        dot_dist = dist.gather(1, labels.view(-1, 1))
        loss_dist = torch.mean(dot_dist)
        
        # 总损失：分类 + 紧凑度约束
        return loss_ce + self.weight_dist * loss_dist, logits