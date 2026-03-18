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

    def forward(self, feat, labels):
        # 计算特征到所有互惠点的欧氏距离
        # dist shape: [Batch, Num_Classes]
        dist = torch.cdist(feat, self.reciprocal_points)
        
        # ARPL 核心：最小化样本与其对应类互惠点之间的负距离（即推开它们）
        # 这里使用负距离作为 Logits
        logits = -dist 
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits