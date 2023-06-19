import torch.nn as nn
import torch

from pytorch_metric_learning.losses.subcenter_arcface_loss import ArcFaceLoss


class ArcFaceFocalLoss(ArcFaceLoss):
    def __init__(self, margin=0.5, scale=64, focal_gamma=2, **kwargs):
        super().__init__(margin=margin, scale=scale, num_classes=kwargs['num_classes'],
                         embedding_size=kwargs['num_features'])
        self.cross_entropy = FocalLoss(gamma=focal_gamma)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction is not 'none':
            return loss.mean()
        else:
            return loss

