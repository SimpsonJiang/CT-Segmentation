"""
Loss functions for segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class CombinedDiceBCELoss(nn.Module):
    """Combined Dice and BCE loss (uses BCEWithLogitsLoss for autocast compatibility)"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


def get_loss_fn(loss_type="dice_ce"):
    """Get loss function by name"""
    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "ce" or loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "dice_ce":
        return CombinedDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
