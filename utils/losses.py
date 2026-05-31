"""
Loss functions for segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation with class weights"""

    def __init__(self, smooth=1e-6, class_weights=None):
        """
        class_weights: list of weights for each class, e.g. [1.0, 1.5, 1.0]
                       gives more weight to class 1 (Femur)
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights if class_weights else [1.0] * 3

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W) logits
        target: (B, D, H, W) long tensor with class indices [0, 1, 2, ...]
        """
        # Get probabilities via softmax
        pred = F.softmax(pred, dim=1)  # (B, C, D, H, W)

        num_classes = pred.shape[1]
        total_weight = 0.0
        weighted_dice = 0.0

        # Compute weighted Dice for each class (including background as class 0)
        for c in range(num_classes):
            weight = self.class_weights[c] if c < len(self.class_weights) else 1.0
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = (target == c).float().contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )
            weighted_dice += weight * dice
            total_weight += weight

        return 1 - (weighted_dice / total_weight)


class BoundaryLoss(nn.Module):
    """Boundary loss that penalizes distance between predicted and GT surfaces"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W) logits
        target: (B, D, H, W) long tensor
        """
        # Get prediction (B, D, H, W) as binary mask for foreground classes
        # Use argmax to get class predictions
        pred_mask = torch.argmax(pred, dim=1)  # (B, D, H, W)

        total_loss = 0.0
        batch_size = pred.shape[0]

        for b in range(batch_size):
            pred_b = pred_mask[b].cpu().numpy()
            target_b = target[b].cpu().numpy()

            # For each foreground class (1 and 2), compute boundary loss
            for class_id in [1, 2]:
                pred_class = (pred_b == class_id).astype(np.float32)
                target_class = (target_b == class_id).astype(np.float32)

                if target_class.sum() == 0:
                    continue  # Skip if no GT for this class

                # Compute distance transforms
                # pred_dist: distance from GT surface to pred surface
                # target_dist: distance from pred surface to GT surface
                if pred_class.sum() > 0:
                    pred_dist = distance_transform_edt(1 - pred_class)
                    target_dist = distance_transform_edt(1 - target_class)

                    # Boundary loss: mean of distances at surfaces
                    # Distance from GT surface to nearest pred voxel
                    gt_surface = target_class - distance_transform_edt(target_class).clip(0, 1)
                    gt_surface = gt_surface.clip(0, 1)
                    loss_gt_to_pred = (gt_surface * pred_dist).sum() / (gt_surface.sum() + 1e-6)

                    # Distance from pred surface to nearest GT voxel
                    pred_surface = pred_class - distance_transform_edt(pred_class).clip(0, 1)
                    pred_surface = pred_surface.clip(0, 1)
                    loss_pred_to_gt = (pred_surface * target_dist).sum() / (pred_surface.sum() + 1e-6)

                    total_loss += (loss_gt_to_pred + loss_pred_to_gt) / 2

        return total_loss / max(batch_size, 1)


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


class MultiClassDiceCELoss(nn.Module):
    """Combined Multi-class Dice and CrossEntropy loss"""

    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = MultiClassDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class MultiClassDiceCEBoundaryLoss(nn.Module):
    """Combined Multi-class Dice, CrossEntropy and Boundary loss"""

    def __init__(self, dice_weight=0.4, ce_weight=0.4, boundary_weight=0.2, femur_weight=1.5, scapula_weight=1.0):
        """
        femur_weight: relative weight for Femur (label=1), e.g. 1.5 means Femur has 1.5x weight
        scapula_weight: relative weight for Scapula (label=2), e.g. 1.0 means Scapula has 1.0x weight
        Background (label=0) weight is 1.0
        Final weights are normalized so that all class weights sum to 3.0
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight

        # Normalize class weights so total = 3.0 (for 3 classes)
        total = 1.0 + femur_weight + scapula_weight  # background + femur + scapula
        class_weights = [
            1.0 / total * 3.0,  # background
            femur_weight / total * 3.0,  # femur
            scapula_weight / total * 3.0   # scapula
        ]

        self.dice_loss = MultiClassDiceLoss(class_weights=class_weights)
        self.ce_loss = nn.CrossEntropyLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce + self.boundary_weight * boundary


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
    elif loss_type == "multi_dice_ce":
        return MultiClassDiceCELoss(dice_weight=0.5, ce_weight=0.5)
    elif loss_type == "multi_dice_ce_boundary":
        # femur_weight=1.5 means Femur gets 1.5x weight, Scapula gets 1.0x weight
        return MultiClassDiceCEBoundaryLoss(dice_weight=0.35, ce_weight=0.35, boundary_weight=0.30, femur_weight=1.5, scapula_weight=1.0)
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
