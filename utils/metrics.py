"""
Metrics for segmentation evaluation
"""
import torch
import numpy as np


def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score"""
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """Calculate Intersection over Union (IoU)"""
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def precision_score(pred, target, smooth=1e-6):
    """Calculate precision"""
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)

    return precision.item()


def recall_score(pred, target, smooth=1e-6):
    """Calculate recall"""
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()

    recall = (tp + smooth) / (tp + fn + smooth)

    return recall.item()


def f1_score(pred, target, smooth=1e-6):
    """Calculate F1 score"""
    precision = precision_score(pred, target, smooth)
    recall = recall_score(pred, target, smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return f1
