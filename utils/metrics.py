"""
Metrics for segmentation evaluation
"""
import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion


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


def pixel_accuracy(pred, target):
    """Calculate pixel accuracy (binary case).

    Args:
        pred: (N,) tensor of predictions (0 or 1)
        target: (N,) tensor of ground truth (0 or 1)

    Returns:
        Accuracy (0 to 1)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    correct = (pred == target).sum().item()
    total = pred.numel()

    return correct / total


def mean_pixel_accuracy(pred, target, num_classes, smooth=1e-6):
    """
    Calculate mean pixel accuracy for multi-class segmentation.

    MPA = (1/C) * Σ(c) (TP_c / (TP_c + FP_c + TN_c + FN_c))

    Args:
        pred: (B, C, D, H, W) tensor of logits
        target: (B, D, H, W) tensor of class indices
        num_classes: number of classes
        smooth: smoothing factor (for numerical stability)

    Returns:
        Dictionary with per-class accuracy and mean accuracy
    """
    # Get predictions as class indices
    pred_indices = torch.argmax(pred, dim=1)  # (B, D, H, W)

    acc_per_class = {}
    for c in range(num_classes):
        pred_c = (pred_indices == c)
        target_c = (target == c)

        # True positives: pred_c=True and target_c=True
        tp = (pred_c & target_c).sum().item()
        # False positives: pred_c=True and target_c=False
        fp = (pred_c & ~target_c).sum().item()
        # True negatives: pred_c=False and target_c=False
        tn = (~pred_c & ~target_c).sum().item()
        # False negatives: pred_c=False and target_c=True
        fn = (~pred_c & target_c).sum().item()

        # Pixel accuracy for class c
        total = tp + fp + tn + fn
        if total > 0:
            acc_per_class[c] = (tp + tn) / total
        else:
            acc_per_class[c] = 0.0

    # Mean over foreground classes only (excluding background class 0)
    fg_classes = [c for c in acc_per_class.keys() if c != 'mean']
    acc_per_class['mean'] = sum(acc_per_class[c] for c in fg_classes) / len(fg_classes)
    return acc_per_class


def multi_class_dice_score(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Dice score for each class in multi-class segmentation.

    Args:
        pred: (B, C, D, H, W) tensor of logits
        target: (B, D, H, W) tensor of class indices
        num_classes: number of classes
        smooth: smoothing factor

    Returns:
        Dictionary with per-class dice and mean dice
    """
    # Get predictions as class indices
    pred_indices = torch.argmax(pred, dim=1)  # (B, D, H, W)

    dice_per_class = {}
    for c in range(num_classes):
        pred_c = (pred_indices == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_per_class[c] = dice.item()

    # Mean over foreground classes only (excluding background class 0)
    fg_classes = [c for c in dice_per_class.keys() if c != 'mean']
    dice_per_class['mean'] = sum(dice_per_class[c] for c in fg_classes) / len(fg_classes)
    return dice_per_class


def multi_class_iou_score(pred, target, num_classes, smooth=1e-6):
    """
    Calculate IoU score for each class in multi-class segmentation.
    """
    pred_indices = torch.argmax(pred, dim=1)

    iou_per_class = {}
    for c in range(num_classes):
        pred_c = (pred_indices == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class[c] = iou.item()

    # Mean over foreground classes only (excluding background class 0)
    fg_classes = [c for c in iou_per_class.keys() if c != 'mean']
    iou_per_class['mean'] = sum(iou_per_class[c] for c in fg_classes) / len(fg_classes)
    return iou_per_class


# ============ Surface-based Metrics ============

def get_surface_points(mask):
    """
    Extract surface points from a binary mask.

    A voxel is a surface point if it is foreground and has at least
    one background neighbor (6-connectivity in 3D).

    Args:
        mask: Binary mask (numpy array), shape (D, H, W)

    Returns:
        Array of surface point coordinates, shape (N, 3)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = (mask > 0).astype(np.uint8)

    # 6-connectivity kernel for surface detection in 3D
    kernel = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]])

    interior = binary_erosion(mask, structure=kernel)
    surface = mask & ~interior

    coords = np.where(surface > 0)
    return np.array(coords).T


def _compute_surface_distances(pred_surface, label_surface):
    """
    Compute bidirectional surface distances.

    Args:
        pred_surface: (N, 3) array of predicted surface points
        label_surface: (M, 3) array of ground truth surface points

    Returns:
        (N,) distances from pred to label, (M,) distances from label to pred
    """
    if len(pred_surface) == 0 or len(label_surface) == 0:
        return np.array([]), np.array([])

    tree = cKDTree(label_surface)
    dist_pred_to_label, _ = tree.query(pred_surface)

    tree = cKDTree(pred_surface)
    dist_label_to_pred, _ = tree.query(label_surface)

    return dist_pred_to_label, dist_label_to_pred


def hausdorff_distance_95(pred, label, voxel_spacing=None):
    """
    Compute the 95th percentile Hausdorff Distance (HD95).

    HD95 is a robust version of Hausdorff distance, less sensitive to outliers.

    Args:
        pred: Predicted binary mask (D, H, W)
        label: Ground truth binary mask (D, H, W)
        voxel_spacing: Spacing for each dimension [D, H, W] (if None, uses voxel units)

    Returns:
        HD95 value (float)
    """
    pred_surface = get_surface_points(pred)
    label_surface = get_surface_points(label)

    if len(pred_surface) == 0 and len(label_surface) == 0:
        return 0.0

    if voxel_spacing is not None:
        pred_surface = pred_surface * np.array(voxel_spacing)
        label_surface = label_surface * np.array(voxel_spacing)

    dist_pred_to_label, dist_label_to_pred = _compute_surface_distances(
        pred_surface, label_surface
    )

    hd95_p2l = np.percentile(dist_pred_to_label, 95) if len(dist_pred_to_label) > 0 else 0.0
    hd95_l2p = np.percentile(dist_label_to_pred, 95) if len(dist_label_to_pred) > 0 else 0.0

    return max(hd95_p2l, hd95_l2p)


def average_surface_distance(pred, label, voxel_spacing=None):
    """
    Compute the Average Surface Distance (ASD).

    Args:
        pred: Predicted binary mask (D, H, W)
        label: Ground truth binary mask (D, H, W)
        voxel_spacing: Spacing for each dimension [D, H, W]

    Returns:
        ASD value (float)
    """
    pred_surface = get_surface_points(pred)
    label_surface = get_surface_points(label)

    if len(pred_surface) == 0 and len(label_surface) == 0:
        return 0.0

    if voxel_spacing is not None:
        pred_surface = pred_surface * np.array(voxel_spacing)
        label_surface = label_surface * np.array(voxel_spacing)

    dist_pred_to_label, dist_label_to_pred = _compute_surface_distances(
        pred_surface, label_surface
    )

    total_points = len(dist_pred_to_label) + len(dist_label_to_pred)
    if total_points == 0:
        return 0.0

    combined_sum = np.sum(dist_pred_to_label) + np.sum(dist_label_to_pred)
    return combined_sum / total_points


def surface_dice(pred, label, threshold=1.0, voxel_spacing=None):
    """
    Compute Surface Dice coefficient.

    Measures overlap of surfaces within a tolerance distance.

    Args:
        pred: Predicted binary mask (D, H, W)
        label: Ground truth binary mask (D, H, W)
        threshold: Distance threshold in mm (or voxels if no spacing)
        voxel_spacing: Spacing for each dimension [D, H, W]

    Returns:
        Surface Dice value (0 to 1)
    """
    pred_surface = get_surface_points(pred)
    label_surface = get_surface_points(label)

    if len(pred_surface) == 0 and len(label_surface) == 0:
        return 1.0

    if len(pred_surface) == 0 or len(label_surface) == 0:
        return 0.0

    if voxel_spacing is not None:
        pred_surface = pred_surface * np.array(voxel_spacing)
        label_surface = label_surface * np.array(voxel_spacing)

    dist_pred_to_label, dist_label_to_pred = _compute_surface_distances(
        pred_surface, label_surface
    )

    pred_within = np.sum(dist_pred_to_label <= threshold)
    label_within = np.sum(dist_label_to_pred <= threshold)

    # Corrected Surface Dice formula: mean of bidirectional overlap within threshold
    # Value range: [0, 1]
    surface_dice = (pred_within + label_within) / (len(pred_surface) + len(label_surface))

    return surface_dice


def multi_class_surface_metrics(pred, target, num_classes, voxel_spacing=None, threshold=1.0):
    """
    Compute surface metrics for each foreground class.

    Args:
        pred: (B, C, D, H, W) logits
        target: (B, D, H, W) class indices
        num_classes: number of classes
        voxel_spacing: spacing for each dimension
        threshold: threshold for Surface Dice

    Returns:
        Dictionary with per-class and mean metrics
    """
    pred_indices = torch.argmax(pred, dim=1)

    results = {
        'hd95': {},
        'asd': {},
        'surface_dice': {}
    }

    # Only compute for foreground classes (1, 2, ...)
    for c in range(1, num_classes):
        pred_c = (pred_indices == c).cpu().numpy()
        target_c = (target == c).cpu().numpy()

        # Average over batch
        hd95_sum = 0
        asd_sum = 0
        sd_sum = 0
        count = 0

        for b in range(pred.shape[0]):
            if target_c[b].sum() > 0:  # Only if GT exists
                hd95_sum += hausdorff_distance_95(pred_c[b], target_c[b], voxel_spacing)
                asd_sum += average_surface_distance(pred_c[b], target_c[b], voxel_spacing)
                sd_sum += surface_dice(pred_c[b], target_c[b], threshold, voxel_spacing)
                count += 1

        if count > 0:
            results['hd95'][c] = hd95_sum / count
            results['asd'][c] = asd_sum / count
            results['surface_dice'][c] = sd_sum / count
        else:
            results['hd95'][c] = 0.0
            results['asd'][c] = 0.0
            results['surface_dice'][c] = 1.0 if pred_c.sum() == 0 else 0.0

    # Compute means (excluding class 0 which is background)
    for metric in ['hd95', 'asd', 'surface_dice']:
        values = [v for k, v in results[metric].items() if k > 0]
        results[f'{metric}_mean'] = sum(values) / len(values) if values else 0.0

    return results
