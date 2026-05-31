"""
Inference script for CT segmentation using trained model
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import *
from models import AttentionResUNet3D
from utils import multi_class_dice_score, multi_class_iou_score, mean_pixel_accuracy, multi_class_surface_metrics


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = AttentionResUNet3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_depths=FEATURE_DEPTHS,
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def resample_to_spacing(data, original_spacing, target_spacing):
    """Resample 3D volume to target spacing"""
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    resampled = ndimage.zoom(data, zoom_factors, order=1)
    return resampled


def preprocess_ct(ct_nifti):
    """Preprocess CT scan with same preprocessing as training"""
    # Keep original info for converting back to LPS later
    original_shape = ct_nifti.shape
    original_affine = ct_nifti.affine

    # Get original orientation (LPS)
    original_ornt = nib.io_orientation(original_affine)

    # 1. Orientation to RAS
    ct_ras = nib.as_closest_canonical(ct_nifti)
    ct_data = np.asarray(ct_ras.get_fdata(), dtype=np.float32)
    ras_spacing = ct_ras.header.get_zooms()

    # Get RAS orientation
    ras_ornt = nib.io_orientation(ct_ras.affine)

    # 2. Resample to target spacing
    ct_data = resample_to_spacing(ct_data, ras_spacing, TARGET_SPACING)

    # 3. Clip HU window
    ct_data = np.clip(ct_data, HU_WINDOW_MIN, HU_WINDOW_MAX)

    # 4. Z-score normalization
    if ct_data.std() > 0:
        ct_data = (ct_data - ct_data.mean()) / ct_data.std()
    else:
        ct_data = ct_data - ct_data.mean()

    return ct_data, original_shape, original_affine, original_ornt, ras_ornt


def predict_ct_sliding_window(model, ct_data, device, patch_size=PATCH_SIZE, overlap=0.5):
    """Predict using sliding window approach with reflection padding for edge coverage"""
    d, h, w = ct_data.shape
    patch_d, patch_h, patch_w = patch_size

    # Reflect pad edges to ensure full coverage of original volume
    pad_d = patch_d // 2
    pad_h = patch_h // 2
    pad_w = patch_w // 2

    ct_data_padded = np.pad(ct_data,
                             ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
                             mode='reflect')

    # Padded volume size
    d_padded, h_padded, w_padded = ct_data_padded.shape

    # Calculate stride (50% overlap by default)
    stride_d = int(patch_d * (1 - overlap))
    stride_h = int(patch_h * (1 - overlap))
    stride_w = int(patch_w * (1 - overlap))

    # Initialize output accumulator for multi-class (C, D, H, W) and count map
    num_classes = OUT_CHANNELS
    output = np.zeros((num_classes, d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)

    # Sliding window inference on padded volume
    with torch.no_grad():
        for start_d in range(0, d_padded - patch_d + 1, stride_d):
            for start_h in range(0, h_padded - patch_h + 1, stride_h):
                for start_w in range(0, w_padded - patch_w + 1, stride_w):
                    # Extract patch from padded volume
                    patch = ct_data_padded[start_d:start_d + patch_d,
                                           start_h:start_h + patch_h,
                                           start_w:start_w + patch_w]

                    # Convert to tensor (1, 1, D, H, W)
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    # Predict - get softmax probabilities for multi-class
                    pred = model(patch_tensor)
                    pred_prob = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy()  # (C, D, H, W)

                    # Map patch prediction back to original (non-padded) coordinates
                    out_start_d = max(0, start_d - pad_d)
                    out_start_h = max(0, start_h - pad_h)
                    out_start_w = max(0, start_w - pad_w)

                    out_end_d = min(d, start_d - pad_d + patch_d)
                    out_end_h = min(h, start_h - pad_h + patch_h)
                    out_end_w = min(w, start_w - pad_w + patch_w)

                    # Corresponding region within the patch
                    patch_out_start_d = out_start_d - (start_d - pad_d)
                    patch_out_start_h = out_start_h - (start_h - pad_h)
                    patch_out_start_w = out_start_w - (start_w - pad_w)

                    # Accumulate class-wise predictions
                    for c in range(num_classes):
                        output[c, out_start_d:out_end_d,
                               out_start_h:out_end_h,
                               out_start_w:out_end_w] += pred_prob[c,
                                                                     patch_out_start_d:patch_out_start_d + (out_end_d - out_start_d),
                                                                     patch_out_start_h:patch_out_start_h + (out_end_h - out_start_h),
                                                                     patch_out_start_w:patch_out_start_w + (out_end_w - out_start_w)]

                    count_map[out_start_d:out_end_d,
                              out_start_h:out_end_h,
                              out_start_w:out_end_w] += 1

    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
    for c in range(num_classes):
        output[c] = output[c] / count_map

    # Get final class prediction via argmax
    pred_mask = np.argmax(output, axis=0).astype(np.uint8)

    return pred_mask, output


def predict_ct_full_volume(model, ct_data, device):
    """Predict on full volume (may cause memory issues for large volumes)"""
    # Convert to tensor (1, 1, D, H, W)
    ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(ct_tensor)
        # Use softmax for multi-class
        pred_prob = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy()  # (C, D, H, W)

    # Get final class prediction via argmax
    pred_mask = np.argmax(pred_prob, axis=0).astype(np.uint8)
    return pred_mask, pred_prob


def resample_to_shape(data, target_shape):
    """Resample 3D volume to target shape"""
    zoom_factors = np.array(target_shape) / np.array(data.shape)
    resampled = ndimage.zoom(data, zoom_factors, order=0)  # order=0 for nearest neighbor (binary mask)
    return resampled


def remove_small_islands(mask, min_size=100):
    """Remove small isolated islands of each class using connected components"""
    cleaned = mask.copy()
    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # Skip background
        class_mask = (mask == class_id)
        labeled, num_features = ndimage.label(class_mask)
        for i in range(1, num_features + 1):
            component_size = (labeled == i).sum()
            if component_size < min_size:
                cleaned[labeled == i] = 0  # Set to background
    return cleaned


def enforce_class_exclusivity(mask, prob_output, min_confidence=0.3):
    """
    Clean up predictions where one class is embedded within another.
    If label A is surrounded entirely by label B, and the probability of B is high,
    change label A to B.
    """
    cleaned = mask.copy()
    num_classes = OUT_CHANNELS

    # For each class, check if it's surrounded by another class
    for target_class in range(1, num_classes):
        class_mask = (cleaned == target_class)
        if not class_mask.any():
            continue

        # Find connected components of this class
        labeled, num_features = ndimage.label(class_mask)

        for i in range(1, num_features + 1):
            component = (labeled == i)
            # Check if this component is surrounded by exactly one other class
            # Get neighbors (6-connectivity)
            footprint = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                  [[0,1,0],[1,1,1],[0,1,0]],
                                  [[0,0,0],[0,1,0],[0,0,0]]])
            boundary_mask = ndimage.binary_dilation(component, structure=footprint) & ~component

            # Find what classes are at the boundary
            boundary_classes = cleaned[boundary_mask]
            unique_boundary = np.unique(boundary_classes)

            # If surrounded by only one class (and that class is not background or itself)
            if len(unique_boundary) == 1 and unique_boundary[0] != 0 and unique_boundary[0] != target_class:
                surrounding_class = unique_boundary[0]
                # Check if the probability for surrounding class is high enough at these voxels
                component_coords = np.where(component)
                prob_at_component = prob_output[surrounding_class][component_coords[0], component_coords[1], component_coords[2]]
                if prob_at_component.mean() > min_confidence:
                    cleaned[component] = surrounding_class

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Run inference on CT scans")
    parser.add_argument("--checkpoint", type=str, default="F:\\2.0\\outputs_v3.2\\best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="F:\\2.0\\pre_ct\\117__Se301__0.625mm__43808702.nii.gz", help="Input CT file or directory")
    parser.add_argument("--output", type=str, default="F:\\2.0\\pre_ct_seg_pred_v3.2", help="Output directory")
    parser.add_argument("--sliding_window", action="store_true", default=True, help="Use sliding window inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    if Path(args.input).is_dir():
        ct_files = glob.glob(str(Path(args.input) / "*.nii.gz"))
    else:
        ct_files = [args.input]

    print(f"Found {len(ct_files)} CT file(s) to process")

    # Process each file
    for ct_path in ct_files:
        print(f"Processing: {ct_path}")

        # Extract ID from filename to find corresponding label
        ct_filename = Path(ct_path).stem
        try:
            ct_id = int(ct_filename.split('__')[0])
            # Search in both original and new label directories
            label_path = None
            for label_dir in [LABEL_DIR_ORIG, LABEL_DIR_NEW]:
                candidate = label_dir / f"{ct_id}_seg.nii.gz"
                if candidate.exists():
                    label_path = candidate
                    break
        except (ValueError, IndexError):
            label_path = None

        # Preprocess label first to get center for CT cropping
        label_data_for_ct_crop = None
        if label_path and label_path.exists():
            label_nifti = nib.load(str(label_path))
            label_ras = nib.as_closest_canonical(label_nifti)
            label_data_ras = np.asarray(label_ras.get_fdata(), dtype=np.float32)
            label_spacing = label_ras.header.get_zooms()

            # Resample label to target spacing
            label_data_ras = resample_to_spacing(label_data_ras, label_spacing, TARGET_SPACING)

            # Find label center
            label_slices = []
            for d_idx in range(label_data_ras.shape[0]):
                if label_data_ras[d_idx, :, :].max() > 0:
                    label_slices.append(d_idx)
            if label_slices:
                label_center_d = (min(label_slices) + max(label_slices)) // 2
            else:
                label_center_d = label_data_ras.shape[0] // 2

            # Crop label to TARGET_D_SIZE
            target_d = TARGET_D_SIZE
            current_d = label_data_ras.shape[0]
            if current_d >= target_d:
                start = max(0, label_center_d - target_d // 2)
                if start + target_d > current_d:
                    start = current_d - target_d
                label_data_ras = label_data_ras[start:start + target_d]
            else:
                pad_before = (target_d - current_d) // 2
                pad_after = target_d - current_d - pad_before
                label_data_ras = np.pad(label_data_ras, [(pad_before, pad_after), (0, 0), (0, 0)], mode='constant', constant_values=0)

            label_data_for_ct_crop = label_data_ras

        # Load and preprocess CT
        ct_nifti = nib.load(ct_path)
        ct_data, original_shape, original_affine, original_ornt, ras_ornt = preprocess_ct(ct_nifti)
        print(f"  Original shape: {original_shape}, Preprocessed shape: {ct_data.shape}")

        # Predict (returns argmaxed mask for multi-class)
        if args.sliding_window:
            print(f"  Using sliding window inference with patch size {PATCH_SIZE}")
            pred_mask, pred_prob = predict_ct_sliding_window(model, ct_data, device, PATCH_SIZE, overlap=0.5)
        else:
            print(f"  Using full volume inference")
            pred_mask, pred_prob = predict_ct_full_volume(model, ct_data, device)

        print(f"  Predicted shape (preprocessed space): {pred_mask.shape}")

        # Post-processing: remove small isolated islands
        pred_mask = remove_small_islands(pred_mask, min_size=500)
        print(f"  After removing small islands: {(pred_mask > 0).sum()} voxels")

        # Post-processing: enforce class exclusivity (fix boundary confusion)
        pred_mask = enforce_class_exclusivity(pred_mask, pred_prob, min_confidence=0.5)

        print(f"  Predicted shape (preprocessed space): {pred_mask.shape}")

        # Compute metrics if ground truth label exists
        if label_path and label_path.exists() and label_data_for_ct_crop is not None:
            # Use preprocessed label (same shape as CT after preprocessing)
            label_binary = (label_data_for_ct_crop > 0).astype(np.float32)

            # Clip to same size if needed
            if label_binary.shape != pred_mask.shape:
                min_d = min(label_binary.shape[0], pred_mask.shape[0])
                min_h = min(label_binary.shape[1], pred_mask.shape[1])
                min_w = min(label_binary.shape[2], pred_mask.shape[2])
                label_binary = label_binary[:min_d, :min_h, :min_w]
                pred_mask_crop = pred_mask[:min_d, :min_h, :min_w]
            else:
                pred_mask_crop = pred_mask

            # Convert to tensors for metrics
            # pred for metrics: (1, C, D, H, W), target: (1, D, H, W)
            pred_tensor = torch.from_numpy(pred_mask_crop[np.newaxis]).unsqueeze(0).long()
            label_tensor = torch.from_numpy(label_binary).unsqueeze(0).long()

            # Compute metrics
            dice_result = multi_class_dice_score(pred_tensor.float(), label_tensor, OUT_CHANNELS)
            iou_result = multi_class_iou_score(pred_tensor.float(), label_tensor, OUT_CHANNELS)
            mpa_result = mean_pixel_accuracy(pred_tensor.float(), label_tensor, OUT_CHANNELS)
            surface_metrics = multi_class_surface_metrics(pred_tensor.float(), label_tensor, OUT_CHANNELS, voxel_spacing=TARGET_SPACING, threshold=1.0)

            print(f"  Metrics:")
            print(f"    Dice: {dice_result['mean']:.4f} (class0: {dice_result.get(0, 0):.4f}, class1: {dice_result.get(1, 0):.4f}, class2: {dice_result.get(2, 0):.4f})")
            print(f"    IoU: {iou_result['mean']:.4f}")
            print(f"    MPA: {mpa_result['mean']:.4f}")
            print(f"    HD95: {surface_metrics['hd95_mean']:.4f}, ASD: {surface_metrics['asd_mean']:.4f}, Surface Dice: {surface_metrics['surface_dice_mean']:.4f}")

        # Resample prediction back to original input dimensions
        pred_mask_original = resample_to_shape(pred_mask, original_shape)
        print(f"  Resampled to original shape: {pred_mask_original.shape}")

        # Convert from RAS back to LPS orientation
        # ras_ornt represents RAS, original_ornt represents LPS
        # To go from RAS to LPS, we need the inverse of (ras_ornt - original_ornt)
        ras_to_lps = nib.orientations.ornt_transform(ras_ornt, original_ornt)
        pred_mask_lps = nib.orientations.apply_orientation(pred_mask_original, ras_to_lps)
        print(f"  Converted to LPS orientation: {pred_mask_lps.shape}")

        # Save prediction with original LPS affine
        output_name = Path(ct_path).stem + "_pred.nii.gz"
        output_path = output_dir / output_name

        pred_nifti = nib.Nifti1Image(pred_mask_lps, original_affine)
        nib.save(pred_nifti, output_path)

        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
