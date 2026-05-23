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


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = AttentionResUNet3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_depths=FEATURE_DEPTHS,
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return pred_mask


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
    return pred_mask


def resample_to_shape(data, target_shape):
    """Resample 3D volume to target shape"""
    zoom_factors = np.array(target_shape) / np.array(data.shape)
    resampled = ndimage.zoom(data, zoom_factors, order=0)  # order=0 for nearest neighbor (binary mask)
    return resampled


def main():
    parser = argparse.ArgumentParser(description="Run inference on CT scans")
    parser.add_argument("--checkpoint", type=str, default="F:\\2.0\\outputs\\best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="F:\\2.0\\pre_ct\\3__Se3__Shoulder 1.0 I40s 1__00010870.nii.gz", help="Input CT file or directory")
    parser.add_argument("--output", type=str, default="F:\\2.0\\pre_ct_seg_pred", help="Output directory")
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

        # Load CT scan
        ct_nifti = nib.load(ct_path)

        # Preprocess (same as training) - returns processed data, original shape, original affine, and orientation transforms
        ct_data, original_shape, original_affine, original_ornt, ras_ornt = preprocess_ct(ct_nifti)
        print(f"  Original shape: {original_shape}, Preprocessed shape: {ct_data.shape}")

        # Predict
        if args.sliding_window:
            print(f"  Using sliding window inference with patch size {PATCH_SIZE}")
            pred_mask = predict_ct_sliding_window(model, ct_data, device, PATCH_SIZE, overlap=0.5)
        else:
            print(f"  Using full volume inference")
            pred_mask = predict_ct_full_volume(model, ct_data, device)

        print(f"  Predicted shape (preprocessed space): {pred_mask.shape}")

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
