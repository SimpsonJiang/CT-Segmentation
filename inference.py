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


def predict_ct(model, ct_path, device, threshold=0.5):
    """Predict segmentation for a single CT scan"""
    # Load CT scan
    ct_nifti = nib.load(ct_path)
    ct_data = ct_nifti.get_fdata().astype(np.float32)

    # Normalize CT to [0, 1]
    ct_min, ct_max = ct_data.min(), ct_data.max()
    if ct_max > ct_min:
        ct_data = (ct_data - ct_min) / (ct_max - ct_min)

    # Convert to tensor
    ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 1, D, H, W)

    # Predict
    with torch.no_grad():
        pred = model(ct_tensor)

    # Apply sigmoid to get probabilities, then threshold
    pred_prob = torch.sigmoid(pred)
    pred_mask = (pred_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)

    return pred_mask, ct_nifti.affine, ct_nifti.header


def main():
    parser = argparse.ArgumentParser(description="Run inference on CT scans")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input CT file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
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

        pred_mask, affine, header = predict_ct(model, ct_path, device, args.threshold)

        # Save prediction
        output_name = Path(ct_path).stem + "_pred.nii.gz"
        output_path = output_dir / output_name

        pred_nifti = nib.Nifti1Image(pred_mask, affine, header=header)
        nib.save(pred_nifti, output_path)

        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
