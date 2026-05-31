"""
Dataset for CT segmentation with preprocessing
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy import ndimage
from pathlib import Path


class CTSegmentationDataset(Dataset):
    """
    Dataset for loading CT scans and corresponding segmentation masks with preprocessing
    """

    def __init__(self, data_dir, label_dir, ids, target_spacing=(1.0, 1.0, 1.0),
                 target_d_size=392, hu_window=(-300, 1200), patch_size=(128, 128, 128),
                 augment=False, extra_data_dirs=None, extra_label_dirs=None):
        """
        Args:
            data_dir: Primary path to CT scans
            label_dir: Primary path to segmentation labels
            ids: List of IDs to use for training
            target_spacing: Target spacing (D, H, W) in mm
            target_d_size: Target D dimension size
            hu_window: HU window range (min, max)
            patch_size: Size of random patches to extract (D, H, W)
            augment: Whether to apply data augmentation
            extra_data_dirs: Additional CT data directories to search
            extra_label_dirs: Additional label directories to search
        """
        self.data_dirs = [Path(data_dir)]
        self.label_dirs = [Path(label_dir)]
        if extra_data_dirs:
            self.data_dirs.extend([Path(d) for d in extra_data_dirs])
        if extra_label_dirs:
            self.label_dirs.extend([Path(d) for d in extra_label_dirs])
        self.ids = ids
        self.target_spacing = target_spacing
        self.target_d_size = target_d_size
        self.hu_window = hu_window
        self.patch_size = patch_size
        self.augment = augment

        self.file_pairs = []
        skipped = 0
        for id_ in ids:
            ct_file = self._find_ct_file(id_)
            label_file = self._find_label_file(id_)

            if ct_file and label_file:
                # Check if H and W are both 128 after preprocessing
                if self._check_hw_size(ct_file, label_file):
                    self.file_pairs.append((ct_file, label_file))
                else:
                    skipped += 1

        print(f"Found {len(self.file_pairs)} file pairs for training (skipped {skipped} files with H or W != 128)")

    def _find_ct_file(self, id_):
        """Find CT file by ID in multiple directories"""
        # Search in data_dirs list
        for data_dir in self.data_dirs:
            pattern = str(data_dir / f"{id_}__*.nii.gz")
            files = glob.glob(pattern)
            if files:
                return files[0]
        return None

    def _find_label_file(self, id_):
        """Find label file by ID in multiple directories"""
        # Search in label_dirs list
        for label_dir in self.label_dirs:
            # Try exact match first
            label_file = label_dir / f"{id_}_seg.nii.gz"
            if label_file.exists():
                return str(label_file)

            # Try versioned files
            for version in ["", "v2", "v3"]:
                label_file = label_dir / f"{id_}_seg{version}.nii.gz"
                if label_file.exists():
                    return str(label_file)
        return None

    def _check_hw_size(self, ct_file, label_file):
        """Check if W is at least 128 after preprocessing (H is always >= 128 based on data analysis)"""
        try:
            ct_nifti = nib.load(ct_file)
            label_nifti = nib.load(label_file)

            # Preprocess CT
            ct_ras = nib.as_closest_canonical(ct_nifti)
            ct_data = np.asarray(ct_ras.get_fdata(), dtype=np.float32)
            ct_data = self._resample_to_spacing(ct_data, ct_ras.header.get_zooms(), self.target_spacing)

            # Preprocess label
            label_ras = nib.as_closest_canonical(label_nifti)
            label_data = np.asarray(label_ras.get_fdata(), dtype=np.float32)
            label_data = self._resample_to_spacing(label_data, label_ras.header.get_zooms(), self.target_spacing)

            # Crop D
            label_center_d = self._find_label_center_d(label_data)
            ct_data = self._crop_d_centered(ct_data, label_center_d, self.target_d_size)
            label_data = self._crop_d_centered(label_data, label_center_d, self.target_d_size)

            _, h, w = ct_data.shape
            # Only need to check W >= 128, H is always >= 128 based on data analysis
            return w >= 128
        except Exception:
            return False

    def _resample_to_spacing(self, data, original_spacing, target_spacing):
        """Resample 3D volume to target spacing"""
        zoom_factors = np.array(original_spacing) / np.array(target_spacing)
        resampled = ndimage.zoom(data, zoom_factors, order=1)
        return resampled

    def _crop_or_pad_d(self, data, target_d):
        """Crop or pad D dimension to target size, center-aligned"""
        current_d = data.shape[0]
        if current_d <= target_d:
            # Pad
            pad_before = (target_d - current_d) // 2
            pad_after = target_d - current_d - pad_before
            pad_width = [(pad_before, pad_after), (0, 0), (0, 0)]
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        else:
            # Crop center
            start = (current_d - target_d) // 2
            data = data[start:start + target_d]
        return data

    def _find_label_center_d(self, label_data):
        """Find center D of label region"""
        # Find slices with non-zero values
        label_slices = []
        for d_idx in range(label_data.shape[0]):
            if label_data[d_idx, :, :].max() > 0:
                label_slices.append(d_idx)
        if label_slices:
            return (min(label_slices) + max(label_slices)) // 2
        return label_data.shape[0] // 2

    def _crop_d_centered(self, data, center_d, target_d):
        """Crop or pad D dimension centered at center_d"""
        current_d = data.shape[0]

        if current_d >= target_d:
            # Crop: center at center_d
            half = target_d // 2
            start = max(0, center_d - half)
            end = start + target_d

            # Adjust if end exceeds data shape
            if end > current_d:
                end = current_d
                start = max(0, end - target_d)

            return data[start:end]
        else:
            # Pad: pad to target_d, centered
            total_pad = target_d - current_d
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before

            pad_width = [(pad_before, pad_after), (0, 0), (0, 0)]
            return np.pad(data, pad_width, mode='constant', constant_values=0)

    def _extract_random_patch(self, ct_data, label_data):
        """Extract patch of fixed size from CT and label.
        50% chance to sample centered on label region, 50% random.
        All files have H >= 128 and W >= 128 after filtering at load time.
        """
        d, h, w = ct_data.shape
        patch_d, patch_h, patch_w = self.patch_size

        # Find label center for "centered" sampling
        label_slices = []
        for d_idx in range(d):
            if label_data[d_idx, :, :].max() > 0:
                label_slices.append(d_idx)

        # 50% chance: sample centered on label region
        if label_slices and np.random.rand() < 0.5:
            label_center_d = (min(label_slices) + max(label_slices)) // 2
            start_d = max(0, min(label_center_d - patch_d // 2 + np.random.randint(-10, 10), d - patch_d))
        else:
            start_d = np.random.randint(0, max(1, d - patch_d + 1)) if d > patch_d else 0

        # H and W are always >= 128, so we center the crop at the middle
        start_h = (h - patch_h) // 2
        start_w = (w - patch_w) // 2

        # Extract patch
        ct_patch = ct_data[start_d:start_d + patch_d,
                          start_h:start_h + patch_h,
                          start_w:start_w + patch_w]
        label_patch = label_data[start_d:start_d + patch_d,
                                start_h:start_h + patch_h,
                                start_w:start_w + patch_w]

        return ct_patch, label_patch

    def _preprocess_ct(self, ct_nifti, label_data):
        """Preprocess CT scan: orient, resample, clip HU, crop D, normalize"""
        # 1. Orientation to RAS
        ct_ras = nib.as_closest_canonical(ct_nifti)
        ct_data = np.asarray(ct_ras.get_fdata(), dtype=np.float32)
        original_spacing = ct_ras.header.get_zooms()

        # 2. Resample to target spacing
        ct_data = self._resample_to_spacing(ct_data, original_spacing, self.target_spacing)

        # 3. Find label center for cropping
        if isinstance(label_data, nib.Nifti1Image):
            label_data_for_center = np.asarray(label_data.get_fdata(), dtype=np.float32)
        else:
            label_data_for_center = label_data

        # Resample label to same space for finding center
        label_spacing = label_data.header.get_zooms() if isinstance(label_data, nib.Nifti1Image) else original_spacing
        label_resampled = self._resample_to_spacing(label_data_for_center, label_spacing, self.target_spacing)
        label_center_d_resampled = self._find_label_center_d(label_resampled)

        # 4. Crop D dimension centered at label
        ct_data = self._crop_d_centered(ct_data, label_center_d_resampled, self.target_d_size)

        # 5. Clip HU window
        ct_data = np.clip(ct_data, self.hu_window[0], self.hu_window[1])

        # 6. Z-score normalization
        if ct_data.std() > 0:
            ct_data = (ct_data - ct_data.mean()) / ct_data.std()
        else:
            ct_data = ct_data - ct_data.mean()

        return ct_data

    def _preprocess_label(self, label_nifti):
        """Preprocess label: orient, resample, crop D"""
        # 1. Orientation to RAS
        label_ras = nib.as_closest_canonical(label_nifti)
        label_data = np.asarray(label_ras.get_fdata(), dtype=np.float32)
        original_spacing = label_ras.header.get_zooms()

        # 2. Resample to target spacing
        label_data = self._resample_to_spacing(label_data, original_spacing, self.target_spacing)

        # 3. Find label center
        label_center_d = self._find_label_center_d(label_data)

        # 4. Crop D dimension centered at label
        label_data = self._crop_d_centered(label_data, label_center_d, self.target_d_size)

        # Keep multi-class labels (0=background, 1=scapula, 2=femur)
        # No binarization needed
        return label_data

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """Get item from dataset"""
        ct_path, label_path = self.file_pairs[idx]

        # Load CT scan and label
        ct_nifti = nib.load(ct_path)
        label_nifti = nib.load(label_path)

        # Preprocess
        ct_data = self._preprocess_ct(ct_nifti, label_nifti)
        label_data = self._preprocess_label(label_nifti)

        # Extract random patch
        ct_data, label_data = self._extract_random_patch(ct_data, label_data)

        # Convert to tensor (C, D, H, W) for CT, (D, H, W) for label
        ct_tensor = torch.from_numpy(ct_data).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label_data).long()

        return ct_tensor, label_tensor


def collate_fn(batch):
    """Custom collate function for 3D volumes"""
    ct_tensors = []
    label_tensors = []

    for ct, label in batch:
        ct_tensors.append(ct)
        label_tensors.append(label)

    return torch.stack(ct_tensors, dim=0), torch.stack(label_tensors, dim=0)


def get_dataset(data_dir, label_dir, ids, config, augment=False):
    """Factory function to create dataset with config"""
    return CTSegmentationDataset(
        data_dir=data_dir,
        label_dir=label_dir,
        ids=ids,
        target_spacing=tuple(config.get('TARGET_SPACING', [1.0, 1.0, 1.0])),
        target_d_size=config.get('TARGET_D_SIZE', 392),
        hu_window=(config.get('HU_WINDOW_MIN', -300), config.get('HU_WINDOW_MAX', 1200)),
        augment=augment
    )
