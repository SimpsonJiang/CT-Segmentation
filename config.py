"""
Configuration for 3D UNet training on CT segmentation
"""
import os
from pathlib import Path

# Paths - supporting both original and new data
DATA_DIR_ORIG = Path(__file__).parent.parent / "pre_ct"
LABEL_DIR_ORIG = Path(__file__).parent.parent / "pre_ct_seg"
DATA_DIR_NEW = Path(__file__).parent.parent / "pre_ct_1"
LABEL_DIR_NEW = Path(__file__).parent.parent / "pre_ct_seg_1"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs_v3.2"  # New output directory for v3 training

# Training IDs - use all available labeled data
# Original: 1-60, 77-117 (97 IDs with labels)
# New: 118-172 (53 IDs with labels, missing 134, 159, 167)
TRAIN_IDS_ORIG = list(range(1, 118))
TRAIN_IDS_NEW = list(range(118, 174))  # 118-173
TRAIN_IDS = TRAIN_IDS_ORIG + TRAIN_IDS_NEW

# Model
IN_CHANNELS = 1          # CT is single channel (grayscale)
OUT_CHANNELS = 3        # Multi-class segmentation (background + scapula + femur)
FEATURE_DEPTHS = [32, 64, 128, 256]  # Encoder depths
USE_ATTENTION = True    # Use attention gates
USE_RESIDUAL = True     # Use residual connections

# Training
BATCH_SIZE = 2           # Adjust based on GPU memory
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues

# Loss
LOSS_TYPE = "multi_dice_ce_boundary"    # multi_dice_ce_boundary, multi_dice_ce, dice_ce, dice, ce
BOUNDARY_LOSS_WEIGHT = 0.35  # Increased from 0.2 to better handle boundary confusion

# Preprocessing
TARGET_SPACING = [1.0, 1.0, 1.0]  # mm (D, H, W)
TARGET_D_SIZE = 392  # Cover all labels + 10 margin
HU_WINDOW_MIN = -300
HU_WINDOW_MAX = 1200
NORMALIZE_METHOD = "zscore"  # zscore or minmax

# Patch size for training (D, H, W)
PATCH_SIZE = (128, 128, 128)

# Mixed precision training
USE_MIXED_PRECISION = True

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 5
