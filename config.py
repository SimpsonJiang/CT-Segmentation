"""
Configuration for 3D UNet training on CT segmentation
"""
import os
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "pre_ct"
LABEL_DIR = Path(__file__).parent.parent / "pre_ct_seg"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs_v2"

# Training IDs (already annotated: 77-117)
TRAIN_IDS = list(range(1, 118))  # 77 to 117 inclusive

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
