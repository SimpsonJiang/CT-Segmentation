"""
Training script for 3D UNet on CT segmentation
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import *
from models import AttentionResUNet3D
from data import CTSegmentationDataset, collate_fn
from utils import get_loss_fn, dice_score, iou_score, f1_score


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for ct, label in pbar:
        ct = ct.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        if USE_MIXED_PRECISION:
            with autocast('cuda'):
                pred = model(ct)
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(ct)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_dice += dice_score(torch.sigmoid(pred), label)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice_score(torch.sigmoid(pred), label):.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for ct, label in pbar:
            ct = ct.to(device)
            label = label.to(device)

            pred = model(ct)
            loss = criterion(pred, label)
            pred_sig = torch.sigmoid(pred)

            running_loss += loss.item()
            running_dice += dice_score(pred_sig, label)
            running_iou += iou_score(pred_sig, label)
            running_f1 += f1_score(pred_sig, label)

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_f1 = running_f1 / len(dataloader)

    return epoch_loss, epoch_dice, epoch_iou, epoch_f1


def main():
    parser = argparse.ArgumentParser(description="Train 3D UNet for CT segmentation")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    print(f"Loading dataset with IDs: {TRAIN_IDS}")
    full_dataset = CTSegmentationDataset(
        data_dir=DATA_DIR,
        label_dir=LABEL_DIR,
        ids=TRAIN_IDS,
        target_spacing=TARGET_SPACING,
        target_d_size=TARGET_D_SIZE,
        hu_window=(HU_WINDOW_MIN, HU_WINDOW_MAX),
        patch_size=PATCH_SIZE,
        augment=True,
    )

    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = AttentionResUNet3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_depths=FEATURE_DEPTHS,
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = get_loss_fn(LOSS_TYPE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Mixed precision scaler
    scaler = GradScaler('cuda') if USE_MIXED_PRECISION else None

    # Training loop
    best_dice = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_dice, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, device, epoch)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"           Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                output_dir / "best_model.pth",
            )
            print(f"  -> Saved best model with Dice: {best_dice:.4f}")

        # Save periodic checkpoint
        if epoch % SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pth",
            )

    print(f"\nTraining completed! Best Dice: {best_dice:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
