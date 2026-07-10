import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMAGE_HEIGHT, IMAGE_WIDTH, C, LR, EPOCHS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATASET_ROOT, CHECKPOINT_DIR, LOG_DIR
)
from dataset import YOLOv1Dataset
from yolov1.model import YOLOv1
from yolov1.loss import YOLOv1Loss
from yolov1.utils import load_class_names


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    """Runs a single training epoch and returns the average loss."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    # Iterate over the tqdm progress bar wrapper to enable progress logging
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Computes validation loss over the specified data loader split."""
    model.eval()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item()
        
    return total_loss / len(loader)


def main(args):
    """Executes the full training and validation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names
    data_yaml_path = os.path.join(DATASET_ROOT, "data.yaml")

    class_names = load_class_names(data_yaml_path)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    if num_classes != C:
        print(f"Warning: config has C={C} but dataset has {num_classes} classes")

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.90, 1.10), rotate=5, border_mode=0, p=0.4),
        A.RandomSizedBBoxSafeCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, erosion_rate=0.0, p=0.3),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
        clip=True
    ))

    valid_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        clip=True
    ))

    # Instantiate datasets and dataloaders
    train_set = YOLOv1Dataset(DATASET_ROOT, "train", transforms=train_transforms)

    valid_set = YOLOv1Dataset(DATASET_ROOT, "valid", transforms=valid_transforms)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    valid_loader = DataLoader(
        valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Initialize model, loss, and optimizer
    model = YOLOv1().to(device)

    loss_fn = YOLOv1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    best_valid_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")
    elif args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    writer = SummaryWriter(log_dir=LOG_DIR)
    # Main training loop
    for epoch in range(start_epoch, EPOCHS):
        # Stage 2: Fine-tune only the last residual block of the ResNet backbone
        if epoch == 15:
            print("\n=== Stage 2: Unfreezing backbone for fine-tuning ===")
            model.freeze_backbone(False) 
            optimizer = torch.optim.Adam([
                {"params": model.backbone[-1].parameters(), "lr": 1e-6}, 
                {"params": model.head.parameters(), "lr": LR}
            ])

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        valid_loss = validate(model, valid_loader, loss_fn, device)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", valid_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:2d}/{EPOCHS}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {valid_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save latest checkpoint state for recovery
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
        }
        save_checkpoint(checkpoint_state, os.path.join(CHECKPOINT_DIR, "latest.pth"))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"  → New best model! (valid_loss: {valid_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pth"))
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights to initialize (ignored if --resume passed)")
    args = parser.parse_args()
    main(args)