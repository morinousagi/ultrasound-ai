"""
Train an EfficientNet-B0 classifier on the TN5000 thyroid ultrasound dataset.

Features
--------
- Transfer learning using ImageNet pretrained weights
- Class-weighted CrossEntropyLoss
- AdamW optimizer
- ReduceLROnPlateau scheduler
- Early stopping based on validation F1-score
- Saves the best model checkpoint
- Records training history for plotting
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ThyroidDataset, build_transforms
from utils import ensure_dir, load_config, save_checkpoint, set_seed


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

IMAGE_ROOT = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

ensure_dir(CHECKPOINT_DIR)
ensure_dir(RESULTS_DIR)

# ============================================================
# Load configuration
# ============================================================

config = load_config(CONFIG_PATH)

set_seed(config["training"]["random_seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}\n")

# ============================================================
# Dataset
# ============================================================

train_dataset = ThyroidDataset(
    METADATA_CSV,
    IMAGE_ROOT,
    split="train",
    transform=build_transforms(CONFIG_PATH, train=True),
)

val_dataset = ThyroidDataset(
    METADATA_CSV,
    IMAGE_ROOT,
    split="val",
    transform=build_transforms(CONFIG_PATH, train=False),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["training"]["num_workers"],
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)

# ============================================================
# Compute class weights automatically
# mistakes on benign images incur a larger penalty,
# mistakes on malignant images incur a smaller penalty.
# ============================================================

train_df = pd.read_csv(METADATA_CSV)
train_df = train_df[train_df["split"] == "train"]

counts = train_df["label"].value_counts().sort_index()

# Computes class weights for loss function to compensate for dataset imbalance
weights = len(train_df) / (2 * counts.values)

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

print(f"Class weights: {weights}")

# ============================================================
# Model
# pretrained=True: timm auto downloads (once), loads ImageNet-pretrained EfficientNet-B0 weights
# ============================================================

model = timm.create_model(
    config["model"]["name"],
    pretrained=config["model"]["pretrained"],
    num_classes=2,      # Transfer Learning: replace final classifier layer to output 2 classes (original 1000 ImageNet classes)
)


# # Freeze all pretrained feature extraction layers.
# for param in model.parameters():
#     param.requires_grad = False

# # Train only the classification head.
# for param in model.classifier.parameters():
#     param.requires_grad = True


model.to(DEVICE)

# crossentropyloss performs logit normalization internally, no need softmax before loss fn
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(
    model.parameters(),    # model.classifier.parameters() if freezing
    lr=config["training"]["learning_rate"],
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# ============================================================
# Training
# ============================================================

history = []

best_f1 = -1

early_counter = 0

for epoch in range(config["training"]["epochs"]):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch+1}/{config['training']['epochs']}")
    print("=" * 60)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning Rate : {current_lr:.6f}")

    ##########################################################
    # Training
    ##########################################################

    model.train()

    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    ##########################################################
    # Validation
    ##########################################################

    model.eval()

    val_loss = 0

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in tqdm(val_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            prediction = outputs.argmax(dim=1)

            preds.extend(prediction.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)

    accuracy = accuracy_score(targets, preds)

    precision = precision_score(targets, preds)

    recall = recall_score(targets, preds)

    f1 = f1_score(targets, preds)

    scheduler.step(f1)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    if f1 > best_f1:

        best_f1 = f1

        early_counter = 0

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_f1,
            CHECKPOINT_DIR / "best_model.pth",
        )

        print("✓ Best model saved.")

    else:

        early_counter += 1

        if early_counter >= config["training"]["patience"]:

            print("\nEarly stopping triggered.")

            break

# ============================================================
# Save history
# ============================================================

history = pd.DataFrame(history)

history.to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False,
)

# ============================================================
# Plot loss
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(history["train_loss"], label="Train")

plt.plot(history["val_loss"], label="Validation")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.tight_layout()

plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=300)

plt.close()

# ============================================================
# Plot F1
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(history["f1"], marker="o")

plt.xlabel("Epoch")

plt.ylabel("Validation F1")

plt.title("Validation F1-score")

plt.tight_layout()

plt.savefig(RESULTS_DIR / "f1_curve.png", dpi=300)

plt.close()

print("\nTraining complete.")
print(f"Best validation F1: {best_f1:.4f}")