"""
Generate Grad-CAM visualizations for representative test images.
1. Loads the trained EfficientNet-B0 model.
2. Reads predictions.csv produced by evaluate.py.
3. Automatically selects representative examples.
4. Computes Grad-CAM heatmaps.
5. Saves overlay visualizations.

Outputs: results/gradcam/
        correct_benign.png
        correct_malignant.png
        false_positive.png
        false_negative.png
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import build_transforms
from utils import load_config


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

PREDICTION_CSV = PROJECT_ROOT / "results" / "predictions.csv"

IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "cropped"

OUTPUT_DIR = PROJECT_ROOT / "results" / "gradcam"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

config = load_config(CONFIG_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Load trained model
# ============================================================

model = timm.create_model(
    config["model"]["name"],
    pretrained=False,
    num_classes=2,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


# ============================================================
# EfficientNet target layer
# ============================================================
# EfficientNet-B0 final convolution block.
# Grad-CAM works best on the last high-level feature map.

target_layers = [model.conv_head]

cam = GradCAM(
    model=model,
    target_layers=target_layers,
)


# ============================================================
# Image preprocessing
# ============================================================

image_size = config["dataset"]["image_size"]

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(               # ImageNet mean, std
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Read prediction table
# ============================================================

df = pd.read_csv(PREDICTION_CSV)


# ============================================================
# Automatically select representative examples
# ============================================================

selected = {}

correct = df[df["correct"] == True]

wrong = df[df["correct"] == False]

# Highest confidence benign

x = correct[
    correct["predicted_label"] == 0
].sort_values(
    "prediction_confidence",
    ascending=False,
)

if len(x):

    selected["correct_benign"] = x.iloc[0]

# Highest confidence malignant

x = correct[
    correct["predicted_label"] == 1
].sort_values(
    "prediction_confidence",
    ascending=False,
)

if len(x):

    selected["correct_malignant"] = x.iloc[0]

# False positive

x = wrong[
    (wrong["predicted_label"] == 1)
].sort_values(
    "prediction_confidence",
    ascending=False,
)

if len(x):

    selected["false_positive"] = x.iloc[0]

# False negative

x = wrong[
    (wrong["predicted_label"] == 0)
].sort_values(
    "prediction_confidence",
    ascending=False,
)

if len(x):

    selected["false_negative"] = x.iloc[0]


# ============================================================
# Generate Grad-CAM
# ============================================================

for name, row in selected.items():

    image_path = IMAGE_DIR / row.filename

    image = Image.open(image_path).convert("RGB")

    rgb = np.array(
        image.resize(
            (image_size, image_size)
        )
    ).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0)

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
    )[0]

    visualization = show_cam_on_image(
        rgb,
        grayscale_cam,
        use_rgb=True,
    )

    output_path = OUTPUT_DIR / f"{name}.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print(f"Saved {output_path}")

print("\nGrad-CAM generation complete.")