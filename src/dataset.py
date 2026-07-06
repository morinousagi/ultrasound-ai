"""
Reads the metadata.csv generated during preprocessing,
loads cropped thyroid nodule images, applies Albumentations transforms,
and returns tensors suitable for model training.
"""

from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml


class ThyroidDataset(Dataset):
    """
    Dataset class for cropped thyroid ultrasound images.

    Parameters
    ----------
    metadata_csv : Path
        Path to metadata.csv.

    image_root : Path
        Root directory containing cropped images.

    split : str
        One of {"train", "val", "test"}.

    transform : Albumentations Compose, optional
        Image augmentation pipeline.
    """

    def __init__(
        self,
        metadata_csv: Path,
        image_root: Path,
        split: str,
        transform: Optional[A.Compose] = None,
    ):

        self.df = pd.read_csv(metadata_csv)

        self.df = (
            self.df[self.df["split"] == split]
            .reset_index(drop=True)
        )

        self.image_root = image_root
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = self.image_root / row["crop_path"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        # OpenCV loads images as BGR.
        # Convert to RGB to match ImageNet pretrained models.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row["label"])

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# ---------------------------------------------------------------------
# Image augmentation
#
# Ultrasound images require conservative augmentation because excessive
# geometric distortion can alter medically meaningful structures.
# ---------------------------------------------------------------------

def build_transforms(config_path: Path, train: bool = True):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_size = config["dataset"]["image_size"]

    if train:

        return A.Compose(
            [
                A.Resize(image_size, image_size),

                A.HorizontalFlip(p=0.5),

                A.Rotate(
                    limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.5,
                ),

                A.Normalize(),

                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.Normalize(),

            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------
# Simple sanity check
#
# Running this file directly verifies that images can be loaded,
# transformed, and returned in the expected tensor format.
# ---------------------------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    dataset = ThyroidDataset(
        metadata_csv=PROJECT_ROOT / "data/processed/metadata.csv",
        image_root=PROJECT_ROOT / "data/processed",
        split="train",
        transform=build_transforms(
            PROJECT_ROOT / "config.yaml",
            train=True,
        ),
    )

    image, label = dataset[0]

    print(f"Dataset size : {len(dataset)}")
    print(f"Image shape  : {image.shape}")
    print(f"Label        : {label}")