"""
Evaluate the trained model on the test set.
1. Loads the trained model checkpoint.
2. Evaluates the model on the test split.
3. Computes common classification metrics.
4. Saves prediction probabilities for every test image.
5. Generates publication-quality evaluation plots.

Outputs are written to: results/
    metrics.json
    predictions.csv
    classification_report.txt
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ThyroidDataset, build_transforms
from utils import load_config


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "metadata.csv"

IMAGE_ROOT = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load configuration
# ============================================================

config = load_config(CONFIG_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}")


# ============================================================
# Build test dataset
# ============================================================

test_dataset = ThyroidDataset(
    metadata_csv=METADATA_CSV,
    image_root=IMAGE_ROOT,
    split="test",
    transform=build_transforms(CONFIG_PATH, train=False),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)

print(f"Number of test images: {len(test_dataset)}")


# ============================================================
# Build model
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

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)

model.eval()

print("\nLoaded trained model successfully.")


# ============================================================
# Run inference
# ============================================================

predictions = []

true_labels = []

malignant_probabilities = []

filenames = []

prediction_confidences = []

metadata = pd.read_csv(METADATA_CSV)
metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

print("\nRunning inference on test dataset...\n")

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(tqdm(test_loader)):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = F.softmax(outputs, dim=1)   # softmax converts logits into probabilities sum to 1

        predicted = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        for i in range(batch_size):

            predictions.append(int(predicted[i]))

            true_labels.append(int(labels[i]))

            malignant_probabilities.append(
                float(probabilities[i, 1])
            )

            prediction_confidences.append(
                float(probabilities[i, predicted[i]])
            )

            filenames.append(
                metadata.iloc[len(filenames)]["filename"]
            )


# ============================================================
# Compute evaluation metrics
# ============================================================

accuracy = accuracy_score(true_labels, predictions,)
precision = precision_score(true_labels, predictions,)
recall = recall_score(true_labels, predictions,)
f1 = f1_score(true_labels, predictions,)

roc_auc = roc_auc_score(true_labels, malignant_probabilities,)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
}

print("\n==============================")
print("Evaluation Results")
print("==============================")

for key, value in metrics.items():

    print(f"{key:12s}: {value:.4f}")


# ============================================================
# Save metrics
# ============================================================

with open(RESULTS_DIR / "metrics.json", "w") as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# Save prediction table
# ============================================================

prediction_df = pd.DataFrame(
    {
        "filename": filenames,
        "true_label": true_labels,
        "predicted_label": predictions,
        "malignant_probability": malignant_probabilities,
        "prediction_confidence": prediction_confidences,
    }
)

prediction_df["true_class"] = prediction_df[
    "true_label"
].map(
    {
        0: "benign",
        1: "malignant",
    }
)

prediction_df["predicted_class"] = prediction_df[
    "predicted_label"
].map(
    {
        0: "benign",
        1: "malignant",
    }
)

prediction_df["correct"] = (
    prediction_df["true_label"] ==
    prediction_df["predicted_label"]
)


prediction_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

print("\nSaved predictions.csv")

# ============================================================
# Generate classification report
# ============================================================

report = classification_report(
    true_labels,
    predictions,
    target_names=["Benign", "Malignant"],
    digits=4,
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
) as f:

    f.write(report)

print("\nSaved classification_report.txt")


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(true_labels, predictions,)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

class_names = ["Benign", "Malignant"]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=12,
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()

print("Saved confusion_matrix.png")


# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(
    true_labels,
    malignant_probabilities,
)

roc_auc = auc(
    fpr,
    tpr,
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=300,
)

plt.close()

print("Saved roc_curve.png")


# ============================================================
# Precision–Recall Curve
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    true_labels,
    malignant_probabilities,
)

pr_auc = auc(
    recall_curve,
    precision_curve,
)

plt.figure(figsize=(6, 6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2,
    label=f"AUC = {pr_auc:.3f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "precision_recall_curve.png",
    dpi=300,
)

plt.close()

print("Saved precision_recall_curve.png")


# ============================================================
# Display evaluation summary
# ============================================================

print("\n" + "=" * 60)
print("Evaluation Summary")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(report)

print("\nResults saved to:")
print(f"  {RESULTS_DIR}")

print("\nEvaluation complete.")