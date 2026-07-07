"""
Gradio application for thyroid ultrasound classification.

Features
--------
1. Upload a thyroid ultrasound image.
2. Predict Benign or Malignant.
3. Display prediction confidence.
4. Generate a Grad-CAM explanation.

This demo is intended for research and educational purposes only.
"""

from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.utils import load_config


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[0]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"


# ============================================================
# Configuration
# ============================================================

config = load_config(CONFIG_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant",
}

IMAGE_SIZE = config["dataset"]["image_size"]


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Load model once
# ============================================================

print("Loading model...")

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

print("Model loaded.")


# ============================================================
# Grad-CAM
# ============================================================

target_layers = [
    model.conv_head
]

cam = GradCAM(
    model=model,
    target_layers=target_layers,
)


# ============================================================
# Prediction function
# ============================================================

def predict(image):

    if image is None:

        return (
            None,
            "Please upload an image.",
            None,
        )

    image = image.convert("RGB")

    rgb = np.array(
        image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    ).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        prediction = outputs.argmax(dim=1).item()

        confidence = probabilities[
            0,
            prediction,
        ].item()

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb,
        grayscale_cam,
        use_rgb=True,
    )

    label = CLASS_NAMES[prediction]

    result = (
        f"### Prediction: **{label}**\n\n"
        f"Confidence: **{confidence:.2%}**"
    )

    heatmap = Image.fromarray(
        visualization
    )

    return (
        heatmap,
        result,
        {
            "Benign": float(probabilities[0, 0]),
            "Malignant": float(probabilities[0, 1]),
        },
    )



# ============================================================
# Gradio User Interface
# ============================================================

DESCRIPTION = """
Upload a thyroid ultrasound image to classify it as **Benign** or **Malignant**.

⚠️ **Research Use Only**

This model was trained using lesion-centered crops extracted from expert annotations in the TN5000 dataset.
It is intended solely for research and educational demonstration and must **not** be used for clinical diagnosis.

Refer to README for more information.
"""

with gr.Blocks(
    title="AI Thyroid Ultrasound Classifier"
) as demo:

    gr.Markdown("# AI Thyroid Ultrasound Classifier")

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="Upload Ultrasound Image",
        )

        gradcam_image = gr.Image(
            label="Grad-CAM Visualization",
        )

    with gr.Row():

        prediction_text = gr.Markdown()

        probability_label = gr.Label(
            label="Prediction Probabilities",
            num_top_classes=2,
        )

    with gr.Row():

        predict_button = gr.Button(
            "Predict",
            variant="primary",
        )

        clear_button = gr.ClearButton(
            [
                input_image,
                gradcam_image,
                prediction_text,
                probability_label,
            ]
        )

    predict_button.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            gradcam_image,
            prediction_text,
            probability_label,
        ],
    )



# ============================================================
# Launch application
# ============================================================

if __name__ == "__main__":

    demo.launch()