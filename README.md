# Thyroid Ultrasound AI (In-progress)

This project is completed collaboratively with **ChatGPT** through prompt engineering and AI-assisted coding.

Explainable Thyroid Nodule Classification using Lesion-Centric Deep Learning.

An AI-assisted thyroid nodule classification from ultrasound images using EfficientNet with Grad-CAM explainability and Hugging Face deployment.


Demo app deployment: [Hugging Face link]()

## Ultrasound Dataset

TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification
- Zhang, H., Liu, Q., Han, X. et al. TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification. Sci Data 12, 1437 (2025). https://doi.org/10.1038/s41597-025-05757-4

5,000 ultrasound .jpg images
- 3,572 malignant (label = 1)
- 1,428 benign (label = 0)
- PASCAL VOC annotations
- One .xml annotation per image
- Variable image sizes
- Bounding boxes provided

Class imbalance:
- Benign 0: 28.6%
- Malignant 1: 71.4%

## Preprocessing & Augmentation

`prepare_dataset.py`
- Expands bounding boxes by 15%
- Crops each nodule ROI, saves cropped images to a new directory
- Generates train/validation/test splits
- Creates a metadata.csv
```
label    0     1
split           
test   285   715
train  998  2502
val    143   357
```
`dataset.py`
- Loads cropped thyroid nodule images
- Applies Albumentations transforms,
- Returns tensors suitable for model training.
```
Dataset size : 3500
Image shape  : torch.Size([3, 224, 224])
Label        : 0
```

## EfficientNet-B0 Transfer Learning

```
Model          EfficientNet-B0
Input size     224×224
Epochs         20
Batch size     16
Optimizer      AdamW
LR             5e-5
Loss           Weighted CrossEntropy
Scheduler      ReduceLROnPlateau
Early Stop     F1 (patience=5)

Class weights: [1.75350701 0.69944045]

checkpoints/best_model.pth

============================================================
Epoch 5/20
============================================================
Learning Rate : 0.000050
Train Loss : 0.4950
Val Loss   : 0.7160
Accuracy   : 0.8120
Precision  : 0.8856
Recall     : 0.8459
F1 Score   : 0.8653
✓ Best model saved.

============================================================
Epoch 6/20
============================================================
Learning Rate : 0.000050
Train Loss : 0.4031
Val Loss   : 0.6640
Accuracy   : 0.7980
Precision  : 0.8879
Recall     : 0.8207
F1 Score   : 0.8530

============================================================
Epoch 7/20
============================================================
Learning Rate : 0.000050
Train Loss : 0.3342
Val Loss   : 0.6376
Accuracy   : 0.7880
Precision  : 0.8702
Recall     : 0.8263
F1 Score   : 0.8477

Early stopping triggered.

Training complete.
Best validation F1: 0.8653

```


## Evaluation

- Because missing malignant nodules is undesirable in screening applications, `recall` is emphasized alongside F1-score during model evaluation.


## Grad-CAM Visualization


## Demo App


## Files
```
ultrasound-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── gradcam.py
│   └── utils.py
│
├── models/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## ENV
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```