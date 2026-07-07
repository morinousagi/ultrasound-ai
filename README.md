---
title: Ultrasound Ai
emoji: 📚
colorFrom: indigo
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# AI Thyroid Ultrasound Classifier

This project is completed collaboratively with **ChatGPT** through prompt engineering and AI-assisted coding.

An AI-assisted thyroid nodule classification from ultrasound images using EfficientNet with Grad-CAM explainability and Hugging Face deployment.

- Model was trained using lesion-centered regions extracted from expert annotations.
- Demo accepts full ultrasound images for educational and research demonstration only.
- Not intended for clinical use.

Demo app deployment: [Hugging Face link](https://huggingface.co/spaces/morinousagi/ultrasound-ai)

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

`preprocess.py`
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

Missing malignant nodules is undesirable in screening applications, `recall` is emphasized alongside F1-score during model evaluation.

`evaluate.py`
- Load best_model.pth
- Evaluate on the test split (1000 images)
```
==============================
Evaluation Results
==============================
accuracy    : 0.8290
precision   : 0.8931
recall      : 0.8643
f1_score    : 0.8785
roc_auc     : 0.8813

Saved predictions.csv

Saved classification_report.txt
Saved confusion_matrix.png
Saved roc_curve.png
Saved precision_recall_curve.png

============================================================
Evaluation Summary
============================================================
Accuracy : 0.8290
Precision: 0.8931
Recall   : 0.8643
F1 Score : 0.8785
ROC AUC  : 0.8813

Confusion Matrix
[[211  74]
 [ 97 618]]

Classification Report
              precision    recall  f1-score   support

      Benign     0.6851    0.7404    0.7116       285
   Malignant     0.8931    0.8643    0.8785       715

    accuracy                         0.8290      1000
   macro avg     0.7891    0.8023    0.7951      1000
weighted avg     0.8338    0.8290    0.8309      1000
```

## Grad-CAM Visualization

`gradcam.py`
1. Loads the trained model.
2. Reads predictions.csv produced by evaluate.py.
3. Selects representative examples and computes Grad-CAM heatmaps.
5. Saves overlay visualizations to `results/gradcam/`


## Files
```
ultrasound-ai/
├── app.py                  
├── requirements.txt
├── README.md
├── config.yaml
│
├── checkpoints/
│   └── best_model.pth     # Available in HF
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── gradcam/
│
├── data/
│   └── processed/
│       └── metadata.csv

```

## Future Improvements

This project focuses on building an end-to-end deep learning pipeline for thyroid ultrasound classification within a short development timeline. Several enhancements could further improve its usability and research value:

* **Improved Grad-CAM visualization:** Blend the heatmap onto the original full-resolution image with adjustable transparency. This would make it easier to localize the highlighted regions while preserving anatomical context.
* **Nodule localization:** Integrate an object detection or segmentation model (e.g., YOLO or U-Net) to identify the thyroid nodule before classification, removing the reliance on lesion-centered crops.
* **Threshold optimization:** Tune the decision threshold to prioritize sensitivity (recall) for screening-oriented applications, reducing missed malignant cases.
* **Model comparison:** Benchmark multiple architectures (e.g., EfficientNet, ConvNeXt, Vision Transformer) to evaluate the trade-offs between accuracy, inference speed, and computational cost.
