# Salvage Vehicle Damage Classification (CNN)

A deep learning computer vision pipeline designed to automatically classify the location and severity of vehicle damage from raw images. 

By implementing **Transfer Learning** via a pre-trained EfficientNetB0 architecture and engineering a robust dynamic data augmentation pipeline, this model effectively bypasses severe class imbalances and real-world noise to achieve an **89% validation accuracy** on unseen salvage data.

## The Business Problem
When assessing salvage vehicles, real-world photos are exceptionally noisy. They are captured at unpredictable angles, in poor lighting, and feature severe class imbalances (e.g., front-end collisions are statistically far more common than roof damage). 

Standard Convolutional Neural Networks (CNNs) trained from scratch on this data suffer from **overfitting**. The models lazily memorize the background or orientation of the training photos and fail to learn how the actual geometric features of the damage influence the classification.

## Transfer Learning & Augmentation Strategy
To force the model to learn the specific visual traits of vehicular damage, the architecture was redesigned to leverage a pre-trained **EfficientNetB0** base:

By freezing the core layers pre-trained on the ImageNet database, the model dedicated 100% of its computational power to fine-tuning the top layers for the 8 specific damage classes. Furthermore, dynamic data augmentation (15-degree rotations, spatial shifting, zooming, and horizontal flipping) was applied to the training set, ensuring the model learned the true geometry of dents and scratches regardless of the camera angle.

## Model Performance
Trained on a structured vehicle damage assessment dataset, the model was evaluated on its ability to correctly classify unseen validation images across 8 distinct categories. Early Stopping and Model Checkpointing were utilized to capture the peak performing weights.

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Training Accuracy** | 91.54% | Baseline |
| **Validation Accuracy** | **88.68%** | **Peak Generalization** |
| **Validation Loss** | 0.3781 | Optimized |

---

## Tech Stack & Architecture
* Data Ingestion & Manipulation: `kagglehub`, `pandas`
* Computer Vision: `TensorFlow`, `Keras` (EfficientNetB0)
* Visualization & Metrics: `Plotly`, `scikit-learn`
* Compute Environment: Google Colab (T4 GPU)

---
Developed by Jose Thomas
