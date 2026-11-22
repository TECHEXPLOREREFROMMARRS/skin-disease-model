# Skin Disease Classification (EfficientNetB3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TECHEXPLOREREFROMMARRS/skin-disease-model/blob/main/skin_disease_model.ipynb)

A transfer-learning image classifier that sorts dermatoscopic skin-lesion images into 7 classes, built with TensorFlow/Keras on top of an ImageNet-pretrained **EfficientNetB3**.

> ⚠️ Educational project only. This model is not a medical device and must not be used for diagnosis.

## Classes

| Code | Condition |
|------|-----------|
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

## Dataset

[Skin Diseases Dataset (Processed)](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-dataset-processed) on Kaggle, downloaded with `kagglehub`. It comes pre-split into train / valid / test folders:

| Split | Images |
|-------|--------|
| Train | 30,819 |
| Validation | 3,820 |
| Test | 3,820 |

## Approach

- **Input:** images resized to 300×300 (EfficientNetB3's native size), loaded with `image_dataset_from_directory`, one-hot labels.
- **Pipeline:** `tf.data` caching, shuffling and prefetching; mixed precision (`mixed_float16`).
- **Augmentation:** random flips, rotation and zoom.
- **Model:** EfficientNetB3 (no top) → GlobalAveragePooling → BatchNorm → Dropout(0.5) → Dense(512, ReLU) → BatchNorm → Dropout(0.4) → Dense(7, softmax).
- **Class imbalance:** balanced class weights from scikit-learn.
- **Training, two stages:**
  1. Frozen base, Adam `1e-3`, up to 30 epochs with EarlyStopping, ReduceLROnPlateau and ModelCheckpoint (`best_effb3_model.h5`).
  2. Fine-tune: unfreeze the last 100 base layers and recompile with Adam `1e-5`.

## Running it

1. Open the notebook in Google Colab using the badge above.
2. Switch the runtime to a GPU (Runtime → Change runtime type).
3. Run the cells in order. The dataset downloads automatically with `kagglehub`.

Requirements: `tensorflow`, `kagglehub`, `scikit-learn`, `numpy`.

## Status / known issues

Work in progress. Final test-set accuracy has not been recorded yet.

- The Stage 2 cells rebuild the base model from ImageNet weights rather than loading the Stage 1 checkpoint. Load `best_effb3_model.h5` before fine-tuning so Stage 1 training isn't lost.
- Some cells are duplicated and call `.batch(8)` on datasets that are already batched (batch size 16). Clean that up before re-running end to end.
- No evaluation on the test set yet. Next: `model.evaluate(test_ds)`, a confusion matrix and a per-class report.
