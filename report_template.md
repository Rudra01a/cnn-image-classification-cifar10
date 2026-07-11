# Project Report: CIFAR-10 Image Classification Using CNNs

**Name:** _[Your name]_
**Course / Program:** _[e.g. B.Tech CSE, 3rd Year]_
**Date:** _[Date]_

---

## 1. Objective

_State in 2-4 sentences what the project sets out to do. Example starting point:_

> The objective of this project is to design, train, and evaluate a Convolutional
> Neural Network (CNN) capable of classifying 32×32 color images into one of 10
> categories using the CIFAR-10 dataset, and to analyze the model's strengths and
> weaknesses through quantitative metrics and visual inspection of predictions.

## 2. Dataset

- **Name:** CIFAR-10
- **Size:** 60,000 images total — 50,000 training / 10,000 test (further split into
  train/validation as described in Methodology)
- **Image dimensions:** 32 × 32 × 3 (RGB)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Class balance:** _[Fill in after running Step 2 — e.g. "perfectly balanced, 6,000
  images per class"]_
- **Source:** Loaded directly via `tensorflow.keras.datasets.cifar10`

## 3. Methodology

### 3.1 Preprocessing
- _Describe normalization (pixel scaling to [0,1]) and why it matters for training
  stability._
- _Describe the train/validation/test split ratios and why stratification was used._

### 3.2 Model Architecture
- _Briefly describe the 3-block CNN structure: number of conv layers, filter
  progression, role of BatchNormalization, MaxPooling, and Dropout._
- _State total parameter count: [Fill in from model.summary() — expect ~552,874]._

### 3.3 Training Configuration
| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss function | Sparse categorical crossentropy |
| Batch size | 64 |
| Max epochs | 20 |
| Data augmentation | Rotation, width/height shift, horizontal flip |
| Regularization | Dropout (0.25–0.5), BatchNormalization, EarlyStopping |
| Early stopping triggered at epoch | _[Fill in]_ |

## 4. Results

| Metric | Value |
|---|---|
| Final training accuracy | _[Fill in]_ |
| Final validation accuracy | _[Fill in]_ |
| **Test accuracy** | _[Fill in]_ |
| Test loss | _[Fill in]_ |

**Accuracy/Loss curves:** _[Insert or reference the plots from Step 7. Comment on
whether the curves show healthy convergence, overfitting, or underfitting.]_

**Confusion matrix observations:** _[Which classes were most confused with each other?
e.g. "cat and dog were the most frequently confused pair, consistent with their visual
similarity at low resolution."]_

**Classification report highlights:** _[Which class had the highest/lowest F1-score?
Any notable precision/recall imbalance?]_

**Sample predictions:** _[Briefly describe patterns noticed among incorrectly classified
images — blurry images, occlusion, ambiguous poses, etc.]_

## 5. Challenges

_Document real obstacles you hit and how you resolved them. Examples to adapt:_
- Balancing model capacity against overfitting risk on a relatively small (32×32) image
  dataset.
- Tuning dropout rates and learning rate schedule to get stable convergence.
- _[Add your own — e.g. training time constraints, GPU availability, specific
  debugging steps.]_

## 6. Future Improvements

- **Deeper/residual architecture:** Try a ResNet-style model with skip connections to
  push past the ~80% accuracy ceiling of a plain CNN this size.
- **Stronger augmentation:** Add Cutout, MixUp, or RandAugment for better
  generalization.
- **Transfer learning:** Fine-tune a pretrained backbone (e.g. EfficientNet, ResNet)
  upscaled to a larger input size.
- **Hyperparameter tuning:** Systematic search (Keras Tuner / Optuna) over learning
  rate, dropout rates, and filter counts.
- **Global Average Pooling** instead of Flatten before the dense head, to cut
  parameter count significantly (the current Dense(128) layer alone accounts for
  ~262k of the model's ~553k parameters).

## 7. Conclusion

_Summarize in 3-5 sentences: what was built, what accuracy was achieved, and the single
biggest takeaway from the project._
