# CIFAR-10 CNN Image Classification

A complete, beginner-friendly image classification pipeline built with TensorFlow/Keras,
trained on the CIFAR-10 dataset (60,000 32×32 color images, 10 classes).

## Folder Structure

```
cifar10_cnn_project/
├── cifar10_cnn_classification.ipynb   # Main notebook — run this end to end
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── report_template.md                  # Fill-in-the-blank project report
├── viva_questions.md                   # Interview/viva prep Q&A
└── saved_model/                        # Created automatically after training
    ├── cifar10_cnn_model.keras         # Model in modern Keras v3 format
    └── cifar10_cnn_model.h5            # Model in legacy HDF5 format
```

## Setup Instructions

### Option A — Google Colab (recommended for beginners, free GPU)
1. Upload `cifar10_cnn_classification.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type → GPU** (T4 is fine).
3. Run all cells (**Runtime → Run all**). TensorFlow, NumPy, etc. are pre-installed on Colab.

### Option B — Local Jupyter Notebook
```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook cifar10_cnn_classification.ipynb
```

**Note on runtime:** 20 epochs on CIFAR-10 takes roughly 10–20 minutes on a Colab GPU,
and can take well over an hour on CPU only. If you're on CPU and just want to confirm
the pipeline works end to end, temporarily lower `EPOCHS` in the training cell.

## Model Architecture Summary

A 3-block CNN (Conv2D ×2 → BatchNorm → MaxPool → Dropout, per block, with filters
doubling 32→64→128) followed by a dense classification head. Full parameter breakdown:

| Component | Trainable Params | Non-trainable Params |
|---|---|---|
| Conv Block 1 (32 filters) | 10,272 | 128 |
| Conv Block 2 (64 filters) | 55,680 | 512 |
| Conv Block 3 (128 filters) | 221,952 | 1,024 |
| Dense Head (128 → 10) | 263,562 | 512 |
| **Total** | **551,722** | **1,152** |

**Total parameters: 552,874.** See the notebook's markdown cell right after
`model.summary()` for the full layer-by-layer table and the reasoning behind each
design choice (why double convs, why increasing dropout, why BatchNorm placement, etc.).

## What Each Notebook Section Does

| Step | Section | Purpose |
|---|---|---|
| 0 | Environment setup | Imports, seeds, GPU check |
| 1 | Dataset loading | Load CIFAR-10 via Keras, with error handling |
| 2 | Data visualization | Sample image grid + class balance check |
| 3 | Preprocessing | Normalize pixels to [0,1]; stratified train/val split |
| 4 | CNN architecture | Build the model; inspect `model.summary()` |
| 5 | Compilation | Adam optimizer, sparse categorical crossentropy |
| 6 | Training | 20 epochs, data augmentation, early stopping, LR scheduling |
| 7 | Accuracy/loss plots | Diagnose over/underfitting visually |
| 8 | Evaluation | Unbiased test-set accuracy and loss |
| 9 | Confusion matrix | Per-class error patterns |
| 10 | Correct/incorrect examples | Visual inspection of predictions |
| 11 | Classification report | Per-class precision/recall/F1 |
| 12 | Save model | `.keras` and `.h5` exports, with reload sanity check |

## Expected Results

With this architecture and 20 epochs of training (subject to early stopping), typical
test accuracy on CIFAR-10 lands in the **~78–85%** range. This is a solid result for a
from-scratch CNN of this size — state-of-the-art CIFAR-10 models (deeper ResNets, heavy
augmentation, hundreds of epochs) reach 95%+, which is a useful benchmark for the
"Future Improvements" section of your report.
