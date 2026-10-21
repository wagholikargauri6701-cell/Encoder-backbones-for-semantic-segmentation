# Encoder Comparison for Semantic Segmentation

> **Comparative Study of VGG16-BN and ResNet34 Encoders in U-Net for Semantic Segmentation on the CamVid Dataset**

**Author:** Gauri Nitin Wagholikar  
**Supervisor:** Prof. Dr. Tatyana Ivanovska  
**Institution:** Ostbayerische Technische Hochschule Amberg-Weiden Department of Electrical Engineering, Media and Computer Science  
**Report Date:** June 25, 2026


## Overview

This project presents a controlled comparison of two U-Net variants for semantic segmentation on the **CamVid** urban driving benchmark. The two variants differ only in their encoder backbone **VGG16-BN** vs **ResNet34** keeping the decoder architecture identical so that any performance gap is attributable to the encoder alone.

The study evaluates both models across segmentation quality (mIoU, Dice, Boundary F1) and computational efficiency (FPS, GFLOPs, latency), providing a concrete accuracy-vs-efficiency trade-off analysis for real-time deployment scenarios.

**Keywords:** Semantic Segmentation, U-Net, VGG16-BN, ResNet34, CamVid, Transfer Learning, Optuna, PyTorch

---

## Key Results

### Segmentation Quality (CamVid Test Set)

| Metric             | UNet-VGG16-BN | UNet-ResNet34 |
|--------------------|:-------------:|:-------------:|
| **mIoU**           | **0.713**     | 0.673         |
| Pixel Accuracy     | **0.933**     | 0.922         |
| Mean Dice          | **0.815**     | 0.784         |
| Macro Precision    | **0.843**     | 0.817         |
| Macro Recall       | **0.793**     | 0.760         |
| Mean Boundary F1   | **0.663**     | 0.605         |

### Computational Efficiency (CamVid Test Set)

| Metric              | UNet-VGG16-BN | UNet-ResNet34 |
|---------------------|:-------------:|:-------------:|
| Parameters          | 25.79 M       | 31.57 M       |
| Model Size          | 98.4 MB       | 120.4 MB      |
| GFLOPs              | 76.42         | **17.75**     |
| GPU Peak Memory     | 199.3 MB      | **178.6 MB**  |
| Avg. Latency        | 19.34 ms      | **8.65 ms**   |
| **FPS**             | 51.7          | **115.6**     |

### Core Takeaways

- **VGG16-BN wins on every segmentation quality metric**, leading by 3.9 mIoU points and 5.8 boundary F1 points. Its full-resolution (256×256) skip connection gives the decoder access to fine spatial detail that ResNet34 cannot match.
- **ResNet34 wins on every efficiency metric**, running at ~4.3× lower GFLOPs and more than twice the FPS. Despite having more parameters and a larger file size, it uses *less* GPU memory at inference because its aggressively downsampled activations occupy far less space than VGG16-BN's full-resolution feature maps.
- **Both models struggle on thin, infrequent classes** (ColumnPole: IoU ~0.28/0.22, Pedestrian: ~0.50/0.43), reflecting the severe class imbalance inherent to CamVid rather than an encoder-specific limitation.
- The deployment choice depends on the application: VGG16-BN for boundary-critical use cases, ResNet34 for constrained real-time systems.

---

## Project Structure

```
encoder-comparison-for-semantic_segmentation/
│
├── src/
│   ├── dataset.py          # CamVid dataset loader with RGB→class LUT and augmentation pipeline
│   ├── model.py            # UNet with pluggable VGG16-BN / ResNet34 encoder backbones
│   ├── utils.py            # Loss functions (Dice, Combined), metrics (mIoU, Dice, BF1), latency tools
│   ├── train.py            # Full training loop with early stopping, cosine LR scheduling, AMP
│   ├── test.py             # Evaluation script segmentation quality + efficiency metrics
│   ├── optimization.py     # Optuna hyperparameter search (20 trials per encoder)
│   ├── analysis.py         # Plotting and visualisation (run after test.py)
│   └── requirements.txt    # Python dependencies
│
├── CamVid/                 # Dataset directory
│   ├── train/              # Training images
│   ├── train_labels/       # Training segmentation masks
│   ├── val/                # Validation images
│   ├── val_labels/         # Validation segmentation masks
│   ├── test/               # Test images
│   └── test_labels/        # Test segmentation masks
│
└── results/
    ├── checkpoints/        # Saved model weights (.pth)
    │   ├── unet_vgg16_best.pth
    │   ├── unet_vgg16_last.pth
    │   ├── unet_resnet34_best.pth
    │   └── unet_resnet34_last.pth
    ├── plots/              # All generated visualisations
    ├── optuna_results/     # Optuna trial CSVs and plots
    ├── history_vgg16.json
    ├── history_resnet34.json
    ├── test_metrics_vgg16.json
    ├── test_metrics_resnet34.json
    ├── confusion_matrix_vgg16.json
    ├── confusion_matrix_resnet34.json
    ├── latency_vgg16.json
    ├── latency_resnet34.json
    └── best_params_*.json
```

---

## Dataset

**CamVid** (Cambridge-Driving Labelled Video Database) is an urban driving dataset with pixel-level semantic annotations.

https://www.kaggle.com/datasets/carlolepelaars/camvid/data

### 11-Class Mapping

Multiple original CamVid colour labels are consolidated into 11 classes:

| ID | Class       | Pixel % (Train) |
|----|-------------|-----------------|
| 0  | Sky         | ~17%            |
| 1  | Building    | ~25%            |
| 2  | ColumnPole  | ~1.1%           |
| 3  | Road        | ~31.9%          |
| 4  | Sidewalk    | ~6%             |
| 5  | Tree        | ~8%             |
| 6  | SignSymbol  | ~1.7%           |
| 7  | Fence       | ~1.5%           |
| 8  | Car         | ~5%             |
| 9  | Pedestrian  | ~1.5%           |
| 10 | Bicyclist   | ~0.8%           |

Void pixels (black / unlabelled) use `IGNORE_INDEX = 255` and are excluded from all loss and metric computations.

### Data Augmentation (Training only)

Horizontal flip · Random rotation (±10°) · Random scale (±20%) · Random crop · Colour jitter · Random gamma · Gaussian blur · Gaussian noise · ImageNet normalisation

---


## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, CPU inference is supported but slow)
- Git

### Setup

```bash
# Clone the repository
git clone https://git.oth-aw.de/1d5f/encoder-comparison-for-semantic-segmentation.git
cd encoder-comparison-for-sementic_segmentation

# Install dependencies
pip install -r src/requirements.txt
```

### Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
albumentations>=1.4.0
opencv-python>=4.8.0
optuna>=3.4.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.11.0
```


## Usage

All scripts are run from the `src/` directory. The recommended order is:

```
optimization → train → test → analysis
```

### 1. Hyperparameter Search (optional)

Runs 20 Optuna trials per encoder, searching over learning rate, weight decay, optimizer, batch size, and loss weights. Results are saved to `results/optuna_results/`.

```bash
cd src
python optimization.py
```

Best parameters are written to `results/best_params_vgg16.json` and `results/best_params_resnet34.json`, and are already hard-coded into `train.py`.

### 2. Training

```bash
# Train both encoders
python train.py --encoder both

# Train a specific encoder
python train.py --encoder vgg16
python train.py --encoder resnet34
```

Checkpoints are saved to `results/checkpoints/unet_<encoder>_best.pth` whenever validation mIoU improves. Training history (loss, mIoU, LR, epoch time) is saved to `results/history_<encoder>.json`.
               |

### 3. Evaluation

Runs the full evaluation suite on the test set: mIoU, Dice, Boundary F1, precision/recall, confusion matrix, model size, GFLOPs, latency, and FPS.

```bash
# Evaluate both encoders
python test.py --encoder both

# Evaluate a specific encoder
python test.py --encoder vgg16
python test.py --encoder resnet34
```

Outputs are saved to `results/test_metrics_<encoder>.json`, `results/confusion_matrix_<encoder>.json`, and `results/latency_<encoder>.json`.

### 4. Analysis and Visualisation

Generates all plots from the saved results. **Run after `test.py`.**

```bash
python analysis.py
```

Plots are saved to `results/plots/` and include:

- Loss and mIoU convergence curves
- Per-class IoU and Boundary F1 comparisons
- Confusion matrices
- Accuracy-vs-efficiency trade-off (mIoU vs FPS / GFLOPs / parameters)
- Inference time distribution (box plot over 50 runs)
- Gradient norm distribution
- Prediction samples and failure cases
- Class imbalance (train vs test pixel frequency)

---

## Results

### Best Hyperparameters (from Optuna)

| Setting        | UNet-VGG16-BN          | UNet-ResNet34          |
|----------------|------------------------|------------------------|
| Optimizer      | AdamW                  | Adam                   |
| Learning rate  | 1.10 × 10⁻⁴            | 1.84 × 10⁻⁴            |
| Weight decay   | 5.43 × 10⁻⁵            | 3.19 × 10⁻⁶            |
| Batch size     | 2                      | 2                      |
| CE weight      | 0.623                  | 0.557                  |
| Dice weight    | 0.377                  | 0.443                  |
| Best val mIoU  | 0.748                  | 0.720                  |

### Per-Class IoU (Test Set)

| Class       | VGG16-BN | ResNet34 |
|-------------|:--------:|:--------:|
| Sky         | 0.93     | 0.93     |
| Building    | 0.87     | 0.86     |
| ColumnPole  | 0.28     | 0.22     |
| Road        | 0.96     | 0.96     |
| Sidewalk    | 0.84     | 0.81     |
| Tree        | 0.80     | 0.78     |
| SignSymbol  | 0.61     | 0.56     |
| Fence       | 0.60     | 0.53     |
| Car         | 0.82     | 0.78     |
| Pedestrian  | 0.50     | 0.43     |
| Bicyclist   | 0.62     | 0.55     |

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
2. Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
4. Brostow, G. J., Shotton, J., Fauqueur, J., & Cipolla, R. (2008). CamVid Cambridge-Driving Labelled Video Database.
5. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *arXiv:1907.10902*.
6. Google. (2025). Gemini Large language model. Used for coding assistance and debugging.

---

*Project Report Computer Vision and AI, OTH Amberg-Weiden, June 2026*
