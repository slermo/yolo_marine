# YOLO Marine — Detection of Underwater Organisms

Comparison of **YOLOv8s** and **YOLO26s** for marine organism detection on the [Brackish Dataset (AAU)](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset/).

Part of a master's thesis studying the effect of augmentation strategies across YOLO model generations.

## Dataset

6 classes: `fish`, `small_fish`, `crab`, `shrimp`, `jellyfish`, `starfish`

| Split | Frames | Objects |
|-------|--------|---------|
| Train | 11 739 | 28 518  |
| Val   | 1 467  | 3 581   |
| Test  | 1 468  | 3 466   |

## Project Structure

```
yolo_marine/
├── configs/
│   ├── augmentations.yaml   # weak / medium / strong presets
│   ├── dataset.yaml         # dataset paths and class names
│   └── train_base.yaml      # shared training hyperparameters
├── src/
│   ├── prepare_dataset.py   # Brackish videos → YOLO format
│   ├── train.py             # training CLI
│   ├── evaluate.py          # test-set evaluation with per-class metrics
│   └── compare.py           # cross-model comparison charts
├── notebooks/               # EDA and results visualization
├── runs/                    # training artifacts (gitignored)
└── data/                    # dataset files (gitignored)
```

## Setup

Requires Python 3.10–3.12, CUDA 12.8+, PyTorch 2.7+.

```bash
# Install dependencies
poetry install

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Usage

### 1. Prepare dataset

```bash
python -c "import kagglehub; kagglehub.dataset_download('aalborguniversity/brackish-dataset')"
python -m src.prepare_dataset
```

### 2. Train

```bash
# Single experiment
python src/train.py --model yolo26s --aug weak

# All augmentation presets for a model
python src/train.py --model yolo26s --aug weak medium strong

# Reproduce YOLOv8s baseline
python src/train.py --model yolov8s --aug weak medium strong
```

Training parameters (shared, see `configs/train_base.yaml`):
- Epochs: 30, Image size: 640, Batch: 32, Patience: 10, Seed: 42

### 3. Evaluate

```bash
# Evaluate on test set (confusion matrix, per-class AP, speed)
python src/evaluate.py --model yolo26s --aug weak medium strong

# Both models at once
python src/evaluate.py --model yolov8s yolo26s --aug weak medium strong
```

### 4. Compare

```bash
# Generate comparison charts and summary table
python src/compare.py
```

Outputs saved to `runs/figures/`.

## Augmentation Presets

| Parameter | Weak   | Medium | Strong |
|-----------|--------|--------|--------|
| fliplr    | 0.5    | 0.5    | 0.5    |
| hsv_h     | 0.014  | 0.05   | 0.1    |
| hsv_s     | 0.1    | 0.2    | 0.3    |
| hsv_v     | 0.1    | 0.2    | 0.3    |
| degrees   | 0      | 10     | 15     |
| translate | 0.0    | 0.05   | 0.1    |
| scale     | 0.0    | 0.15   | 0.3    |

## Hardware

Target: NVIDIA RTX 5060 Ti 16GB (Blackwell), CUDA 12.8, Windows
