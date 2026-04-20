"""
Train a YOLO model on the Brackish dataset with a specified augmentation preset.

Usage:
    python src/train.py --model yolo26s --aug weak
    python src/train.py --model yolov8s --aug strong
    python src/train.py --model yolo26s --aug weak medium strong   # run all three
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Map CLI model names to pretrained weights
MODEL_WEIGHTS = {
    "yolov8s": "yolov8s.pt",
    "yolo26s": "yolo26s.pt",
}


def train_single(model_name: str, aug_name: str, base_cfg: dict, aug_cfg: dict) -> Path:
    weights = MODEL_WEIGHTS[model_name]
    project = str(ROOT / "runs" / model_name)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Augmentation: {aug_name}")
    print(f"{'='*60}\n")

    model = YOLO(weights)

    train_args = {
        **base_cfg,
        **aug_cfg,
        "data": str(CONFIGS_DIR / "dataset.yaml"),
        "project": project,
        "name": aug_name,
    }

    model.train(**train_args)

    return Path(project) / aug_name


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Brackish dataset")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_WEIGHTS),
        help="Model variant to train",
    )
    parser.add_argument(
        "--aug",
        required=True,
        nargs="+",
        choices=["weak", "medium", "strong"],
        help="Augmentation preset(s)",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(CONFIGS_DIR / "train_base.yaml")
    aug_cfgs = load_yaml(CONFIGS_DIR / "augmentations.yaml")

    for aug_name in args.aug:
        train_single(args.model, aug_name, base_cfg, aug_cfgs[aug_name])

    print("\nAll training runs complete.")


if __name__ == "__main__":
    main()
