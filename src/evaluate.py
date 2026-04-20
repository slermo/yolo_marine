"""
Evaluate trained models on the test split.

Produces per-model:
  - mAP@50, mAP@50-95, precision, recall (overall + per-class)
  - Confusion matrix (saved as PNG)
  - Inference speed (preprocess + inference + postprocess ms)

Usage:
    python src/evaluate.py --model yolo26s --aug weak
    python src/evaluate.py --model yolo26s --aug weak medium strong
    python src/evaluate.py --model yolov8s yolo26s --aug weak medium strong
"""

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
DATASET_YAML = str(CONFIGS_DIR / "dataset.yaml")


def find_best_weights(model_name: str, aug_name: str) -> Path:
    run_dir = ROOT / "runs" / model_name / aug_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Weights not found: {best}")
    return best


def evaluate_single(model_name: str, aug_name: str) -> dict:
    best = find_best_weights(model_name, aug_name)
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name} / {aug_name}")
    print(f"  Weights:    {best}")
    print(f"{'='*60}\n")

    model = YOLO(str(best))

    metrics = model.val(
        data=DATASET_YAML,
        split="test",
        batch=1,
        device=0,
        plots=True,
        save_json=True,
    )

    speed = metrics.speed  # dict: preprocess, inference, postprocess (ms)

    result = {
        "model": model_name,
        "augmentation": aug_name,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_preprocess_ms": speed.get("preprocess", 0),
        "speed_inference_ms": speed.get("inference", 0),
        "speed_postprocess_ms": speed.get("postprocess", 0),
    }

    # Per-class AP
    class_names = metrics.names
    for i, name in class_names.items():
        result[f"AP50_{name}"] = float(metrics.box.ap50[i])
        result[f"AP50-95_{name}"] = float(metrics.box.ap[i])

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on test set")
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=["yolov8s", "yolo26s"],
    )
    parser.add_argument(
        "--aug",
        required=True,
        nargs="+",
        choices=["weak", "medium", "strong"],
    )
    args = parser.parse_args()

    rows = []
    for model_name in args.model:
        for aug_name in args.aug:
            result = evaluate_single(model_name, aug_name)
            rows.append(result)

    df = pd.DataFrame(rows)

    out_path = ROOT / "runs" / "evaluation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
