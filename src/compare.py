"""
Compare YOLOv8s vs YOLO26s results across augmentation presets.

Reads evaluation_results.csv and generates:
  - Side-by-side bar charts (mAP, precision, recall)
  - Per-class AP comparison
  - Inference speed comparison
  - Training curves overlay (from results.csv in each run)

Usage:
    python src/compare.py
    python src/compare.py --out figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

MODELS = ["yolov8s", "yolo26s"]
AUGS = ["weak", "medium", "strong"]
MODEL_COLORS = {"yolov8s": "#5B8CF5", "yolo26s": "#E05C5C"}
AUG_MARKERS = {"weak": "o", "medium": "s", "strong": "D"}

CLASS_NAMES = ["fish", "small_fish", "crab", "shrimp", "jellyfish", "starfish"]


def load_eval_results() -> pd.DataFrame:
    path = RUNS_DIR / "evaluation_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluate.py first."
        )
    return pd.read_csv(path)


def load_training_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    for model in MODELS:
        for aug in AUGS:
            path = RUNS_DIR / model / aug / "results.csv"
            if path.exists():
                df = pd.read_csv(path, skipinitialspace=True)
                curves[f"{model}/{aug}"] = df
    return curves


def plot_overall_metrics(df: pd.DataFrame, out_dir: Path):
    metrics = ["mAP50", "mAP50-95", "precision", "recall"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    x = np.arange(len(AUGS))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(MODELS):
            subset = df[df["model"] == model].set_index("augmentation")
            vals = [subset.loc[a, metric] if a in subset.index else 0 for a in AUGS]
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, vals, width, label=model,
                          color=MODEL_COLORS[model], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(AUGS)
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("YOLOv8s vs YOLO26s — Overall Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_overall.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'compare_overall.png'}")


def plot_per_class_ap(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, cls in enumerate(CLASS_NAMES):
        ax = axes[idx]
        col = f"AP50_{cls}"
        if col not in df.columns:
            continue

        x = np.arange(len(AUGS))
        width = 0.35

        for i, model in enumerate(MODELS):
            subset = df[df["model"] == model].set_index("augmentation")
            vals = [subset.loc[a, col] if a in subset.index else 0 for a in AUGS]
            offset = -width / 2 + i * width
            ax.bar(x + offset, vals, width, label=model,
                   color=MODEL_COLORS[model], alpha=0.85)

        ax.set_title(f"{cls} — AP@50")
        ax.set_xticks(x)
        ax.set_xticklabels(AUGS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Per-Class AP@50 Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_per_class.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'compare_per_class.png'}")


def plot_speed(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(AUGS))
    width = 0.35

    for i, model in enumerate(MODELS):
        subset = df[df["model"] == model].set_index("augmentation")
        vals = [subset.loc[a, "speed_inference_ms"] if a in subset.index else 0
                for a in AUGS]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=MODEL_COLORS[model], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Inference time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(AUGS)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_title("Inference Speed Comparison (per image)")

    plt.tight_layout()
    plt.savefig(out_dir / "compare_speed.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'compare_speed.png'}")


def plot_training_curves(curves: dict, out_dir: Path):
    metric_cols = {
        "mAP@50": "metrics/mAP50(B)",
        "mAP@50-95": "metrics/mAP50-95(B)",
        "Train Loss (box)": "train/box_loss",
        "Train Loss (cls)": "train/cls_loss",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (title, col) in zip(axes, metric_cols.items()):
        for key, df in curves.items():
            if col not in df.columns:
                continue
            model, aug = key.split("/")
            ax.plot(df[col], label=key,
                    color=MODEL_COLORS.get(model, "gray"),
                    linestyle="-" if aug == "medium" else ("--" if aug == "weak" else ":"),
                    linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — YOLOv8s vs YOLO26s", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_training_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'compare_training_curves.png'}")


def print_summary_table(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    cols = ["model", "augmentation", "mAP50", "mAP50-95",
            "precision", "recall", "speed_inference_ms"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="%.4f"))

    # Delta table
    print("\n" + "-" * 80)
    print("YOLO26s improvement over YOLOv8s (delta)")
    print("-" * 80)
    for aug in AUGS:
        v8 = df[(df["model"] == "yolov8s") & (df["augmentation"] == aug)]
        v26 = df[(df["model"] == "yolo26s") & (df["augmentation"] == aug)]
        if v8.empty or v26.empty:
            continue
        for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
            delta = v26[metric].values[0] - v8[metric].values[0]
            sign = "+" if delta >= 0 else ""
            print(f"  {aug:8s} {metric:12s}: {sign}{delta:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO model results")
    parser.add_argument("--out", default="runs/figures", help="Output directory")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_results()

    plot_overall_metrics(df, out_dir)
    plot_per_class_ap(df, out_dir)
    plot_speed(df, out_dir)

    curves = load_training_curves()
    if curves:
        plot_training_curves(curves, out_dir)

    print_summary_table(df)


if __name__ == "__main__":
    main()
