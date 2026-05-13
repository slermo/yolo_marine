"""
Prepare DUO underwater dataset (COCO JSON) for YOLO training.

Reads:
  data/DUO/annotations/instances_train.json
  data/DUO/annotations/instances_test.json
  data/DUO/images/{train,test}/*.jpg

Writes (YOLO layout):
  data/DUO/yolo/images/{train,val,test}/*.jpg
  data/DUO/yolo/labels/{train,val,test}/*.txt

Split:
  - Train COCO file is split 90/10 into train/val (seed=42).
  - Test COCO file becomes test split as-is.

Class mapping (COCO 1-indexed -> YOLO 0-indexed):
  0: holothurian   1: echinus   2: scallop   3: starfish

Cross-platform: uses shutil.copy2 and pathlib (no symlinks, no shell ops).

Usage:
    python src/prepare_duo.py
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUO_DIR = ROOT / "data" / "DUO"
ANN_DIR = DUO_DIR / "annotations"
SRC_IMG_DIR = DUO_DIR / "images"
OUT_DIR = DUO_DIR / "yolo"

VAL_FRACTION = 0.10
SEED = 42

# COCO category_id (1-indexed) -> YOLO class_id (0-indexed)
CAT_ID_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3}
CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        grouped[ann["image_id"]].append(ann)
    return grouped


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] (pixels, top-left origin) -> YOLO [xc, yc, w, h] (normalised)."""
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, w / img_w)),
        max(0.0, min(1.0, h / img_h)),
    )


def write_label_file(label_path: Path, anns: list[dict], img_w: int, img_h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for ann in anns:
        cat_id = ann["category_id"]
        if cat_id not in CAT_ID_TO_YOLO:
            continue
        cls = CAT_ID_TO_YOLO[cat_id]
        xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def process_split(
    split_name: str,
    images: list[dict],
    anns_by_image: dict[int, list[dict]],
    src_img_dir: Path,
) -> None:
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied = labeled = 0
    missing: list[str] = []

    for img in images:
        file_name = img["file_name"]
        src = src_img_dir / file_name
        if not src.exists():
            missing.append(file_name)
            continue

        dst = img_out / file_name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

        label_path = lbl_out / (Path(file_name).stem + ".txt")
        write_label_file(label_path, anns_by_image.get(img["id"], []), img["width"], img["height"])
        labeled += 1

    print(f"  {split_name:5s}: {copied} images, {labeled} labels"
          + (f", {len(missing)} missing" if missing else ""))
    if missing[:3]:
        print(f"          missing examples: {missing[:3]}")


def main() -> None:
    print(f"Output: {OUT_DIR}")

    # ---- train -> train/val 90/10 ----
    print("\nLoading instances_train.json ...")
    train_coco = load_coco(ANN_DIR / "instances_train.json")
    train_anns = group_annotations_by_image(train_coco)

    images = list(train_coco["images"])
    rng = random.Random(SEED)
    rng.shuffle(images)
    n_val = int(round(len(images) * VAL_FRACTION))
    val_images = images[:n_val]
    train_images = images[n_val:]
    print(f"  total: {len(images)}  ->  train: {len(train_images)}  val: {len(val_images)}")

    print("\nWriting train split ...")
    process_split("train", train_images, train_anns, SRC_IMG_DIR / "train")

    print("Writing val split ...")
    process_split("val", val_images, train_anns, SRC_IMG_DIR / "train")

    # ---- test (as-is) ----
    print("\nLoading instances_test.json ...")
    test_coco = load_coco(ANN_DIR / "instances_test.json")
    test_anns = group_annotations_by_image(test_coco)
    print(f"  total: {len(test_coco['images'])}")

    print("Writing test split ...")
    process_split("test", test_coco["images"], test_anns, SRC_IMG_DIR / "test")

    print("\nDone.")
    print(f"  images: {OUT_DIR / 'images'}")
    print(f"  labels: {OUT_DIR / 'labels'}")
    print(f"  classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
