"""
Prepare YOLO dataset from Brackish underwater videos and CSV annotations.

Output structure:
  data/images/{train,val,test}/<frame>.png
  data/labels/{train,val,test}/<frame>.txt  (YOLO format, normalised coords)

Class mapping (matches Brackish.names, 0-indexed for YOLO):
  0: fish
  1: small_fish
  2: crab
  3: shrimp
  4: jellyfish
  5: starfish
"""

import re
import csv
import shutil
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "dataset" / "videos"
ANN_DIR   = DATA_DIR / "annotations" / "annotations_AAU"

IMG_W, IMG_H = 960, 540   # output frame size (matches frameExtractor.py)

SPLITS = {
    "train": (DATA_DIR / "train.txt",  ANN_DIR / "train.csv"),
    "val":   (DATA_DIR / "valid.txt",  ANN_DIR / "valid.csv"),
    "test":  (DATA_DIR / "test.txt",   ANN_DIR / "test.csv"),
}

CLASS_MAP = {
    "fish":       0,
    "small_fish": 1,
    "crab":       2,
    "shrimp":     3,
    "jellyfish":  4,
    "starfish":   5,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches filenames like "2019-03-21_07-40-40to2019-03-21_07-40-50_1-0070.png"
FRAME_RE = re.compile(r"^(.+)-(\d{4})\.png$")


def build_video_index(video_dir: Path) -> dict[str, Path]:
    """Return {video_stem: video_path} for all .avi files found recursively."""
    index = {}
    for path in video_dir.rglob("*.avi"):
        index[path.stem] = path
    return index


def load_annotations(csv_path: Path) -> dict[str, list[tuple]]:
    """
    Parse a split CSV and return {filename: [(class_idx, xc, yc, w, h), ...]}.
    Coordinates are normalised to [0, 1]. Unknown tags are skipped.
    """
    annotations: dict[str, list] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            tag = row["Annotation tag"]
            if tag not in CLASS_MAP:
                print(f"  [skip] unknown tag '{tag}' in {csv_path.name}")
                continue

            x1 = int(row["Upper left corner X"])
            y1 = int(row["Upper left corner Y"])
            x2 = int(row["Lower right corner X"])
            y2 = int(row["Lower right corner Y"])

            xc = ((x1 + x2) / 2) / IMG_W
            yc = ((y1 + y2) / 2) / IMG_H
            w  = (x2 - x1) / IMG_W
            h  = (y2 - y1) / IMG_H

            # Clamp to valid range
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            filename = row["Filename"]
            annotations.setdefault(filename, []).append(
                (CLASS_MAP[tag], xc, yc, w, h)
            )
    return annotations


def extract_frame(video_path: Path, frame_idx: int) -> "cv2.Mat | None":
    """Extract a single frame (0-indexed) from a video, resized to IMG_W×IMG_H."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)


def write_label(label_path: Path, boxes: list[tuple]) -> None:
    """Write YOLO label file (empty file = background frame, also valid)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_split(
    split: str,
    frame_list_path: Path,
    csv_path: Path,
    video_index: dict[str, Path],
) -> None:
    img_out = DATA_DIR / "images" / split
    lbl_out = DATA_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {split} ===")
    annotations = load_annotations(csv_path)

    filenames = frame_list_path.read_text().splitlines()
    ok = skipped = 0

    for filename in filenames:
        filename = filename.strip()
        if not filename:
            continue

        m = FRAME_RE.match(filename)
        if not m:
            print(f"  [warn] unexpected filename format: {filename}")
            skipped += 1
            continue

        video_stem, frame_str = m.group(1), m.group(2)
        frame_idx = int(frame_str) - 1  # ffmpeg %04d is 1-indexed

        if video_stem not in video_index:
            print(f"  [warn] video not found: {video_stem}.avi")
            skipped += 1
            continue

        img_dst = img_out / filename
        lbl_dst = lbl_out / (Path(filename).stem + ".txt")

        # Skip already processed frames
        if img_dst.exists() and lbl_dst.exists():
            ok += 1
            continue

        frame = extract_frame(video_index[video_stem], frame_idx)
        if frame is None:
            print(f"  [warn] could not read frame {frame_idx} from {video_stem}")
            skipped += 1
            continue

        cv2.imwrite(str(img_dst), frame)
        write_label(lbl_dst, annotations.get(filename, []))
        ok += 1

    print(f"  done: {ok} frames, {skipped} skipped")


def main() -> None:
    print("Building video index...")
    video_index = build_video_index(VIDEO_DIR)
    print(f"Found {len(video_index)} videos")

    for split, (frame_list, csv_path) in SPLITS.items():
        prepare_split(split, frame_list, csv_path, video_index)

    print("\nDone. Update data/dataset.yaml classes if needed:")
    print("  0: fish  1: small_fish  2: crab  3: shrimp  4: jellyfish  5: starfish")


if __name__ == "__main__":
    main()
