"""
Build a demo reel from underwater video fragments with YOLOv8 detections + ByteTrack IDs.

Usage:
    python -m src.demo_video --config configs/demo_video.yaml
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO


# BGR. Order matches class indexes from prepare_dataset.py:
# 0 fish, 1 small_fish, 2 crab, 3 shrimp, 4 jellyfish, 5 starfish
CLASS_COLORS = {
    0: (245, 165, 66),
    1: (74, 195, 139),
    2: (38, 167, 255),
    3: (122, 64, 236),
    4: (188, 71, 171),
    5: (79, 213, 255),
}


def latest_weights(glob_pattern: str) -> Path:
    matches = sorted(Path(".").glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No weights match: {glob_pattern}")
    return matches[-1]


def annotate(frame, result, names, title=None) -> None:
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(cls)
        )
        for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
            color = CLASS_COLORS.get(int(c), (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{names[int(c)]} {p:.2f}"
            if tid is not None:
                label = f"#{int(tid)} " + label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(y1 - th - 6, 0)
            cv2.rectangle(
                frame, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 3, y_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    if title:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (10, 10), (26 + tw, 30 + th), (0, 0, 0), -1)
        cv2.putText(
            frame,
            title,
            (18, 22 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def render_scene(model, scene: dict, out_path: Path, conf: float, imgsz: int) -> None:
    video = Path(scene["video"])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(float(scene.get("start", 0.0)) * fps)
    n_frames = int(float(scene.get("duration", 5.0)) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Drop tracker state so IDs restart for each scene.
    if getattr(model, "predictor", None) is not None:
        model.predictor = None

    title = scene.get("title")
    desc = f"{video.stem} @ {start_f / fps:.1f}s +{n_frames / fps:.1f}s"
    for i in tqdm(range(n_frames), desc=desc):
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(
            frame,
            conf=conf,
            imgsz=imgsz,
            persist=(i > 0),
            tracker="bytetrack.yaml",
            verbose=False,
        )
        annotate(frame, results[0], model.names, title)
        writer.write(frame)

    cap.release()
    writer.release()


def concat_with_ffmpeg(scene_paths: list[Path], output: Path) -> None:
    list_file = output.parent / "_concat.txt"
    with list_file.open("w") as f:
        for p in scene_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ],
        check=True,
    )
    list_file.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_video.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if not cfg.get("scenes"):
        raise SystemExit("config has no 'scenes'")

    weights = cfg.get("weights") or latest_weights(
        "runs/runs/yolov8s/weak*/weights/best.pt"
    )
    output = Path(cfg.get("output", "runs/demo/demo.mp4"))
    conf = float(cfg.get("conf", 0.35))
    imgsz = int(cfg.get("imgsz", 640))

    print(f"Weights: {weights}")
    model = YOLO(str(weights))

    scenes_dir = output.parent / "scenes"
    if scenes_dir.exists():
        shutil.rmtree(scenes_dir)
    scenes_dir.mkdir(parents=True)

    scene_paths: list[Path] = []
    for i, scene in enumerate(cfg["scenes"], 1):
        out = scenes_dir / f"scene_{i:02d}.mp4"
        render_scene(model, scene, out, conf, imgsz)
        scene_paths.append(out)

    print(f"\nConcatenating {len(scene_paths)} scenes -> {output}")
    concat_with_ffmpeg(scene_paths, output)
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
