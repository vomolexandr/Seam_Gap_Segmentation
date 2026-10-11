"""
Pseudo-label real images with a trained model (self-training).

Runs the model over unlabeled images, keeps confident single-seam
predictions, and writes YOLO-seg polygon labels next to them in a new
dataset folder ready to merge into training.

Usage:
  python training/pseudolabel.py --weights runs/segment/seam_gap_v1-2/weights/best.pt \
      --source datasets/raw_timepass --out datasets/pseudo_timepass --conf 0.6
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def mask_to_polygon(mask, max_points=60):
    """Largest contour of a binary mask -> simplified polygon (N,2) or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    eps = 0.5
    poly = c
    while len(poly) > max_points:
        poly = cv2.approxPolyDP(c, eps, True)
        eps *= 1.5
    return poly.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True,
                    help="folder with images (searched recursively)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.6,
                    help="keep only predictions above this confidence")
    ap.add_argument("--val-every", type=int, default=8,
                    help="every N-th accepted image goes to val split")
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in Path(args.source).rglob("*")
                   if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"no images under {args.source}")

    for split in ("train", "valid"):
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
    review_dir = args.out / "review"
    review_dir.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    kept = skipped = 0

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model(img, verbose=False)[0]

        ok = (res.masks is not None and len(res.masks) == 1
              and float(res.boxes.conf[0]) >= args.conf)
        if ok:
            mask = res.masks.data[0].cpu().numpy()
            mask = (cv2.resize(mask, (w, h)) > 0.5).astype(np.uint8)
            poly = mask_to_polygon(mask)
            ok = poly is not None and len(poly) >= 4

        if not ok:
            skipped += 1
            continue

        split = "valid" if kept % args.val_every == args.val_every - 1 else "train"
        name = f"pl_{kept:04d}"
        shutil.copy(p, args.out / split / "images" / f"{name}.jpg")
        norm = poly / [w, h]
        line = "0 " + " ".join(f"{v:.5f}" for v in norm.ravel())
        (args.out / split / "labels" / f"{name}.txt").write_text(line + "\n")

        # review image: polygon drawn for human check
        vis = img.copy()
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 0, 255),
                      max(2, w // 400))
        cv2.putText(vis, f"{float(res.boxes.conf[0]):.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imwrite(str(review_dir / f"{name}.jpg"), vis)
        kept += 1

    print(f"accepted {kept}, skipped {skipped} (low conf / no seam / multi-seam)")
    print(f"labels in {args.out}, review overlays in {review_dir}")


if __name__ == "__main__":
    main()
