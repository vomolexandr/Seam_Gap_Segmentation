"""
Quick sanity check of a trained model: run inference on images and
extract the gap edge coordinates from the predicted mask - the same
geometry step that will later feed [x_left, y, x_right, y] to the robot.

Usage:
  python training/predict.py --weights runs/seam_gap/weights/best.pt --source out/preview/images/val
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def edges_from_mask(mask, row_frac=0.5):
    """Left/right gap edge x-coords on a reference row of the binary mask."""
    h = mask.shape[0]
    row = int(h * row_frac)
    xs = np.flatnonzero(mask[row])
    if xs.size == 0:
        return None
    return int(xs[0]), int(xs[-1]), row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--out", type=Path, default=Path("out/predictions"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    paths = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) \
        if src.is_dir() else [src]

    for p in paths:
        img = cv2.imread(str(p))
        res = model(img, verbose=False)[0]
        vis = img.copy()

        if res.masks is not None and len(res.masks) > 0:
            mask = res.masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            vis[mask > 0] = (0.5 * vis[mask > 0]
                             + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            e = edges_from_mask(mask)
            if e:
                xl, xr, row = e
                cv2.circle(vis, (xl, row), 6, (0, 255, 0), 2)
                cv2.circle(vis, (xr, row), 6, (255, 0, 0), 2)
                print(f"{p.name}: left=({xl},{row}) right=({xr},{row}) "
                      f"gap={xr - xl}px")
        else:
            print(f"{p.name}: no gap detected")

        cv2.imwrite(str(args.out / p.name), vis)

    print(f"\nVisualizations in {args.out}")


if __name__ == "__main__":
    main()
