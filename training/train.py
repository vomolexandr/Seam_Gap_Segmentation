"""
Train YOLO segmentation on the seam-gap dataset.

Works with either the synthetic dataset (out/synthetic/dataset.yaml)
or a Roboflow YOLO-seg export (datasets/<name>/data.yaml), or a merged set.

Usage:
  python training/train.py --data out/synthetic/dataset.yaml
  python training/train.py --data datasets/roboflow/data.yaml --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to dataset.yaml")
    ap.add_argument("--model", default="yolov8n-seg.pt",
                    help="base model (yolov8n-seg.pt fits 4GB VRAM)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="seam_gap")
    ap.add_argument("--resume-from", default=None,
                    help="fine-tune from previous best.pt instead of COCO weights")
    args = ap.parse_args()

    model = YOLO(args.resume_from or args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        # photometric augmentation: our domain-shift insurance
        hsv_v=0.5,       # strong brightness variation
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.5,
    )
    print(f"\nBest weights: runs/segment/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
