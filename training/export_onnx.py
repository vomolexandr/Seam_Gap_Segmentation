"""
Export trained weights to ONNX for the Hailo toolchain (Pi 5 + AI HAT+).

Hailo's Dataflow Compiler consumes ONNX; opset 11 is the safest choice.

Usage:
  python training/export_onnx.py --weights runs/seam_gap/weights/best.pt
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, opset=11, simplify=True)
    print(f"\nONNX saved: {path}")
    print("Next: compile with Hailo Dataflow Compiler / hailomz to .hef for the AI HAT+.")


if __name__ == "__main__":
    main()
