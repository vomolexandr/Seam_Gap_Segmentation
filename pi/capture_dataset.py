"""
Dataset capture on Raspberry Pi 5 + IMX477 (runs ONLY on the Pi).

Saves stills for the fine-tuning dataset. Frames go to a plain folder;
transfer them to the laptop (scp / USB) and upload to Roboflow or label
with SAM there.

Usage (on Pi):
  python3 capture_dataset.py --out ~/seam_frames            # ENTER = shot
  python3 capture_dataset.py --out ~/seam_frames --every 2  # every 2 s

Tips for useful frames:
  - vary distance, angle, lighting between shots
  - include hard cases: glare, shadows, dirty plates, minimal gap
"""

import argparse
import time
from pathlib import Path

from picamera2 import Picamera2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=None,
                    help="auto-capture interval in seconds (default: manual)")
    ap.add_argument("--width", type=int, default=2028)
    ap.add_argument("--height", type=int, default=1520)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    start = len(list(args.out.glob("frame_*.jpg")))

    cam = Picamera2()
    cfg = cam.create_still_configuration(
        main={"size": (args.width, args.height)})
    cam.configure(cfg)
    cam.start()
    time.sleep(2)  # let AE/AWB settle

    i = start
    print(f"Saving to {args.out} (starting at frame_{i:04d}). Ctrl+C to stop.")
    try:
        while True:
            if args.every is None:
                input(f"ENTER -> frame_{i:04d}.jpg ")
            else:
                time.sleep(args.every)
            path = args.out / f"frame_{i:04d}.jpg"
            cam.capture_file(str(path))
            print(f"saved {path}")
            i += 1
    except KeyboardInterrupt:
        print(f"\nDone: {i - start} new frames.")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
