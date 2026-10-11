"""
Procedural butt-weld seam generator.

Renders synthetic grayscale frames of two metal plates with a dark gap
(butt joint before welding) and produces pixel-perfect labels for free:

  out/
    images/{train,val}/seam_00000.jpg      - synthetic frame
    labels/{train,val}/seam_00000.txt      - YOLO segmentation polygon (class 0 = gap)
    masks/seam_00000.png                   - binary gap mask (255 = gap)
    edges/seam_00000.json                  - exact left/right gap edge x-coords per row
    dataset.yaml                           - ready for Ultralytics training

The same images+labels can be uploaded straight to Roboflow (YOLO format)
or trained on locally without Roboflow at all.

Usage:
  python synthetic/generate_seams.py --count 500 --out out/synthetic
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 640  # square frames, matches YOLO input


# ----------------------------------------------------------------------------
# metal texture
# ----------------------------------------------------------------------------

def brushed_metal(h, w, rng, base_gray, streak_angle_deg):
    """Gray plate with brushed-metal streaks and low-frequency mottling."""
    img = np.full((h, w), base_gray, np.float32)

    # low-frequency mottling (uneven surface reflectivity)
    mottle = rng.normal(0, 1, (h // 16, w // 16)).astype(np.float32)
    mottle = cv2.resize(mottle, (w, h), interpolation=cv2.INTER_CUBIC)
    img += mottle * rng.uniform(4, 14)

    # brushed streaks: noise smeared along one direction
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    ksize = int(rng.uniform(15, 41)) | 1
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    rot = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), streak_angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
    kernel /= kernel.sum() + 1e-6
    img += cv2.filter2D(noise, -1, kernel) * rng.uniform(6, 18)

    # occasional scratches
    for _ in range(rng.integers(0, 5)):
        p1 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        p2 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        cv2.line(img, p1, p2, float(base_gray + rng.uniform(-35, 35)),
                 1, cv2.LINE_AA)

    return img


# ----------------------------------------------------------------------------
# gap geometry
# ----------------------------------------------------------------------------

def gap_edges(h, rng):
    """
    Returns (x_left, x_right) float arrays of length h: exact gap edge
    x-coordinate for every image row. Centerline is a slanted line plus
    smooth low-frequency wander; width varies slowly along the seam.
    """
    y = np.arange(h, dtype=np.float32)

    center0 = rng.uniform(0.30, 0.70) * IMG_SIZE
    slope = np.tan(np.radians(rng.uniform(-25, 25)))
    center = center0 + slope * (y - h / 2)

    # smooth wander (plate edges are never perfectly straight)
    for _ in range(rng.integers(1, 4)):
        amp = rng.uniform(1, 8)
        freq = rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        center += amp * np.sin(freq * 2 * np.pi * y / h + phase)

    width = np.full(h, rng.uniform(4, 26), np.float32)
    width += rng.uniform(0.5, 4.0) * np.sin(
        rng.uniform(0.5, 1.5) * 2 * np.pi * y / h + rng.uniform(0, 2 * np.pi))
    # per-row jitter from cut roughness
    jitter = cv2.GaussianBlur(rng.normal(0, 1, (h, 1)).astype(np.float32),
                              (1, 31), 0).ravel()
    width = np.clip(width + jitter * 2.0, 2.5, None)

    return center - width / 2, center + width / 2


def render_gap(img, x_left, x_right, rng):
    """Draw the dark gap onto the plate image; returns the binary mask."""
    h, w = img.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    xl = x_left[:, None]
    xr = x_right[:, None]

    inside = (xs >= xl) & (xs < xr)
    mask = inside.astype(np.uint8) * 255

    # gap is dark, slightly uneven (looking into the joint)
    depth_gray = rng.uniform(5, 40)
    gap_tex = rng.normal(0, 6, img.shape).astype(np.float32) + depth_gray
    # soft shading toward gap edges (chamfer / light falloff)
    dist_l = np.clip(xs - xl, 0, None)
    dist_r = np.clip(xr - xs, 0, None)
    edge_dist = np.minimum(dist_l, dist_r)
    shade = np.clip(edge_dist / np.maximum((xr - xl) / 2, 1), 0, 1)
    gap_val = gap_tex + (1 - shade) * rng.uniform(10, 30)

    img[inside] = gap_val[inside]

    # bright chamfer line on one edge sometimes (light catches the cut edge)
    if rng.random() < 0.5:
        edge = np.clip((x_left if rng.random() < 0.5 else x_right)
                       .round().astype(int), 0, w - 1)
        rows = np.arange(h)
        img[rows, edge] = np.clip(
            img[rows, edge] + rng.uniform(20, 60), 0, 255)

    return mask


# ----------------------------------------------------------------------------
# photometric effects
# ----------------------------------------------------------------------------

def apply_effects(img, rng):
    h, w = img.shape

    # uneven illumination: gradient + bright spot
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gdir = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(gdir) * xx / w + np.sin(gdir) * yy / h)
    img += grad * rng.uniform(-30, 30)

    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    r = rng.uniform(0.3, 1.0) * IMG_SIZE
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    img += spot * rng.uniform(0, 50)

    # specular streak highlights
    for _ in range(rng.integers(0, 3)):
        overlay = np.zeros_like(img)
        p1 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        p2 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        cv2.line(overlay, p1, p2, float(rng.uniform(40, 90)),
                 int(rng.uniform(6, 30)), cv2.LINE_AA)
        img += cv2.GaussianBlur(overlay, (31, 31), 0)

    # weld spatter: small bright and dark dots
    for _ in range(rng.integers(0, 25)):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        val = rng.choice([rng.uniform(150, 255), rng.uniform(0, 50)])
        cv2.circle(img, c, int(rng.uniform(1, 4)), float(val), -1, cv2.LINE_AA)

    # occasional simulated arc overexposure blob
    if rng.random() < 0.15:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(30, 120)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img += blob * rng.uniform(150, 300)

    # sensor noise, blur, gamma
    img += rng.normal(0, rng.uniform(2, 9), img.shape)
    if rng.random() < 0.5:
        k = int(rng.uniform(3, 7)) | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.75, 1.3)

    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------

def yolo_seg_label(x_left, x_right, h, w, step=8):
    """YOLO segmentation line: left edge top->bottom, right edge bottom->up."""
    rows = list(range(0, h, step))
    if rows[-1] != h - 1:
        rows.append(h - 1)
    pts = [(x_left[r], r) for r in rows] + [(x_right[r], r) for r in reversed(rows)]
    coords = []
    for x, y in pts:
        coords.append(f"{np.clip(x / w, 0, 1):.5f}")
        coords.append(f"{np.clip(y / (h - 1), 0, 1):.5f}")
    return "0 " + " ".join(coords) + "\n"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def generate_one(rng):
    angle = rng.uniform(0, 180)
    left_plate = brushed_metal(IMG_SIZE, IMG_SIZE, rng,
                               rng.uniform(90, 190), angle)
    right_plate = brushed_metal(IMG_SIZE, IMG_SIZE, rng,
                                rng.uniform(90, 190), angle + rng.uniform(-8, 8))

    x_left, x_right = gap_edges(IMG_SIZE, rng)

    # composite: left plate up to gap, right plate after
    xs = np.arange(IMG_SIZE, dtype=np.float32)[None, :]
    img = np.where(xs < ((x_left + x_right) / 2)[:, None], left_plate, right_plate)

    mask = render_gap(img, x_left, x_right, rng)
    img = apply_effects(img, rng)

    return img, mask, x_left, x_right


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("out/synthetic"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(exist_ok=True)
    (args.out / "edges").mkdir(exist_ok=True)

    for i in range(args.count):
        img, mask, x_left, x_right = generate_one(rng)
        split = "val" if random.random() < args.val_frac else "train"
        name = f"seam_{i:05d}"

        cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.out / "labels" / split / f"{name}.txt").write_text(
            yolo_seg_label(x_left, x_right, IMG_SIZE, IMG_SIZE))
        cv2.imwrite(str(args.out / "masks" / f"{name}.png"), mask)
        (args.out / "edges" / f"{name}.json").write_text(json.dumps({
            "x_left": [round(float(v), 2) for v in x_left],
            "x_right": [round(float(v), 2) for v in x_right],
        }))

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.count}")

    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: gap\n")
    print(f"Done: {args.count} frames in {args.out}")


if __name__ == "__main__":
    main()
