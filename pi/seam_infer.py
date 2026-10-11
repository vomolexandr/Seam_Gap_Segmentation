"""
Standalone YOLOv8-seg decoder + seam edge extractor.

No Ultralytics dependency - only numpy + cv2 + onnxruntime. This is the
exact post-processing the Pi will run (with ONNX Runtime now, with HailoRT
later: the decode after the raw network outputs is identical).

Network I/O (yolov8n-seg, 640x640, 1 class):
  input:  (1, 3, 640, 640) float32, RGB, 0..1
  out[0]: (1, 37, 8400)  -> 4 box (cx,cy,w,h) + 1 class score + 32 mask coeffs
  out[1]: (1, 32, 160, 160) mask prototypes
"""

import cv2
import numpy as np

INPUT_SIZE = 640
PROTO_SIZE = 160


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=INPUT_SIZE):
    """Resize keeping aspect ratio, pad to square. Returns image, scale, pads."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh))
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def preprocess(img_bgr):
    padded, r, left, top = letterbox(img_bgr)
    x = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), r, left, top


def decode(pred, proto, conf_thres=0.5):
    """Best seam candidate -> (mask640 float, box xyxy in 640-space, score)."""
    p = pred[0]                    # (37, 8400)
    scores = sigmoid(p[4]) if p[4].max() > 1 else p[4]  # exported models emit sigmoid already
    i = int(np.argmax(scores))
    if scores[i] < conf_thres:
        return None, None, float(scores[i])

    cx, cy, w, h = p[0, i], p[1, i], p[2, i], p[3, i]
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    coeffs = p[5:, i]              # (32,)

    m = sigmoid(coeffs @ proto[0].reshape(32, -1)).reshape(PROTO_SIZE, PROTO_SIZE)
    m = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE))

    # zero the mask outside the box
    x1, y1, x2, y2 = np.clip(box, 0, INPUT_SIZE).astype(int)
    crop = np.zeros_like(m)
    crop[y1:y2, x1:x2] = m[y1:y2, x1:x2]
    return crop, box, float(scores[i])


def mask_to_original(mask640, img_shape, r, left, top, thresh=0.5):
    """Undo the letterbox: mask in original image coordinates, uint8 {0,1}."""
    h, w = img_shape[:2]
    unpad = mask640[top:top + round(h * r), left:left + round(w * r)]
    mask = cv2.resize(unpad, (w, h))
    return (mask > thresh).astype(np.uint8)


def seam_edges(mask, n_rows=5):
    """
    Edge coordinates along the seam: for n_rows evenly spaced rows that
    intersect the mask, return (y, x_left, x_right). This is the payload
    that goes to the robot.
    """
    rows_with_mask = np.flatnonzero(mask.any(axis=1))
    if rows_with_mask.size == 0:
        return []
    picks = np.linspace(rows_with_mask[0], rows_with_mask[-1], n_rows).astype(int)
    result = []
    for y in picks:
        xs = np.flatnonzero(mask[y])
        if xs.size:
            result.append((int(y), int(xs[0]), int(xs[-1])))
    return result


class SeamDetector:
    """ONNX Runtime wrapper. On Pi this class is swapped for a HailoRT one;
    decode/edge code above stays identical."""

    def __init__(self, onnx_path, conf_thres=0.5):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.conf_thres = conf_thres

    def __call__(self, img_bgr):
        """Returns dict with score, mask (original size) and edge points."""
        x, r, left, top = preprocess(img_bgr)
        pred, proto = self.sess.run(None, {self.input_name: x})
        mask640, box, score = decode(pred, proto, self.conf_thres)
        if mask640 is None:
            return {"score": score, "mask": None, "edges": []}
        mask = mask_to_original(mask640, img_bgr.shape, r, left, top)
        return {"score": score, "mask": mask, "edges": seam_edges(mask)}


if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None, help="dir for visualizations")
    args = ap.parse_args()

    det = SeamDetector(args.onnx)
    paths = sorted(glob.glob(os.path.join(args.source, "*.jpg"))) \
        if os.path.isdir(args.source) else [args.source]

    for p in paths:
        img = cv2.imread(p)
        res = det(img)
        print(f"{os.path.basename(p)}: score={res['score']:.3f} "
              f"edges={res['edges']}")
        if args.out and res["mask"] is not None:
            os.makedirs(args.out, exist_ok=True)
            vis = img.copy()
            vis[res["mask"] > 0] = (0.5 * vis[res["mask"] > 0]
                                    + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            for y, xl, xr in res["edges"]:
                cv2.circle(vis, (xl, y), 5, (0, 255, 0), 2)
                cv2.circle(vis, (xr, y), 5, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(args.out, os.path.basename(p)), vis)
