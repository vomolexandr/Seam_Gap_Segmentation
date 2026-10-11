"""
Live seam detection on the Pi: camera -> model -> edge coords -> Modbus.

Runs the ONNX model with onnxruntime (CPU) for bring-up. Once the model is
compiled to .hef, only the inference call changes - decode and Modbus stay.

The Pi acts as a Modbus TCP SERVER (slave): it keeps fresh seam coordinates
in holding registers and the robot controller polls them.

Register map (holding registers, 1 register = 16 bit unsigned):
  0: frame counter (increments every processed frame)
  1: status: 0 = no seam, 1 = seam detected
  2: confidence x1000
  3: number of edge points N
  4..4+3N: N triplets [y, x_left, x_right] in image pixels

Usage (on Pi):
  python3 run_live.py --onnx best.onnx                 # print only
  python3 run_live.py --onnx best.onnx --modbus 502    # + Modbus server
"""

import argparse
import threading
import time

import cv2

from seam_infer import SeamDetector

N_ROWS = 5


def start_modbus_server(port):
    """pymodbus 3.x TCP server in a background thread; returns context."""
    from pymodbus.datastore import (ModbusSequentialDataBlock,
                                    ModbusServerContext,
                                    ModbusSlaveContext)
    from pymodbus.server import StartTcpServer

    block = ModbusSequentialDataBlock(0, [0] * (4 + 3 * N_ROWS + 4))
    slave = ModbusSlaveContext(hr=block)
    context = ModbusServerContext(slaves=slave, single=True)
    threading.Thread(
        target=StartTcpServer,
        kwargs={"context": context, "address": ("0.0.0.0", port)},
        daemon=True).start()
    print(f"Modbus TCP server on :{port}")
    return context


def publish(context, frame_no, result):
    edges = result["edges"]
    regs = [frame_no & 0xFFFF,
            1 if edges else 0,
            int(result["score"] * 1000),
            len(edges)]
    for y, xl, xr in edges:
        regs += [y, xl, xr]
    context[0].setValues(3, 0, regs)  # fc 3 = holding registers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--modbus", type=int, default=None, help="TCP port")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--roi", default=None, metavar="X1,Y1,X2,Y2",
                    help="search region in pixels; suppresses false positives "
                         "from background lines outside the work area")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--show", action="store_true", help="preview window")
    args = ap.parse_args()

    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
    time.sleep(1)

    det = SeamDetector(args.onnx, conf_thres=args.conf)
    ctx = start_modbus_server(args.modbus) if args.modbus else None
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

    frame_no = 0
    t0, fps = time.time(), 0.0
    while True:
        frame = cam.capture_array()  # RGB888 -> already BGR-compatible 3ch
        if roi:
            x1, y1, x2, y2 = roi
            result = det(frame[y1:y2, x1:x2])
            # edge coords back to full-frame pixels
            result["edges"] = [(y + y1, xl + x1, xr + x1)
                               for y, xl, xr in result["edges"]]
        else:
            result = det(frame)
        frame_no += 1

        if ctx:
            publish(ctx, frame_no, result)

        fps = 0.9 * fps + 0.1 * (1 / max(time.time() - t0, 1e-3))
        t0 = time.time()
        e = result["edges"]
        mid = e[len(e) // 2] if e else None
        print(f"\rframe {frame_no} fps={fps:4.1f} score={result['score']:.2f} "
              f"mid_edge={mid}   ", end="")

        if args.show:
            vis = frame.copy()
            if result["mask"] is not None:
                region = vis[y1:y2, x1:x2] if roi else vis
                region[result["mask"] > 0] = (0, 0, 255)
            cv2.imshow("seam", vis)
            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    main()
