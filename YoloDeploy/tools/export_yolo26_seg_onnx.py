"""Export a fixed-size YOLO26 instance-segmentation model to end-to-end ONNX.

Development-time helper only. The deployed WPF application does not require Python.
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to yolo26n-seg/best.pt")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")

    model = YOLO(args.model)

    output = model.export(
        format="onnx",
        imgsz=(args.height, args.width),  # Ultralytics uses (H, W)
        batch=1,
        dynamic=False,
        end2end=True,
        device=args.device,
    )

    print(f"Exported ONNX: {output}")


if __name__ == "__main__":
    main()
