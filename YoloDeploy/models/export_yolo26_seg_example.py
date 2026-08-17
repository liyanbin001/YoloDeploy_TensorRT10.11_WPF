from ultralytics import YOLO

# Phase 6 recommended export:
# Use the instance-segmentation model as the single geometry source.
#
# IMPORTANT:
# - imgsz order is (height, width)
# - end2end=False gives the traditional raw prediction tensor required by
#   the external C++ NMS path.
# - nms=False keeps NMS outside the graph.
# - dynamic=False is recommended for fixed industrial camera/model geometry.

MODEL = "yolo26n-seg.pt"

# Example fixed industrial network input:
HEIGHT = 512
WIDTH = 1280

model = YOLO(MODEL)

model.export(
    format="onnx",
    imgsz=(HEIGHT, WIDTH),
    batch=1,
    dynamic=False,
    end2end=False,
    nms=False,
    simplify=True,
)

print("Export complete.")
print(f"WPF fixed input width  = {WIDTH}")
print(f"WPF fixed input height = {HEIGHT}")
