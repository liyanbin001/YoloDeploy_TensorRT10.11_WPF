# YoloDeploy

Windows x64 / .NET 8 WPF + C++ TensorRT 10.11 deployment app for Ultralytics YOLO Detect and OBB raw-output engines.

## Supported engine profile

Current Phase 6 supports:

- Standard Ultralytics YOLO **Detect** raw output
- Standard Ultralytics YOLO **OBB** raw output
- **YOLO26 instance segmentation** with prediction + prototype outputs
- Batch = 1
- One image input
- Detect/OBB: one 3D prediction output
- Seg: one 3D prediction output + one 4D prototype output
- FP32 or FP16 TensorRT input/output
- Fixed rectangular input H/W such as `1280x512`
- GPU-specific Engine cache
- ONNX -> TensorRT Engine construction
- One-click self-contained Windows Release publishing

Phase 6 recommended industrial model path:

```text
YOLO26n-seg
  -> instance class + confidence
  -> instance binary mask
  -> mask-derived horizontal bounding box
  -> mask-derived minimum-area rotated rectangle
  -> mask pixel area
```

Recommended YOLO26 Seg ONNX export:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.export(
    format="onnx",
    imgsz=(512, 1280),  # height, width
    batch=1,
    dynamic=False,
    end2end=False,
    nms=False,
    simplify=True,
)
```

Recommended raw segmentation outputs:

```text
prediction: [1, 4 + nc + nm, N] or [1, N, 4 + nc + nm]
proto:      [1, nm, protoH, protoW]
```

YOLO26 end-to-end segmentation prediction `[1,max_det,6+nm]` is also accepted,
but `end2end=False, nms=False` is recommended so the deployment post-processing
remains explicit and easy to verify.

For custom classes, replace `YoloDeploy.App\coco.names` with the exact class
count and training order.

Not targeted in Phase 6:

- Dedicated whole-image classification semantics (`yolo26*-cls`)
- Pose
- Semantic segmentation
- INT8 I/O / calibration flow
- More than two model outputs
- Batch > 1
- Multiple image inputs
- Custom preprocessing that differs from LetterBox + RGB + `/255` + NCHW

## Prerequisites

1. Windows x64
2. NVIDIA GPU and compatible driver
3. CUDA Toolkit 12.3
4. TensorRT 10.11.0.33 ZIP installation
5. Visual Studio 2022 with:
   - Desktop development with C++
   - .NET desktop development
   - Windows 10/11 SDK
   - MSVC v143
6. .NET 8 SDK

## Environment variables

Set:

- `TENSORRT_ROOT` = TensorRT installation directory, e.g. `D:\TensorRT-10.11.0.33`
- `CUDA_PATH` should already point to CUDA 12.3, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3`

Run `setup_env_example.bat` after editing the TensorRT path, or set them permanently in Windows.

## Engine recommendation

For the first successful run, export a simple engine from the same PC:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(
    format="engine",
    imgsz=640,
    batch=1,
    dynamic=False,
    half=False,
    nms=False,
    device=0,
)
```

Verify with TensorRT before opening this solution:

```bat
"%TENSORRT_ROOT%\bin\trtexec.exe" --loadEngine="D:\models\yolov8n.engine" --verbose --dumpLayerInfo
```

## Build

1. Open `YoloDeploy.sln` in Visual Studio 2022.
2. Select `Release | x64`.
3. Build solution.
4. The native DLL is automatically copied to the WPF output directory by the C# project.
5. Ensure TensorRT and CUDA DLL folders are available through PATH:
   - `%TENSORRT_ROOT%\lib` (TensorRT 10.11 ZIP commonly keeps runtime DLLs here)
   - `%TENSORRT_ROOT%\bin` (tools such as `trtexec.exe`)
   - `%CUDA_PATH%\bin`
6. Run `YoloDeploy.App`.

## Usage

1. Choose an ONNX and build/reuse the local TensorRT Engine, or browse to an existing `.engine`.
2. For YOLO26 Seg, select **实例分割 Seg → 多任务** (the app also auto-detects the two-output signature).
3. Set confidence, NMS and Mask thresholds.
4. Choose the desired derived visualization:
   - `显示Mask` -> instance segmentation
   - `显示水平框` -> mask-derived object detection
   - `显示最小外接矩形` -> mask-derived rotated box
5. The result table always shows the per-instance class/confidence, so the same Seg model also supplies instance classification.

## Runtime architecture

```text
WPF (.NET 8)
  |  BGRA pixels + image dimensions
  v
YoloDeploy.Native.dll (C++)
  |  LetterBox / RGB / normalize / CHW
  v
TensorRT 10.11 + CUDA
  |  raw YOLOv8 output
  v
C++ decode + class-aware NMS
  |  Detection structs
  v
WPF draws boxes and labels
```

## Troubleshooting

### `YoloDeploy.Native.dll` cannot be loaded

Check that these are reachable via PATH:

- `%TENSORRT_ROOT%\lib\nvinfer_10.dll` (or the actual TensorRT DLL directory in your ZIP package)
- `%TENSORRT_ROOT%\lib\nvinfer_plugin_10.dll` (or the actual TensorRT DLL directory)
- `%CUDA_PATH%\bin\cudart64_12.dll`

Use:

```bat
dumpbin /dependents YoloDeploy.Native.dll
```

### Engine deserialization fails

The engine should preferably be generated on the same RTX 2080 Super and with TensorRT 10.11. Rebuild it locally if needed.

### Output shape unsupported

Run:

```bat
"%TENSORRT_ROOT%\bin\trtexec.exe" --loadEngine="your.engine" --verbose --dumpLayerInfo
```

This project supports standard raw Detect/OBB outputs and the Phase 6 YOLO26 Seg prediction+proto layout. Other custom multi-output heads still require a dedicated adapter.

### Boxes are wrong

Check that the model uses standard Ultralytics preprocessing:
- LetterBox
- pad value 114
- BGR/RGB equivalent input is RGB
- divide by 255
- NCHW


## Phase 2: GPU information + Engine cache management

This version keeps the existing Phase 1 ONNX -> Engine and YOLOv8 inference flow,
and adds a machine-local TensorRT Engine cache.

New capabilities:

- Shows active CUDA GPU name, Compute Capability, VRAM and SM count.
- Shows CUDA Runtime / CUDA Driver API versions.
- Shows the TensorRT header/build version used by the native DLL.
- Stores generated engines under:
  `%LOCALAPPDATA%\YoloDeploy\EngineCache`
- Cache identity includes:
  - SHA-256 of ONNX contents
  - GPU model
  - Compute Capability
  - SM count
  - TensorRT major/minor/patch/build
  - FP32 / FP16
  - input width/height
  - workspace MiB
- Writes a `.engine.json` metadata file beside every cached engine.
- Reuses a valid cache automatically, skipping TensorRT build.
- Supports force rebuild, open-cache-folder and clear-cache operations.

NVIDIA driver / CUDA runtime versions are recorded in metadata but intentionally
not included in the cache key, so a compatible driver update does not by itself
force an expensive engine rebuild.

The existing `YoloBridge.cpp` inference implementation and
`OnnxEngineBuilder.cpp` TensorRT builder implementation remain intact.


## Phase 3: one-click Windows Release

Run:

```bat
publish_release.bat
```

The script:

1. Locates Visual Studio/MSBuild.
2. Builds `YoloDeploy.Native` as `Release | x64`.
3. Publishes WPF as `.NET 8 win-x64 self-contained`.
4. Copies `YoloDeploy.Native.dll`.
5. Collects TensorRT runtime/ONNX parser DLLs.
6. Collects a portable CUDA user-mode runtime set.
7. Copies `.onnx` files from `models` into the release `Models` directory.
8. Adds `verify_runtime.bat`, `run_YoloDeploy.bat`, deployment documentation and a SHA-256 manifest.
9. Creates `dist\YoloDeploy_v3_win-x64` and a ZIP beside it.

The final package still requires a compatible NVIDIA display driver on the target machine.


## Phase 4: fixed rectangular industrial input

The industrial deployment UI now uses separate fixed dimensions:

```text
Fixed input width
Fixed input height
```

Examples:

```text
1280 x 512
1024 x 768
1920 x 640
```

The native APIs already accept width and height separately; Phase 4 wires those
parameters through the WPF UI, cache key, ONNX builder and Engine loader.

Behavior:

- **Fixed-shape ONNX**: the requested UI width/height must exactly match the ONNX
  `[1,3,H,W]` dimensions. A mismatch is rejected with a clear error.
- **Dynamic-shape ONNX**: TensorRT creates one optimization profile with
  `MIN = OPT = MAX = [1,3,H,W]`, so the resulting Engine is still used as a
  fixed-size industrial Engine.
- Engine cache keys continue to include `width x height`.
- Existing LetterBox preprocessing already accepts `dstW` and `dstH` separately.
- Existing detection decoding/NMS and Phase 3 one-click publishing are unchanged.

For standard YOLO models, dimensions that are compatible with the model stride
(often multiples of 32) are recommended. The deployment does not hard-code this
rule because custom industrial models may have different requirements.


## Phase 5: Detect + OBB rotated boxes

The application now keeps two independent inference paths:

```text
Detect -> YoloDetectBgra()    -> axis-aligned decode + class-aware NMS
OBB    -> YoloDetectObbBgra() -> xywhr+angle decode + ProbIoU rotated NMS
```

OBB raw output is expected as:

```text
[x, y, w, h, class_probs..., angle]
```

so the raw channel count is:

```text
5 + number_of_classes
```

The WPF UI draws OBB results as four-point polygons and displays the angle in degrees.
`coco.names` must contain the actual model classes in training order.

The Phase 1 ONNX builder, Phase 2 GPU Engine cache, Phase 4 fixed rectangular W/H
input and Phase 3 one-click publishing are preserved.


## Phase 6: YOLO26 Seg -> unified industrial multi-task geometry

Phase 6 adds a third inference path while keeping Detect and OBB compatible:

```text
Seg -> YoloDetectSegBgra()
    -> class-aware NMS for raw export
    -> mask coefficients x prototypes
    -> original-resolution binary instance mask
    -> mask AABB
    -> convex hull
    -> minimum-area rotated rectangle
```

No OpenCV dependency is added. The minimum-area rectangle is computed in the
native C++ layer using mask boundary points and a convex-hull/minimum-area
geometry routine.

This means one instance-segmentation model can provide four practical outputs:

```text
classification : class id + confidence for each instance
segmentation   : instance mask
detection      : horizontal bounding box derived from mask
rotated box    : minimum-area rectangle derived from mask
```

See:

```text
docs\PHASE6_YOLO26_SEG_MINRECT_CN.md
models\export_yolo26_seg_example.py
```
