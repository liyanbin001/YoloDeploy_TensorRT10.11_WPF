# YoloDeploy

Windows x64 / .NET 8 WPF + C++ TensorRT 10.11 inference demo for standard Ultralytics YOLOv8 Detect engines.

## Supported engine profile

This first version intentionally targets the easiest and most reliable deployment path:

- Standard Ultralytics YOLOv8 **object detection** model (horizontal boxes)
- Batch = 1
- One image input
- One prediction output
- Engine does **not** contain NMS
- Typical output `[1, 84, 8400]` or `[1, 8400, 84]`
- FP32 or FP16 input/output tensors
- Fixed 640x640 input, or dynamic H/W engine where the app chooses 640x640
- COCO 80 classes by default; replace `YoloDeploy.App\coco.names` for custom classes

Not targeted in this first version:

- OBB / segmentation / pose / classification
- INT8 input/output tensors
- Engines with EfficientNMS / multiple outputs
- Batch > 1
- Multiple inputs
- Custom preprocessing that differs from Ultralytics LetterBox + RGB + /255

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

1. Browse to a `.engine`.
2. Browse to an image.
3. Set confidence/NMS thresholds.
4. Click **执行检测**.
5. The image is decoded by WPF, converted to BGRA, sent to the native DLL, preprocessed, inferred by TensorRT, NMS-filtered, and boxes are rendered by WPF.

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

This project expects a standard raw YOLOv8 detection output. If your engine embeds NMS or has multiple output tensors, adapt `decodeOutput()`.

### Boxes are wrong

Check that the model uses standard Ultralytics preprocessing:
- LetterBox
- pad value 114
- BGR/RGB equivalent input is RGB
- divide by 255
- NCHW
