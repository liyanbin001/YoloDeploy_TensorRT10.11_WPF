# YoloDeploy.SDK — OBB 第一版

本目录直接放到原仓库 `YoloDeploy/` 下。

## 目标 API

调用者只需要：

1. `best.onnx`
2. `classes.names`
3. 图片目录 + 图片名称
4. 完整 SDK Runtime 文件夹

不需要手动调用 `trtexec`，也不需要自己生成 `.engine`。

```csharp
using YoloDeploy.SDK;

using var detector = new ObbDetector(new ObbDetectorOptions
{
    ModelPath = @"D:\Model\best.onnx",
    ClassNamesPath = @"D:\Model\classes.names",
    InputWidth = 1280,
    InputHeight = 512,
    EnableFp16 = true,
    WorkspaceMiB = 1024
});

ObbDetectionResponse result =
    detector.Detect(@"D:\Images", "001.jpg");

foreach (ObbResult box in result.Detections)
{
    Console.WriteLine(
        $"{box.ClassName} {box.Confidence:F3} "
        + $"P1=({box.P1.X:F1},{box.P1.Y:F1}) "
        + $"P2=({box.P2.X:F1},{box.P2.Y:F1}) "
        + $"P3=({box.P3.X:F1},{box.P3.Y:F1}) "
        + $"P4=({box.P4.X:F1},{box.P4.Y:F1})");
}
```

## Engine 行为

ONNX 模式：

- 查询当前 GPU / CUDA / TensorRT
- SHA-256 计算 ONNX 内容
- 生成 GPU/TensorRT/尺寸/精度相关的 cache key
- 查找 `%LOCALAPPDATA%\YoloDeploy\EngineCache`
- 命中：直接加载
- 未命中：调用 `YoloBuildEngineFromOnnx`
- 固定 input W/H
- 保存 `.engine + .engine.json`
- 调用 `YoloCreate`
- 验证 task hint 必须为 OBB

`.engine` 模式仍然保留，便于调试，但对最终客户建议只公开 ONNX。

## 固定尺寸规则

`InputWidth/InputHeight` 在一个 detector 生命周期内固定。

- 固定 shape ONNX：必须与设置完全一致。
- dynamic ONNX：Native Builder 使用 `MIN=OPT=MAX=[1,3,H,W]`，最终仍是固定尺寸 engine。
- 原始图片不要求恰好等于该尺寸；现有 Native 层会 LetterBox 到模型输入尺寸。

## 与当前 GitHub main 的关系

当前 main 已经包含：

- `YoloBuildEngineFromOnnx`
- `YoloGetGpuInfoJson`
- `YoloDetectObbBgra`
- OBB 四角点返回
- 固定矩形 input
- GPU-specific Engine cache（原先位于 WPF App）

因此当前 main 的 `YoloDeploy.Native` 不需要为这个 SDK 再修改 C++。
本 SDK 是把 WPF 中的模型准备/缓存逻辑抽成可复用类库。

## 开发机

需要：

- Windows x64
- VS2022 C++ / .NET desktop workloads
- TensorRT 10.11
- CUDA 12.3（与原仓库一致）
- 环境变量 `TENSORRT_ROOT`
- 环境变量 `CUDA_PATH`

## 目标机

目标机不需要：

- Python
- PyTorch
- Ultralytics
- Visual Studio
- 手动生成 engine

但仍需要：

- Windows x64
- NVIDIA GPU
- 兼容 NVIDIA Driver
- 完整的 SDK Runtime 原生依赖目录
- 如果宿主程序不是 self-contained，则需要相应 .NET 8 Windows Desktop Runtime

> `YoloDeploy.SDK.dll` 是对外唯一的托管 API 引用，不代表物理运行时真的只需要一个文件。
