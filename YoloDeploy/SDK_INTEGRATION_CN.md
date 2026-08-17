# 集成到原 GitHub 仓库

将本压缩包中的 `YoloDeploy` 内容覆盖/合并到原仓库的：

```text
YoloDeploy_TensorRT10.11_WPF/
└─ YoloDeploy/
```

得到：

```text
YoloDeploy/
├─ YoloDeploy.App/
├─ YoloDeploy.Native/
│  ├─ YoloBridge.h
│  ├─ YoloBridge.cpp
│  ├─ OnnxEngineBuilder.cpp
│  ├─ SystemInfo.cpp
│  └─ YoloDeploy.Native.vcxproj
├─ YoloDeploy.SDK/              ← 新增
├─ YoloDeploy.SDK.Demo/         ← 新增
├─ add_sdk_to_solution.bat      ← 新增
├─ publish_sdk.ps1              ← 新增
└─ YoloDeploy.sln
```

## 1. Native 层

以当前 GitHub `main` 为基准，无需修改 Native C++。

它已经提供 SDK 所需的五项关键能力：

- `YoloBuildEngineFromOnnx`
- `YoloGetGpuInfoJson`
- `YoloCreate`
- `YoloGetTaskHint`
- `YoloDetectObbBgra`
- `YoloDestroy`

并且 `YoloObbDetection` 已返回中心点、宽高、角度、类别、置信度和四角点。

## 2. 加入解决方案

双击：

```text
add_sdk_to_solution.bat
```

或在仓库 `YoloDeploy` 目录执行：

```powershell
dotnet sln YoloDeploy.sln add YoloDeploy.SDK\YoloDeploy.SDK.csproj
dotnet sln YoloDeploy.sln add YoloDeploy.SDK.Demo\YoloDeploy.SDK.Demo.csproj
```

## 3. VS2022

选择：

```text
Release | x64
```

先确保 `YoloDeploy.Native` 能成功构建，再构建整个解决方案。

## 4. 测试

运行 Demo：

```powershell
YoloDeploy.SDK.Demo.exe `
  D:\Model\best.onnx `
  D:\Model\classes.names `
  D:\Images\001.jpg `
  1280 `
  512
```

第一次：
- ONNX -> engine
- 写入 cache
- 加载 engine
- OBB 推理

第二次：
- cache hit
- 直接加载 engine
- OBB 推理

## 5. 发布 SDK runtime

在开发机已设置 `TENSORRT_ROOT`、`CUDA_PATH` 后：

```powershell
powershell -ExecutionPolicy Bypass -File .\publish_sdk.ps1
```

输出：

```text
dist\YoloDeploy.SDK_win-x64\
├─ YoloDeploy.SDK.dll
├─ YoloDeploy.Native.dll
├─ nvinfer_10.dll
├─ nvinfer_plugin_10.dll
├─ nvonnxparser_10.dll
├─ nvinfer*.dll
├─ cudart64_*.dll
├─ cublas*.dll
├─ nvrtc*.dll
├─ ...
└─ SDK_DEPLOY_README_CN.txt
```

第三方项目只在代码层引用 `YoloDeploy.SDK.dll`，但部署时必须保留整个 Runtime 文件夹。

## 6. 第三方最终代码

```csharp
using YoloDeploy.SDK;

using var detector = new ObbDetector(new ObbDetectorOptions
{
    ModelPath = @"D:\Model\best.onnx",
    ClassNamesPath = @"D:\Model\classes.names",
    InputWidth = 1280,
    InputHeight = 512,
    EnableFp16 = true
});

var result = detector.Detect(
    @"D:\Images",
    "001.jpg");

foreach (var item in result.Detections)
{
    Console.WriteLine(
        $"{item.ClassName}, "
        + $"{item.Confidence:F3}, "
        + $"({item.P1.X:F1},{item.P1.Y:F1}), "
        + $"({item.P2.X:F1},{item.P2.Y:F1}), "
        + $"({item.P3.X:F1},{item.P3.Y:F1}), "
        + $"({item.P4.X:F1},{item.P4.Y:F1})");
}
```

## 7. 模型导出建议

第一版建议导出标准 raw-output OBB ONNX，固定 batch=1，固定输入尺寸，例如：

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="onnx",
    imgsz=(512, 1280),
    batch=1,
    dynamic=False,
    nms=False,
    simplify=True,
)
```

如果 ONNX 导出为 dynamic，本仓库 Native Builder 也会将指定 W/H 作为
`MIN=OPT=MAX` 构建成固定尺寸的 TensorRT engine。

注意：具体 Ultralytics 版本的导出参数请以你训练/导出环境实际版本为准。
