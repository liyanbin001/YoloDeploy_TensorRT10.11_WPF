# YoloDeploy.SDK.Runtime 一键客户发布

> 这个实现应合并到上一版已经加入 `YoloDeploy.SDK` 的仓库中。

当前 GitHub `main` 已经具备：
- `YoloDeploy.Native`
- ONNX -> TensorRT Engine
- GPU 信息查询
- 固定 W/H
- OBB
- 原 WPF 一键发布脚本

本包新增：
- `YoloDeploy.SDK.Test`
- `sdk_runtime_assets`
- `publish_sdk_runtime.ps1`
- `publish_sdk_runtime.bat`

## 最终目标

在开发电脑运行：

```text
publish_sdk_runtime.bat
```

自动得到：

```text
dist\
├─ YoloDeploy.SDK.Runtime\
│  ├─ YoloDeploy.SDK.dll
│  ├─ YoloDeploy.Native.dll
│  ├─ TestSDK.exe
│  ├─ TestSDK.dll
│  ├─ TestSDK.deps.json
│  ├─ TestSDK.runtimeconfig.json
│  ├─ nvinfer_10.dll
│  ├─ nvinfer_plugin_10.dll
│  ├─ nvonnxparser_10.dll
│  ├─ cudart64_*.dll
│  ├─ cublas64_*.dll
│  ├─ cublasLt64_*.dll
│  ├─ nvrtc64_*.dll
│  ├─ nvrtc-builtins64_*.dll
│  ├─ ...
│  ├─ Examples\
│  ├─ Models\
│  ├─ README_CUSTOMER_CN.txt
│  ├─ verify_runtime.bat
│  ├─ run_test_example.bat
│  ├─ runtime_manifest.json
│  └─ SHA256SUMS.txt
│
└─ YoloDeploy.SDK.Runtime.zip
```

## 前置条件

开发电脑需要已经满足原仓库的构建条件：

- VS2022
- Desktop development with C++
- .NET desktop development
- .NET 8 SDK
- TensorRT 10.11
- CUDA 12.3
- `TENSORRT_ROOT`
- `CUDA_PATH`

可选：

- `CUDNN_ROOT`

## 安装代码到仓库

把本包 `YoloDeploy\` 下内容合并到原：

```text
YoloDeploy_TensorRT10.11_WPF\YoloDeploy\
```

确保上一版存在：

```text
YoloDeploy.SDK\YoloDeploy.SDK.csproj
```

然后可以手工把 Test 项目加入 solution：

```powershell
dotnet sln YoloDeploy.sln add YoloDeploy.SDK.Test\YoloDeploy.SDK.Test.csproj
```

不过发布脚本直接按项目路径构建，因此即使不加入 `.sln` 也能使用。

## 发布

双击：

```text
publish_sdk_runtime.bat
```

或者：

```powershell
powershell -ExecutionPolicy Bypass -File .\publish_sdk_runtime.ps1
```

成功后把：

```text
dist\YoloDeploy.SDK.Runtime.zip
```

直接交给客户。

## 为什么我没有把 .engine 放入客户包

客户应该拿：

```text
best.onnx
classes.names
```

第一次由目标电脑根据：
- 当前 GPU
- TensorRT 版本
- 固定输入 W/H
- FP16/FP32

生成本机 Engine，并缓存到：

```text
%LOCALAPPDATA%\YoloDeploy\EngineCache
```

这样避免开发电脑生成的 `.engine` 在其他 GPU 上无法正常使用。

## 客户代码

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

foreach (var d in result.Detections)
{
    Console.WriteLine(
        $"{d.ClassName} {d.Confidence:F3} " +
        $"P1({d.P1.X:F1},{d.P1.Y:F1}) " +
        $"P2({d.P2.X:F1},{d.P2.Y:F1}) " +
        $"P3({d.P3.X:F1},{d.P3.Y:F1}) " +
        $"P4({d.P4.X:F1},{d.P4.Y:F1})");
}
```

## 客户运行依赖

本脚本会复制 TensorRT/CUDA 的用户态 DLL。

客户电脑仍需：
- NVIDIA GPU
- 兼容 NVIDIA 显卡驱动
- .NET 8 Windows Desktop Runtime x64

客户不需要：
- Python
- PyTorch
- Ultralytics
- VS2022
- CUDA Toolkit 开发环境
- TensorRT SDK 开发环境
- trtexec
