# Phase 3：一键 Release 发布

本阶段建立在：

- Phase 1：ONNX → TensorRT Engine
- Phase 2：GPU 信息 + Engine 缓存管理

之上。

目标是把开发机上的工程一次构建成可以复制到另一台 Windows + NVIDIA GPU 电脑测试的发布目录。

## 1. 一键入口

工程根目录新增：

```text
release_env.bat
publish_release.bat
publish_release.ps1
```

正常情况下只需要双击：

```text
publish_release.bat
```

## 2. release_env.bat

当前默认：

```text
TENSORRT_ROOT=D:\TensorRT-10.11.0.33\TensorRT-10.11.0.33

CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
```

如果以后修改安装目录，只需要修改这个文件。

如果已有全局环境变量，脚本优先使用已有值。

## 3. 发布流程

脚本自动：

```text
检查 CUDA / TensorRT / dotnet / VS2022
        ↓
找到 MSBuild.exe
        ↓
Release | x64 编译 YoloDeploy.Native
        ↓
dotnet publish
win-x64
self-contained
        ↓
复制 YoloDeploy.Native.dll
        ↓
复制 TensorRT DLL
        ↓
复制 CUDA runtime DLL
        ↓
创建 Models
        ↓
创建运行检查脚本
        ↓
创建发布 manifest
        ↓
生成 release folder
        ↓
生成 ZIP
```

## 4. .NET 发布配置

发布命令显式使用：

```text
-c Release
-r win-x64
--self-contained true
PublishSingleFile=false
PublishTrimmed=false
PublishReadyToRun=false
```

不使用 Single File 的原因：

项目本身必须携带：

```text
YoloDeploy.Native.dll
TensorRT DLL
CUDA DLL
ONNX
```

因此即便把 .NET 层压成一个 exe，最终仍然不是单文件产品。
保留目录结构反而更容易排查 TensorRT DLL 问题。

## 5. TensorRT DLL

脚本同时搜索：

```text
%TENSORRT_ROOT%\lib
%TENSORRT_ROOT%\bin
```

并复制：

```text
nvinfer*.dll
nvonnxparser*.dll
```

这样兼容：

- TensorRT 10.11 Windows ZIP 常见的 lib 目录布局
- 后续 TensorRT Windows DLL 移动到 bin 的目录布局

并强制检查：

```text
nvinfer_10.dll
nvinfer_plugin_10.dll
nvonnxparser_10.dll
```

## 6. CUDA DLL

直接依赖：

```text
cudart64_*.dll
```

此外为了让目标机执行 TensorRT Builder 时尽量不要求完整 CUDA Toolkit，
默认还会收集：

```text
cublas64_*.dll
cublasLt64_*.dll
nvrtc64_*.dll
nvrtc-builtins64_*.dll
cufft64_*.dll
curand64_*.dll
```

因此最终发布包可能比较大。

## 7. 可选 cuDNN

Phase 3 不强制要求 cuDNN。

如果你的特定模型/环境确实需要额外 cuDNN DLL，可以在：

```text
release_env.bat
```

配置：

```text
CUDNN_ROOT=...
```

脚本会额外复制：

```text
cudnn*.dll
```

## 8. 模型

把：

```text
*.onnx
```

放到工程根目录：

```text
models\
```

发布脚本自动复制到：

```text
dist\
YoloDeploy_v3_win-x64\
Models\
```

默认不会复制：

```text
*.engine
```

因为目标电脑应该根据自己的 GPU 重新构建 Engine。

## 9. 输出

成功后：

```text
dist\
├─ YoloDeploy_v3_win-x64\
│  ├─ YoloDeploy.App.exe
│  ├─ YoloDeploy.Native.dll
│  ├─ TensorRT DLL...
│  ├─ CUDA DLL...
│  ├─ coco.names
│  ├─ run_YoloDeploy.bat
│  ├─ verify_runtime.bat
│  ├─ DEPLOY_README_CN.txt
│  ├─ publish_manifest.json
│  └─ Models\
│
└─ YoloDeploy_v3_win-x64.zip
```

## 10. 目标电脑怎么使用

目标电脑需要：

```text
Windows x64
NVIDIA GPU
兼容 NVIDIA Driver
```

推荐安装：

```text
Microsoft Visual C++ 2015-2022 Redistributable x64
```

然后：

```text
解压 ZIP
↓
verify_runtime.bat
↓
确认没有 ERROR
↓
run_YoloDeploy.bat
↓
选择 Models 中 ONNX
↓
第一次生成/缓存 Engine
↓
加载模型
↓
检测图片
```

目标电脑不要求：

```text
Python
PyTorch
Ultralytics
Visual Studio
.NET 8 Desktop Runtime
```

因为 .NET 已采用 self-contained 发布。

## 11. 为什么 run_YoloDeploy.bat 比直接双击 exe 更推荐

它先执行：

```text
PATH = 发布目录 + 原 PATH
```

只影响当前程序进程。

这样 TensorRT 的内部/辅助 DLL 更容易从软件目录被找到，
且不会修改目标电脑系统环境变量。

## 12. verify_runtime.bat

它会检查：

```text
YoloDeploy.App.exe
YoloDeploy.Native.dll
nvinfer_10.dll
nvinfer_plugin_10.dll
nvonnxparser_10.dll
cudart64_*.dll
nvidia-smi
VC++ Runtime
```

注意这只是“基础依赖检查”。

最终验证仍然是：

```text
打开程序
→ 从 ONNX 成功生成 Engine
→ 加载 Engine
→ 成功检测图片
```

## 13. publish_manifest.json

发布包记录：

- 发布时间
- RuntimeIdentifier
- self-contained 状态
- Git commit（如果 git 可用）
- 构建时 TensorRT/CUDA 路径
- 关键 EXE/DLL 大小
- SHA-256

方便以后判断客户机器使用的是哪一版发布包。

## 14. 发布包不是“万能 NVIDIA 环境”

虽然脚本带了 TensorRT 和常用 CUDA 用户态运行 DLL，
但 NVIDIA 内核驱动无法跟应用目录一起简单打包。

因此目标机仍必须安装兼容 NVIDIA Driver。

对于特殊模型，如果 TensorRT Builder 报告缺少额外 NVIDIA 运行 DLL，
应该按该模型实际依赖补充，而不是把整个 CUDA Toolkit 开发目录原样发送。

## 15. 最推荐测试

先在一台没有：

```text
Visual Studio
Python
PyTorch
.NET 8 SDK
TensorRT SDK
```

但有：

```text
NVIDIA GPU + Driver
```

的 Windows x64 电脑测试。

如果：

```text
verify_runtime
→ 打开软件
→ ONNX 建 Engine
→ 缓存
→ 推理
```

全部成功，就说明 Phase 3 发布链路真正跑通。
