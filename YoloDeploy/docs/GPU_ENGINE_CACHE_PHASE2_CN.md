# Phase 2：GPU 信息 + Engine 缓存管理

本版本是在已经跑通的 Phase 1 基础上继续增加：

```text
ONNX → Engine
Engine → YOLOv8 Detect 推理
WPF 显示结果
```

现有 `YoloBridge.cpp` 推理核心和 `OnnxEngineBuilder.cpp` 构建核心均保留。

## 一、启动后新增 GPU / Runtime 信息

Native DLL 新增：

```cpp
YoloGetGpuInfoJson(...)
```

它通过 CUDA Runtime 查询：

```cpp
cudaGetDeviceCount()
cudaGetDevice()
cudaGetDeviceProperties()
cudaRuntimeGetVersion()
cudaDriverGetVersion()
```

并通过 TensorRT：

```cpp
NvInferVersion.h
NV_TENSORRT_MAJOR
NV_TENSORRT_MINOR
NV_TENSORRT_PATCH
NV_TENSORRT_BUILD
```

返回类似：

```text
NVIDIA GeForce RTX 2080 SUPER
CC 7.5
8.0 GiB
SM 48
CUDA Runtime 12.3
Driver API 13.1
TensorRT 10.11.0.33
```

## 二、缓存目录

默认缓存位置：

```text
%LOCALAPPDATA%\YoloDeploy\EngineCache
```

通常例如：

```text
C:\Users\<用户名>\AppData\Local\YoloDeploy\EngineCache
```

选择 LocalAppData 而不是 EXE 目录，是为了以后程序安装到 Program Files
时普通用户仍然具有缓存写入权限。

## 三、缓存 Key

缓存文件名会包含：

- ONNX 内容 SHA-256 前 16 位
- GPU 型号
- Compute Capability
- SM 数量
- TensorRT 版本 + build
- FP32 / FP16
- 输入宽高
- Workspace MiB

例如：

```text
yolov8n_12ab34cd56ef7890_NVIDIA_GeForce_RTX_2080_SUPER_cc75_sm48_trt10_11_0_33_fp16_640x640_ws2048.engine
```

旁边同时保存：

```text
...engine.json
```

## 四、为什么用 ONNX SHA-256

如果你重新训练后仍然把模型命名为：

```text
best.onnx
```

仅靠文件名无法知道权重已经变化。

因此程序对 ONNX 文件内容计算 SHA-256。

模型内容只要变化：

```text
SHA-256 变化
→ Cache Key 变化
→ 自动生成新的 Engine
```

不会误用旧 Engine。

## 五、哪些条件变化会重建

以下任意条件变化都会得到新的缓存：

```text
ONNX 内容
GPU 型号
Compute Capability
SM 数量
TensorRT 版本
FP32 / FP16
输入尺寸
Workspace
```

## 六、驱动版本为什么不放进 Key

CUDA Driver / CUDA Runtime 会保存到 JSON 元数据用于排查问题，
但不会加入缓存文件名。

原因：

```text
一次正常兼容的 NVIDIA 驱动升级
```

不应该仅因为驱动小版本变化就自动重建所有 TensorRT Engine。

如果升级后旧 Engine 确实加载异常，可以勾选：

```text
强制重新构建
```

重新构建当前配置。

## 七、第一次运行

推荐：

```text
标准 yolov8n.onnx
输入 640
FP32
Workspace 2048
使用本机 Engine 缓存 = 开
强制重新构建 = 关
```

选择 ONNX 后界面先计算 SHA-256。

第一次应看到：

```text
缓存未命中
```

点击：

```text
生成 / 使用 Engine
```

程序执行 TensorRT Builder，完成后：

```text
写 .engine
写 .engine.json
自动把缓存 Engine 填入原来的 Engine 路径
```

之后点击：

```text
加载模型
```

再执行图片检测。

## 八、第二次运行

关闭软件再打开。

选择相同：

```text
ONNX
GPU
TensorRT
精度
尺寸
Workspace
```

界面应该显示：

```text
缓存命中
```

点击：

```text
生成 / 使用 Engine
```

不会再调用 TensorRT Builder，而是立即：

```text
EnginePathTextBox = 已有缓存 Engine
```

## 九、换 FP16

从：

```text
FP32
```

切换：

```text
FP16
```

缓存 Key 会变化。

第一次 FP16 会重新生成 Engine；
之后同配置再次运行会命中 FP16 缓存。

## 十、换另一台 GPU

例如开发机：

```text
RTX 2080 SUPER
CC 7.5
```

目标机：

```text
RTX 3060
CC 8.6
```

GPU 信息不同，因此 Cache Key 自动不同。

程序不会把 2080S 的缓存 Engine 当作 3060 的本机 Engine；
目标机第一次运行会基于 ONNX 自己构建。

## 十一、缓存管理按钮

### 打开缓存目录

直接打开：

```text
%LOCALAPPDATA%\YoloDeploy\EngineCache
```

### 清空缓存

删除整个 EngineCache 内容并重新创建目录。

不会删除：

```text
原始 ONNX
手动保存到其他目录的 Engine
```

### 强制重新构建

缓存已经有效时仍然执行 TensorRT Builder，
覆盖同一个缓存 Key 的 Engine 和 JSON 元数据。

## 十二、手动输出 Engine

如果取消勾选：

```text
使用本机 Engine 缓存
```

界面恢复 Phase 1 行为：

```text
Engine 输出 → 可以手动另存为
```

此时不会写缓存 JSON。

如果点击“另存为...”，程序也会自动关闭缓存模式。

## 十三、新增文件

Native：

```text
YoloDeploy.Native\SystemInfo.cpp
```

C#：

```text
YoloDeploy.App\GpuInfo.cs
YoloDeploy.App\EngineCacheManager.cs
```

## 十四、修改文件

```text
YoloDeploy.Native\YoloBridge.h
YoloDeploy.Native\YoloDeploy.Native.vcxproj

YoloDeploy.App\NativeMethods.cs
YoloDeploy.App\MainWindow.xaml
YoloDeploy.App\MainWindow.xaml.cs

README.md
```

## 十五、没有新增第三方依赖

Phase 2 本身没有新增外部库。

仍然使用已有：

```text
nvinfer_10.lib
nvinfer_plugin_10.lib
nvonnxparser_10.lib
cudart.lib
```

因此你现有 CUDA 12.3 + TensorRT 10.11 环境继续使用即可。

## 十六、重新编译

Visual Studio：

```text
Release | x64
```

推荐仍然：

```text
YoloDeploy.Native
→ 右键 → 生成
```

确认：

```text
artifacts\native\Release\YoloDeploy.Native.dll
```

然后：

```text
YoloDeploy.App
→ 右键 → 生成
```

最后：

```text
YoloDeploy.App
→ 设为启动项目
→ Ctrl + F5
```

## 十七、重点验证顺序

1. 顶部能否正确显示 GPU 信息。
2. 选择 ONNX 后是否出现“缓存未命中”。
3. 第一次生成 Engine 是否成功。
4. `.engine.json` 是否同时生成。
5. 再次选择相同模型是否显示“缓存命中”。
6. 点击“生成 / 使用 Engine”是否快速返回。
7. 加载缓存 Engine 是否成功。
8. 图片检测是否仍与 Phase 1 一致。
9. 切换 FP16 后是否变成新的缓存。
10. “清空缓存”后是否重新显示未命中。
