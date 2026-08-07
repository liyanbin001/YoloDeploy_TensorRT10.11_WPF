# Phase 1：在现有 YoloDeploy 上增加 ONNX → Engine

此补丁针对仓库：

https://github.com/liyanbin001/YoloDeploy_TensorRT10.11_WPF

基线目录：

YoloDeploy/
- YoloDeploy.App
- YoloDeploy.Native

## 修改内容

### 新增

`YoloDeploy.Native/OnnxEngineBuilder.cpp`

实现：

ONNX
→ NvOnnxParser
→ TensorRT INetworkDefinition
→ IBuilderConfig
→ Optimization Profile（动态输入时）
→ buildSerializedNetwork()
→ 保存 `.engine`

### 替换

- `YoloDeploy.Native/YoloBridge.h`
- `YoloDeploy.Native/YoloDeploy.Native.vcxproj`
- `YoloDeploy.App/NativeMethods.cs`
- `YoloDeploy.App/MainWindow.xaml`
- `YoloDeploy.App/MainWindow.xaml.cs`

原有 `YoloBridge.cpp` 不需要修改。

---

# 1. 先检查 TensorRT ONNX Parser

你的 TensorRT 根目录应存在：

`%TENSORRT_ROOT%\include\NvOnnxParser.h`

以及链接库：

`%TENSORRT_ROOT%\lib\nvonnxparser_10.lib`

在 CMD 中检查：

```bat
dir "%TENSORRT_ROOT%\include\NvOnnxParser.h"
dir "%TENSORRT_ROOT%\lib\nvonnxparser_10.lib"
where /r "%TENSORRT_ROOT%" nvonnxparser_10.dll
```

如果找不到 `nvonnxparser_10.lib`，不要继续编译。

---

# 2. 为什么 vcxproj 要增加 nvonnxparser_10.lib

原工程链接：

```text
nvinfer_10.lib
nvinfer_plugin_10.lib
cudart.lib
```

Phase 1 增加：

```text
nvonnxparser_10.lib
```

所以新的链接项是：

```text
nvinfer_10.lib
nvinfer_plugin_10.lib
nvonnxparser_10.lib
cudart.lib
```

---

# 3. 覆盖文件

推荐先提交当前能运行版本：

```bat
git add .
git commit -m "baseline: working TensorRT engine inference"
```

再把补丁包内 `YoloDeploy` 目录覆盖到仓库中的 `YoloDeploy`。

`YoloBridge.cpp` 不覆盖、不删除。

---

# 4. 重新生成

Visual Studio：

```text
Release | x64
```

先：

```text
YoloDeploy.Native
右键
生成
```

成功后确认：

```text
YoloDeploy\artifacts\native\Release\YoloDeploy.Native.dll
```

然后：

```text
YoloDeploy.App
右键
生成
```

---

# 5. 新的运行时依赖

因为 `YoloDeploy.Native.dll` 现在静态链接了 ONNX Parser import library，
所以运行时还要求 Windows 能找到：

```text
nvonnxparser_10.dll
```

开发机可以把 TensorRT DLL 目录加入 PATH。

后续“一键发布 Release”阶段会把它自动复制到发布目录。

---

# 6. 第一次建议使用标准 YOLOv8 Detect ONNX

第一轮建议：

- batch = 1
- 640 × 640
- 标准 YOLOv8 Detect
- 单输入
- 单输出
- 不包含 NMS
- FP32

例如通过 Ultralytics 导出：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
)
```

然后在软件中：

1. 选择 `.onnx`
2. Engine 输出路径
3. 输入尺寸 640
4. 精度 FP32
5. Workspace 2048 MiB
6. 点击“生成 Engine”

构建完成后，新 Engine 路径会自动填入原来的 Engine 输入框。

接着：

1. 点击“加载模型”
2. 确认 Input/Output shape
3. 选择图片
4. 点击“执行检测”

---

# 7. 动态 ONNX

如果 ONNX 输入类似：

```text
[-1, 3, -1, -1]
```

Phase 1 会创建一个 optimization profile：

```text
MIN = OPT = MAX = [1, 3, 640, 640]
```

其中 640 来自界面的“输入尺寸”。

因此第一阶段生成的 Engine 虽然来自动态 ONNX，但 profile 只允许这一种尺寸。

这是有意设计的：
先保证当前已有的推理逻辑稳定。

以后 Phase 2 可以扩展为：

```text
MIN 320
OPT 640
MAX 1280
```

真正支持动态范围。

---

# 8. 固定 ONNX

如果 ONNX 已经是：

```text
[1, 3, 640, 640]
```

界面的输入尺寸不会强制改变 ONNX。

TensorRT 会按模型固定尺寸生成 Engine。

---

# 9. FP16

2080 Super 支持 FP16。

但第一轮建议先用：

```text
FP32
```

确认：

ONNX → Engine → 加载 → 推理

整条链路正确。

之后切：

```text
FP16
```

生成另一个 Engine 比较速度和结果。

TensorRT 10.x 的 FP16 builder flag 会允许 TensorRT 使用 FP16 实现；
部分层仍可能保留 FP32。

---

# 10. 常见错误

## LNK1104: cannot open file nvonnxparser_10.lib

说明：

`%TENSORRT_ROOT%\lib`

或者 `TENSORRT_ROOT` 配置错误。

## 无法加载 YoloDeploy.Native.dll

新增功能后可能实际上缺的是：

`nvonnxparser_10.dll`

检查：

```bat
where /r "%TENSORRT_ROOT%" nvonnxparser_10.dll
```

并把所在目录加入 PATH。

## ONNX parser failed

看软件顶部的“构建日志”。

TensorRT Parser 会返回具体不支持的节点/算子。

## buildSerializedNetwork failed

常见原因：

- Workspace 太小
- ONNX 有不支持算子
- 动态 shape 没有合法 profile
- GPU 显存不足
- 自定义插件缺失

先把 Workspace 设置为：

```text
2048
```

或：

```text
4096
```

---

# 11. 当前 Phase 1 的边界

本次只保证标准检测模型最小闭环：

```text
ONNX
↓
Engine
↓
现有 YOLOv8 Detect 推理
```

暂不支持：

- OBB
- Seg
- Pose
- 多输入
- 多输出
- Engine 内置 NMS
- INT8
- Calibration
- 自定义 TensorRT Plugin
- 动态 MIN/OPT/MAX 三档 UI
- 自动 GPU/Engine cache 命名

这些建议后续逐步增加。
