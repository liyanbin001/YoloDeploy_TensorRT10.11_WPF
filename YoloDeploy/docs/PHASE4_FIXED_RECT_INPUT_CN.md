# Phase 4：工业现场固定矩形输入（宽度 + 高度）

本版本基于已经验证可部署的 Phase 3，目标只有一个：

```text
原来的单一“输入尺寸 640”
                ↓
固定输入宽度 + 固定输入高度
```

例如：

```text
固定输入宽度：1280
固定输入高度：512
```

本版本不增加动态尺寸范围，不提供 MIN / OPT / MAX 三组 UI。
工业现场仍按“一套模型对应一套固定网络输入尺寸”使用。

---

## 一、为什么这样改

工业相机图像经常不是正方形，例如：

```text
2048 × 512
2448 × 2048
1280 × 512
1920 × 640
```

如果强制所有模型都使用：

```text
640 × 640
```

可能产生大量 LetterBox 填充，并减少有效目标像素。

因此现在软件允许使用矩形网络输入，例如：

```text
1280 × 512
1024 × 768
1920 × 640
```

---

## 二、必须区分两个概念

### 1. 原始工业图像尺寸

例如相机固定输出：

```text
2048 × 512
```

### 2. YOLO 网络固定输入尺寸

例如：

```text
1280 × 320
```

二者不一定相同。

当前程序仍然会执行：

```text
原图
↓
保持宽高比缩放
↓
LetterBox
↓
固定网络输入 W × H
```

如果相机图像本身刚好与网络输入相同，例如都是：

```text
1280 × 512
```

则缩放比例为 1，基本不会产生额外 LetterBox padding。

---

## 三、界面变化

原来：

```text
输入尺寸：[640]
```

现在：

```text
固定输入宽度：[1280]
固定输入高度：[512]
```

下方手动加载 Engine 的区域也从：

```text
动态输入尺寸：[640]
```

改成：

```text
Engine宽度：[1280]
Engine高度：[512]
```

从 ONNX 构建完成后，这两个值会自动同步。

---

## 四、C# 到 Native 的参数

Phase 3 实际已经有 Native API：

```cpp
YoloBuildEngineFromOnnx(
    ...,
    int32_t inputWidth,
    int32_t inputHeight,
    ...
)
```

Phase 4 只是把 WPF 真正改成分别传递：

```csharp
inputWidth,
inputHeight
```

而不是过去的：

```csharp
inputSize,
inputSize
```

Engine 加载同样改成：

```csharp
YoloCreate(
    enginePath,
    inputWidth,
    inputHeight,
    ...
);
```

---

## 五、固定 ONNX 的处理方式

工业现场最推荐：

```text
ONNX 本身就是固定 shape
```

例如：

```text
[1,3,512,1280]
```

注意 NCHW 顺序：

```text
N = 1
C = 3
H = 512
W = 1280
```

界面应填写：

```text
固定输入宽度：1280
固定输入高度：512
```

Phase 4 会验证：

```text
ONNX shape
==
UI requested shape
```

匹配：

```text
[1,3,512,1280]
==
[1,3,512,1280]

→ 允许生成 Engine
```

不匹配，例如：

```text
ONNX = [1,3,512,1280]
UI   = [1,3,640,640]
```

程序会直接报错：

```text
Fixed ONNX input shape mismatch
```

并提示重新导出 ONNX 或修改界面宽高。

这样可以防止工业现场误以为“填写宽高会强行改变固定 ONNX”。

---

## 六、动态 ONNX 也可以使用，但 Engine 仍固定

如果 ONNX 是：

```text
[1,3,-1,-1]
```

仍然可以选择：

```text
W = 1280
H = 512
```

程序创建：

```text
MIN = [1,3,512,1280]
OPT = [1,3,512,1280]
MAX = [1,3,512,1280]
```

因此最终虽然来源是 dynamic ONNX，
但这个 Engine 在本软件中仍然按固定：

```text
1280 × 512
```

使用。

这与工业现场固定尺寸的目标一致。

---

## 七、缓存已经天然支持矩形尺寸

`EngineCacheManager.cs` 原本已经分别保存：

```text
InputWidth
InputHeight
```

Cache Key 也是：

```text
{width}x{height}
```

因此：

```text
1280x512
```

和：

```text
640x640
```

会生成完全不同的 Engine 缓存。

例如：

```text
...fp16_1280x512_ws2048.engine
```

---

## 八、预处理无需重写

现有 `YoloBridge.cpp` 的 LetterBox 已经分别使用：

```cpp
dstW
dstH
```

缩放比例：

```text
r = min(dstW / srcW, dstH / srcH)
```

因此它本身已经支持矩形网络输入。

例如：

```text
原图 2048 × 512
网络 1280 × 512
```

程序会保持原图宽高比进行缩放和 padding，
不会强制把图像拉伸成正方形。

---

## 九、坐标恢复无需重写

现有检测后处理会使用 LetterBox 的：

```text
scale
left
top
```

把 Engine 输出坐标恢复到原始图像空间。

因此网络改成：

```text
1280 × 512
```

不会要求重新设计检测框坐标映射。

---

## 十、推荐工业模型导出方式

对于固定工业现场，推荐直接导出固定 ONNX。

例如网络需要：

```text
W = 1280
H = 512
```

Ultralytics 中 `imgsz` 使用：

```text
(height, width)
```

因此概念上应为：

```python
model.export(
    format="onnx",
    imgsz=(512, 1280),
    batch=1,
    dynamic=False
)
```

导出后确认 ONNX 输入：

```text
[1,3,512,1280]
```

再在软件中填写：

```text
固定输入宽度 = 1280
固定输入高度 = 512
```

---

## 十一、尺寸是否必须是 32 的整数倍

对于标准 YOLOv8，推荐优先使用与模型 stride 兼容的尺寸，
通常最大 stride 为 32。

常见良好示例：

```text
1280 × 512
1024 × 768
1920 × 640
1280 × 640
```

但 Phase 4 没有在 UI 中强制 `% 32 == 0`。

原因是工业项目可能存在：

```text
自定义网络
自定义 stride
固定 ONNX 已经定义特殊尺寸
```

最终应以实际训练 / ONNX 模型要求为准。

---

## 十二、使用流程

### 固定 ONNX

```text
启动软件
↓
读取 GPU
↓
选择固定 ONNX
↓
输入固定宽度
↓
输入固定高度
↓
选择 FP32 / FP16
↓
生成 / 使用 Engine
↓
固定 ONNX shape 一致性检查
↓
生成 Engine
↓
缓存
↓
加载 Engine
↓
选择工业图像
↓
执行检测
```

### 第二次

相同：

```text
ONNX
GPU
TensorRT
FP32/FP16
宽度
高度
Workspace
```

会命中同一个缓存。

---

## 十三、手动加载 Engine

如果直接加载已有 Engine：

```text
Engine宽度
Engine高度
```

仅在 Engine 是动态 shape 时用于解析具体输入 shape。

如果 Engine 本身已经固定：

```text
[1,3,512,1280]
```

Native 会优先使用 Engine 自己的固定 shape，
界面宽高不会改变固定 Engine。

为了减少现场误操作，仍建议填写实际 Engine 对应的宽、高。

---

## 十四、Phase 3 一键发布仍然保留

仍然可以：

```text
publish_release.bat
```

一键发布。

Phase 4 默认发布包名称改为：

```text
YoloDeploy_v5_detect-obb_win-x64
```

发布后的目标电脑使用方式不变。

---

## 十五、推荐现场配置方式

对于真正固定的工业视觉项目，建议一套模型固定以下内容：

```text
ONNX
固定 W
固定 H
FP32 / FP16
类别文件
置信度
NMS
```

例如：

```text
模型：surface_defect.onnx
输入：1280 × 512
精度：FP16
Confidence：0.35
NMS：0.45
```

上线后不要让操作人员频繁修改尺寸。

如果之后更换成另一种检测尺寸：

```text
2048 × 512
```

更推荐：

```text
重新导出对应固定 ONNX
↓
重新生成对应 Engine
↓
形成新的 Engine 缓存
```

而不是在同一个固定 ONNX 上随意修改 W/H。
