# Phase 5：普通 Detect + 旋转框 OBB 双模式部署

本版本基于已经成功部署的 Phase 4。

目标：

```text
同一个 WPF / TensorRT 部署程序
        ├─ 普通目标检测 Detect
        └─ 旋转框检测 OBB
```

不删除原有 Detect，OBB 使用独立后处理接口。

---

## 1. 为什么原程序不能直接处理 OBB

原来的水平框 Detect 输出按：

```text
[x, y, w, h, class0, class1, ...]
```

解析。

输出通道：

```text
4 + class_count
```

例如 3 类：

```text
7 channels
```

而标准 Ultralytics OBB raw output 是：

```text
[x, y, w, h, class0, class1, ..., angle]
```

输出通道：

```text
5 + class_count
```

例如 3 类：

```text
8 channels
```

因此不能继续使用：

```cpp
classCount = channels - 4;
```

来解析 OBB。

---

## 2. OBB 内部表示

本程序按照：

```text
xywhr
```

处理旋转框：

```text
centerX
centerY
width
height
rotation
```

其中 rotation 为弧度。

Native 最终同时返回四个角点：

```text
P1
P2
P3
P4
```

给 WPF 直接绘制 Polygon。

---

## 3. 新增 Native 返回结构

```cpp
struct YoloObbDetection
{
    float centerX;
    float centerY;
    float width;
    float height;
    float angleRadians;
    float score;
    int32_t classId;

    float p1x, p1y;
    float p2x, p2y;
    float p3x, p3y;
    float p4x, p4y;
};
```

普通 Detect 的：

```cpp
YoloDetection
```

保持不变。

---

## 4. 新增 Native API

```cpp
YoloDetectObbBgra(...)
```

普通 Detect 继续：

```cpp
YoloDetectBgra(...)
```

因此二者互不影响。

---

## 5. OBB 解码

OBB raw output 支持：

```text
[1, C, N]
```

或：

```text
[1, N, C]
```

其中：

```text
C = 5 + class_count
```

解码：

```text
channel 0 = center x
channel 1 = center y
channel 2 = width
channel 3 = height
channel 4 ... = class probabilities
last channel = angle
```

---

## 6. LetterBox 坐标恢复

预处理仍然使用 Phase 4 已有的固定矩形 LetterBox：

```text
原始工业图像
↓
保持宽高比
↓
LetterBox 到固定 W × H
```

OBB 恢复：

```text
centerX = (rawCenterX - left) / scale
centerY = (rawCenterY - top) / scale

width  = rawWidth  / scale
height = rawHeight / scale

angle = angle
```

因为 LetterBox 是统一比例缩放 + 平移，所以旋转角本身不需要缩放。

---

## 7. 为什么不能继续使用普通 IoU NMS

水平框 NMS 使用：

```text
x1, y1, x2, y2
```

计算 axis-aligned IoU。

对于旋转目标，这会把旋转框外接水平矩形当成真实检测框，
导致两个倾斜目标的重叠关系判断失真。

Phase 5 OBB 改为：

```text
ProbIoU
+
rotated NMS
```

对：

```text
centerX, centerY, width, height, angle
```

直接计算旋转框相似度。

程序仍然保持 class-aware：

```text
不同类别不会互相抑制
```

---

## 8. WPF 绘制

Detect：

```text
Rectangle
```

OBB：

```text
Polygon
P1 → P2 → P3 → P4
```

OBB 标签显示：

```text
类别
置信度
角度
```

结果表显示：

```text
中心坐标
宽 × 高
角度
```

角度显示时：

```text
degree = radians × 180 / π
```

---

## 9. 模型任务选择

界面增加：

```text
模型任务：
[普通检测 Detect]
[旋转框 OBB]
```

加载 Engine 后程序会结合：

```text
Engine输出通道数
+
coco.names类别数
```

自动判断。

例如类别数为 3：

```text
channels == 7
→ Detect

channels == 8
→ OBB
```

如果无法判断，会保留人工选择。

---

## 10. 类别文件非常重要

自训练 OBB 模型必须修改：

```text
coco.names
```

例如你的工业模型有 3 类：

```text
scratch
crack
dent
```

则文件必须是：

```text
scratch
crack
dent
```

数量和训练顺序必须完全相同。

如果 Engine 解析得到：

```text
3 classes
```

但 `coco.names` 有 80 行，
Phase 5 OBB 会直接报 class count mismatch，
避免错误显示类别。

---

## 11. OBB ONNX 导出建议

对于工业固定尺寸，推荐：

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=(512, 1280),
    batch=1,
    dynamic=False
)
```

其中 Ultralytics `imgsz` 为：

```text
(height, width)
```

所以：

```text
imgsz=(512, 1280)
```

对应程序：

```text
固定输入宽度 = 1280
固定输入高度 = 512
```

关键要求是导出标准 raw-output OBB。

如果你当前 Ultralytics 版本提供 `nms` 导出参数，
本程序路线应保持：

```text
nms=False
```

即不要把 NMS 烘焙进 ONNX/Engine。

---

## 12. Engine 构建、GPU 缓存不需要分叉

OBB 的 ONNX 仍通过同一套：

```text
NvOnnxParser
→ TensorRT Builder
→ serialized Engine
```

GPU 缓存仍使用：

```text
ONNX SHA-256
GPU
Compute Capability
TensorRT版本
FP32 / FP16
输入宽高
Workspace
```

所以不需要单独创建 OBB Cache Manager。

因为 OBB ONNX 内容不同，SHA-256 本身也会不同。

---

## 13. 一键 Release 继续使用

仍运行：

```text
publish_release.bat
```

默认输出包名升级为：

```text
YoloDeploy_v5_detect-obb_win-x64
```

TensorRT/CUDA Runtime 依赖与 Detect 版本相同。

---

## 14. 当前 OBB 支持范围

本阶段目标是：

```text
Ultralytics YOLO OBB
batch = 1
one input
one raw prediction output
fixed industrial input W × H
FP32 / FP16
raw output without embedded NMS
```

典型 raw output：

```text
[1, 5 + nc, N]
```

或：

```text
[1, N, 5 + nc]
```

不针对：

```text
end-to-end NMS output
[1, max_det, 7] 等已后处理输出
多输出 OBB
自定义 OBB head
TensorRT plugin NMS
Seg / Pose
```

如果模型是这类结构，应单独适配输出。

---

## 15. 工业上线前的对照验证

不要只验证“能画出旋转框”。

建议选 20～50 张代表性工业图像，同时运行：

```text
Python Ultralytics best.pt / best.onnx
```

和：

```text
C# TensorRT程序
```

逐项比较：

```text
类别
confidence
center x/y
width/height
angle
四角点
最终目标数量
```

重点包含：

```text
0°附近
接近45°
接近90°
细长目标
两个旋转目标互相重叠
靠近图像边缘
低置信度目标
```

确认结果一致后再用于现场。

---

## 16. 推荐测试顺序

1. 先确认原有 Detect 模型仍正常。
2. 准备一个 OBB 固定尺寸 ONNX。
3. 修改 coco.names 为 OBB 类别。
4. ONNX → Engine。
5. 加载 Engine，确认 Task 自动识别为 OBB。
6. 选择图片执行检测。
7. 检查 Polygon 方向、中心、宽高和角度。
8. 对照 Python Ultralytics 输出。
9. 再测试 FP16。
10. 最后运行 publish_release.bat 到另一台现场电脑验证。
