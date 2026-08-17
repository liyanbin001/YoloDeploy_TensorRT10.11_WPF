# Phase 6：YOLO26 实例分割统一多任务 + Mask 最小外接矩形

## 1. 目标

Phase 6 不再把 OBB 角度回归作为工业旋转框的唯一来源，而是推荐：

```text
YOLO26n-seg
    ↓
实例分割
    ↓
每个实例同时得到：
    ├─ 类别 / confidence
    ├─ 二值 Mask
    ├─ Mask 水平外接框
    └─ Mask 最小面积旋转矩形
```

因此一套实例分割模型可以在工业软件里提供：

```text
实例分类
实例分割
普通目标检测框
旋转框 / 方向角
Mask面积
```

原有 Detect 和 OBB 推理路径仍然保留，方便旧模型继续使用。

---

## 2. “分类”的含义

这里的分类是：

```text
每一个被分割实例的类别
```

例如：

```text
实例1 → scratch 0.97
实例2 → dent    0.94
```

它不是独立的整图分类网络：

```text
yolo26n-cls
```

如果你的工业问题是“找到工件/缺陷后判断它属于哪个类别”，Seg 的实例类别已经可以直接使用。

---

## 3. 推荐 YOLO26n-seg 导出

为了与 C++ TensorRT 后处理最稳定地对应，Phase 6 推荐：

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

model.export(
    format="onnx",
    imgsz=(512, 1280),   # height, width
    batch=1,
    dynamic=False,
    end2end=False,
    nms=False,
    simplify=True,
)
```

固定工业尺寸时：

```text
WPF固定输入宽度  = 1280
WPF固定输入高度  = 512
```

---

## 4. 推荐输出结构

推荐 `end2end=False`。

预测输出：

```text
[1, 4 + nc + nm, N]
```

或转置形式：

```text
[1, N, 4 + nc + nm]
```

内容：

```text
cx
cy
w
h
class probabilities...
mask coefficients...
```

Prototype 输出：

```text
[1, nm, protoH, protoW]
```

典型：

```text
nm = 32
```

程序根据第二个 4D 输出自动识别 Seg。

---

## 5. End-to-End 输出兼容

Phase 6 也兼容 YOLO26 的 end-to-end segmentation prediction：

```text
[1, max_det, 6 + nm]
```

内容：

```text
x1
y1
x2
y2
confidence
class_id
mask coefficients...
```

但工业部署推荐统一使用：

```text
end2end=False
nms=False
```

原因是输出定义更直观，而且与你已有 Detect/OBB “外部 C++ 后处理”架构一致。

---

## 6. TensorRT Engine Builder 的变化

Phase 5：

```text
1 input
1 output
```

Phase 6：

```text
1 input

1 output：
Detect / OBB

或

2 outputs：
Seg prediction
Seg proto
```

`OnnxEngineBuilder.cpp` 现在允许 1～2 个输出，并把每个输出的名称和 shape 写入构建日志。

---

## 7. TensorRT Session 的变化

以前：

```text
outputName
outputDims
outputDevice
```

现在保留 prediction，同时可增加：

```text
protoName
protoDims
protoDevice
```

程序不依赖具体 ONNX tensor 名字，而按 rank 判断：

```text
3D → prediction
4D → proto
```

因此不同 Ultralytics 导出版本只要保持标准 segmentation 两输出结构，通常不需要硬编码 `output0` / `output1`。

---

## 8. Raw Seg 后处理

对于推荐的 raw prediction：

```text
[1,4+nc+nm,N]
```

执行：

```text
每个候选框
↓
取最大 class score
↓
confidence threshold
↓
xywh → xyxy
↓
class-aware NMS
↓
保留 mask coefficients
```

NMS 阈值继续使用界面中的：

```text
NMS 阈值
```

---

## 9. Mask 重建

对于一个实例：

```text
maskCoefficients = [nm]
proto = [nm, protoH, protoW]
```

首先：

```text
maskLogit =
maskCoefficients × proto
```

也就是：

```text
[nm]
×
[nm, H, W]
=
[H, W]
```

之后将 mask logit 双线性映射回输入/原图坐标，并按 LetterBox 关系去除 padding。

界面默认：

```text
Mask阈值 = 0.50
```

等价于：

```text
sigmoid(maskLogit) > 0.5
```

也即：

```text
maskLogit > 0
```

---

## 10. 为什么仍然在 bbox 范围内裁剪 mask

Ultralytics 标准实例分割后处理会按照实例 bbox 对 mask 进行 crop。

Phase 6 也保持这个逻辑：

```text
proto mask
↓
双线性恢复
↓
在对应检测 bbox 内进行二值 mask 判断
```

这样更容易与官方 Python 推理结果对齐。

---

## 11. Mask 水平框

二值 mask 生成后直接统计：

```text
maskMinX
maskMinY
maskMaxX
maskMaxY
```

得到：

```text
x1
y1
x2
y2
```

这个框不是再次使用模型的 bbox，而是：

```text
最终二值 Mask 的水平外接框
```

因此软件的“目标检测显示”可以由最终分割形状派生。

---

## 12. Mask 面积

对二值 mask 中的正像素计数：

```text
MaskAreaPixels
```

例如：

```text
12458 px
```

如果工业现场后续完成像素标定，例如：

```text
0.02 mm / pixel
```

就可以继续换算实际面积。

Phase 6 目前只输出 pixel area，不自动假设物理标定比例。

---

## 13. 最小面积旋转矩形

Phase 6 没有增加 OpenCV 依赖。

Native C++ 自己执行：

```text
Mask边界点
↓
Convex Hull
↓
遍历凸包边方向
↓
在每个候选角度下投影
↓
求最小面积矩形
```

得到：

```text
centerX
centerY
rotatedWidth
rotatedHeight
angle
P1
P2
P3
P4
```

其中：

```text
rotatedWidth
```

统一定义为长边。

角度：

```text
[0, π)
```

表示长轴方向。

WPF 中显示为 degree：

```text
angleDegrees = angleRadians * 180 / PI
```

---

## 14. 为什么 Mask → MinAreaRect 适合工业目标

直接 OBB Head 需要网络同时学习：

```text
中心
宽高
角度
类别
```

如果角度标注不稳定、目标接近对称、边缘细节复杂，OBB 角度训练可能不理想。

Seg 路线改为：

```text
网络学习目标轮廓
↓
确定性几何算法计算方向
```

对于轮廓清晰的工业零件、缺陷、条形目标、倾斜工件等，往往更容易解释和调试。

但最终精度仍然取决于：

```text
Mask轮廓是否准确
```

因此应优先提高 segmentation 数据集的轮廓标注质量。

---

## 15. WPF 界面

模型任务新增：

```text
实例分割 Seg → 多任务
```

并增加：

```text
Mask阈值
☑ 显示Mask
☑ 显示水平框
☑ 显示最小外接矩形
```

因此同一个 Seg Engine 可以切换显示目的。

### 分类

看结果表：

```text
类别
置信度
```

### 分割

勾选：

```text
显示Mask
```

### 普通目标检测

勾选：

```text
显示水平框
```

此框来自最终 mask。

### 旋转框

勾选：

```text
显示最小外接矩形
```

此框来自最终 mask 的最小面积矩形。

---

## 16. 结果表

Seg 模式显示：

```text
#
类别
置信度
MaskBox
旋转宽×高
角度
Mask面积
```

例如：

```text
1
part_A
0.973
MaskBox(113,72)-(418,330) R=304×88
16.8°
18244 px
```

---

## 17. Mask Overlay

Native 返回一个原始图像尺寸的：

```text
uint16 instance-id map
```

其中：

```text
0 = background
1 = 第1个实例
2 = 第2个实例
...
```

WPF 根据实例 ID 生成半透明颜色 overlay。

因此不需要把每个完整 bool mask 通过 P/Invoke 单独传输。

---

## 18. 多实例重叠

实例本身的：

```text
面积
水平框
最小外接矩形
```

都分别根据自己的 mask 计算。

显示用 instance-id map 如果多个实例重叠，则优先保留置信度较高的实例 ID。

这只影响 overlay 的重叠像素显示，不影响各实例自己的几何统计。

---

## 19. 类别文件

自训练模型一定要替换：

```text
YoloDeploy.App\coco.names
```

例如：

```text
part_ok
scratch
dent
crack
```

必须满足：

```text
行数 = nc
顺序 = 训练数据类别顺序
```

Raw Seg 输出依靠这个数量区分：

```text
4 + nc + nm
```

因此类别文件数量错误会导致 shape 校验失败。

---

## 20. 固定矩形输入继续保留

例如工业相机适合：

```text
Width = 1280
Height = 512
```

导出：

```python
imgsz=(512, 1280)
dynamic=False
```

软件：

```text
固定输入宽度 = 1280
固定输入高度 = 512
```

现有：

```text
Engine Cache
GPU信息
TensorRT版本
FP32/FP16
Workspace
```

全部继续工作。

---

## 21. Engine Cache

Seg 模型仍然根据：

```text
ONNX SHA-256
GPU
Compute Capability
TensorRT
FP32/FP16
Width×Height
Workspace
```

生成缓存 Key。

Seg 有两个输出不会改变缓存设计。

---

## 22. 一键发布

仍然：

```text
publish_release.bat
```

Phase 6 默认发布包：

```text
YoloDeploy_v6_seg-minrect_win-x64
```

没有增加 OpenCV DLL。

目标电脑依然主要需要：

```text
NVIDIA Driver
VC++ Runtime（推荐安装）
```

TensorRT/CUDA 用户态 DLL 仍由一键发布脚本收集。

---

## 23. 推荐验证方法

上线之前必须与 Python Ultralytics 对照。

Python：

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    "test.jpg",
    imgsz=(512, 1280),
    conf=0.25
)

r = results[0]

print(r.boxes.cls)
print(r.boxes.conf)
print(r.boxes.xyxy)
print(r.masks.data.shape)
print(r.masks.xy)
```

C# TensorRT 对比：

```text
类别
置信度
Mask水平框
Mask面积
Mask轮廓位置
MinAreaRect
角度
```

重点选：

```text
小目标
细长目标
接近水平
接近垂直
45°附近
凹形目标
多个目标接近/重叠
目标靠近图像边缘
```

---

## 24. 关于凹形目标

最小面积矩形是：

```text
整个实例轮廓的外接矩形
```

对于：

```text
L形
C形
弯曲目标
```

它仍然只给出一个全局方向矩形。

如果以后你需要：

```text
骨架方向
主轴 PCA
局部方向
长度/宽度测量
轮廓曲率
```

可以继续在 Phase 6 的 Mask 几何层扩展，而不需要改 TensorRT 模型。

---

## 25. 推荐工业架构

最终推荐：

```text
best-seg.pt
↓
固定尺寸 ONNX
end2end=False
nms=False
↓
目标电脑 TensorRT Builder
↓
GPU专属 Engine Cache
↓
Seg推理
↓
Class + Score
↓
Instance Mask
├─ Mask本身 → 分割任务
├─ Mask AABB → 普通检测任务
├─ Mask MinAreaRect → 旋转框任务
└─ Class/Score → 实例分类任务
```

这样模型层只维护一个 segmentation 模型，几何需求全部统一在确定性的 C++ 后处理层完成。
