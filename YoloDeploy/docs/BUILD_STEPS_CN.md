# YoloDeploy 构建说明

## 1. 为什么采用这个结构

本方案把图片解码、界面、绘制放在 C# WPF；把 TensorRT、CUDA、YOLOv8 前后处理放在 C++ DLL。

优点：
- 不需要 OpenCV
- WPF 页面容易改
- TensorRT 仍使用官方 C++ API
- 模型仍为独立 `.engine`
- C# 和 C++ 之间只传 BGRA 图片字节和检测框结构

## 2. C# -> C++ 数据流

WPF 使用 BitmapDecoder 加载图片，然后转为 `PixelFormats.Bgra32`。

传给 C++：
- BGRA byte[]
- width
- height
- stride
- confidence threshold
- NMS threshold

C++ 返回：
- x1, y1, x2, y2
- score
- class id

WPF 用 Canvas 覆盖在图片上绘制矩形和文字。

## 3. C++ 预处理

执行标准 Ultralytics 风格预处理：

1. 根据模型输入尺寸计算缩放比例：
   `r = min(inputW/srcW, inputH/srcH)`
2. 保持比例缩放
3. 居中 LetterBox，填充值 114
4. BGRA -> RGB
5. `float = value / 255`
6. HWC -> CHW
7. 拷贝到 CUDA device buffer

为了不依赖 OpenCV，缩放使用 C++ 手写双线性插值。

## 4. TensorRT 10.x 推理

加载流程：

1. 读取 engine 二进制
2. `createInferRuntime`
3. `deserializeCudaEngine`
4. `createExecutionContext`
5. `getNbIOTensors/getIOTensorName` 枚举 I/O
6. 动态尺寸时 `setInputShape`
7. 为输入输出 `cudaMalloc`
8. `setTensorAddress`
9. `cudaMemcpyAsync`
10. `enqueueV3`
11. `cudaMemcpyAsync` 输出回 CPU
12. `cudaStreamSynchronize`

## 5. YOLOv8 解码

标准 YOLOv8 Detect 无 NMS 输出常见为：

`[1, 4 + nc, N]`

COCO 80 类时：

`[1, 84, 8400]`

每个候选：
- cx
- cy
- w
- h
- nc 个类别分数

程序取最大类别分数作为置信度，过滤低分框，再把 LetterBox 坐标还原到原图。

代码同时兼容 `[1, N, 4+nc]`。

## 6. NMS

按置信度从高到低排序。
只对相同 class id 的框进行 IoU 抑制，所以不同类别不会互相压掉。

## 7. 为什么 Engine 不转 DLL

`.engine` 是 TensorRT 序列化执行计划，DLL 是程序代码。它们作用不同。

最终结构是：

`WPF -> Native DLL -> TensorRT Runtime -> engine`

这样换模型时只换 `.engine` 和 `coco.names`，不需要重新编译 DLL（前提是输出格式相同）。

## 8. 第一次运行建议

第一版强烈建议：
- yolov8n.pt
- imgsz=640
- batch=1
- dynamic=False
- half=False
- nms=False
- 当前 RTX 2080 Super 本机导出

FP32 跑通以后再试 `half=True`。

## 9. 自定义数据集

如果模型仍是标准 YOLOv8 Detect，只需要把 `coco.names` 替换成训练类别，一行一个。

程序会用 engine 输出通道数推断 class count。
如果 `coco.names` 数量少于模型类别数，多出来的类别会显示 `class_XX`。

## 10. 下一步可扩展

当前工程可继续增加：
- 模型常驻内存，连续检测多张图
- 文件夹批量推理
- 视频/摄像头
- CUDA 预处理
- GPU NMS
- FP16 优化
- 性能统计
- 自定义类别颜色
- YOLOv8 OBB
