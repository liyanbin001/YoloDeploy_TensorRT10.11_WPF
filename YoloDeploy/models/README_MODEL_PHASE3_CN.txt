Phase 3 发布模型目录说明

把希望随发布包一起分发的 .onnx 放到当前 models 目录。

publish_release.ps1 会自动复制：
- *.onnx
- *.names
- *.txt

到最终发布包的 Models 目录。

默认不会复制 *.engine。
原因：Phase 2 的设计是目标电脑根据自己的 GPU / TensorRT 从 ONNX 构建并缓存本机 Engine。


Phase 4 固定矩形输入说明：
- 工业现场推荐固定输入宽度 + 固定输入高度，例如 1280 × 512。
- 如果 ONNX 是固定 shape，界面宽高必须与 ONNX 的 [1,3,H,W] 一致。
- 如果 ONNX 是动态 shape，程序会用 MIN=OPT=MAX=[1,3,H,W] 生成固定尺寸 Engine。
- 推荐网络输入尺寸接近工业相机图像宽高比；标准 YOLO 通常优先考虑 stride 32 的整数倍。


Phase 5 Detect / OBB 说明：
- 普通水平框模型选择“普通检测 Detect”。
- 旋转框模型选择“旋转框 OBB”；模型加载时会根据输出通道数和类别数自动尝试识别。
- OBB 需要标准 Ultralytics raw output：
  [x, y, w, h, class_probs..., angle]
- 不要把 NMS 烘焙进 OBB ONNX/Engine；本程序在 Native 层执行 ProbIoU rotated NMS。
- 自训练模型必须把程序目录中的 coco.names 替换为真实类别名称，数量和顺序与训练完全一致。
- 工业固定尺寸仍推荐 fixed-shape ONNX，例如输入 [1,3,512,1280] 对应 UI 宽 1280、高 512。


Phase 6 YOLO26-Seg：
- 推荐使用 yolo26n-seg / 自训练 best.pt 导出的 end-to-end ONNX。
- 固定工业输入保持 dynamic=False。
- 推荐 end2end=True（YOLO26 默认）。
- Seg 输出应为 prediction + proto 两个 tensor。
- 程序会由实例 Mask 自动得到分类、Mask、水平 bbox、Mask 面积和最小面积旋转矩形。
- 自训练模型必须把 YoloDeploy.App\coco.names 改成真实类别列表。


Phase 6 YOLO26-Seg 多任务建议：
- 推荐使用 yolo26n-seg 或自训练 best-seg.pt。
- 推荐 ONNX 导出：
  end2end=False
  nms=False
  dynamic=False
- 固定工业尺寸示例：
  imgsz=(512, 1280)
  对应软件 Width=1280, Height=512。
- Seg 模型提供：
  1) 每实例类别/置信度
  2) 实例 Mask
  3) Mask 水平外接框
  4) Mask 最小面积旋转矩形
  5) Mask 像素面积
- 自训练模型务必把 coco.names 替换为真实类别名称，行数必须等于 nc。
- 参考：models\export_yolo26_seg_example.py
- 详细说明：docs\PHASE6_YOLO26_SEG_MINRECT_CN.md
