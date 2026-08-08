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
