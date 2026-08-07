Phase 3 发布模型目录说明

把希望随发布包一起分发的 .onnx 放到当前 models 目录。

publish_release.ps1 会自动复制：
- *.onnx
- *.names
- *.txt

到最终发布包的 Models 目录。

默认不会复制 *.engine。
原因：Phase 2 的设计是目标电脑根据自己的 GPU / TensorRT 从 ONNX 构建并缓存本机 Engine。
