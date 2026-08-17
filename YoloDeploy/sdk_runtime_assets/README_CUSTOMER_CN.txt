YoloDeploy.SDK.Runtime — 客户使用说明
=====================================

一、您需要什么
--------------
1. Windows x64
2. NVIDIA GPU
3. 兼容的 NVIDIA 显卡驱动
4. .NET 8 Windows Desktop Runtime x64
   当前 SDK 内部使用 WPF BitmapDecoder 读取图片，因此客户端项目建议设置：
   <UseWPF>true</UseWPF>
5. 本 ZIP 中的全部文件
6. 模型文件：
   - best.onnx
   - classes.names

不需要：
- Python
- PyTorch
- Ultralytics
- Visual Studio
- CUDA Toolkit 开发环境
- TensorRT SDK 开发环境
- trtexec
- 手工生成 .engine


二、重要：不要只复制 YoloDeploy.SDK.dll
---------------------------------------
YoloDeploy.SDK.dll 是程序代码中唯一需要直接引用的托管 SDK API。

但是运行时还依赖同目录中的：
- YoloDeploy.Native.dll
- TensorRT DLL
- ONNX Parser DLL
- CUDA 用户态 DLL

因此，请整体解压/复制本 Runtime 包。


三、模型
--------
推荐仅向 SDK 提供：

Models\best.onnx
Models\classes.names

classes.names 每行一个类别，顺序必须与训练模型一致。

例如：

OK
NG
scratch
hole


四、固定输入尺寸
----------------
创建 ObbDetector 时指定固定模型输入尺寸，例如：

InputWidth  = 1280
InputHeight = 512

原始待检测图片不需要刚好为 1280x512。
SDK Native 层会 LetterBox 到固定模型尺寸。

如果 ONNX 本身是固定 shape，则 ONNX 的输入尺寸必须与设置一致。
如果 ONNX 是 dynamic shape，SDK 会使用 MIN=OPT=MAX 将其构建为固定尺寸 Engine。


五、第一次与后续启动
--------------------
第一次：
ONNX
 -> 检查 GPU / TensorRT
 -> 构建当前电脑专用 TensorRT Engine
 -> 缓存 Engine
 -> 执行检测

后续：
ONNX
 -> 命中本机 Engine Cache
 -> 直接加载
 -> 执行检测

缓存默认位置：

%LOCALAPPDATA%\YoloDeploy\EngineCache


六、C# 调用
-----------
参考：

Examples\CSharp\YoloDeploySdkExample.cs


七、快速验证
------------
先双击：

verify_runtime.bat

确认关键 DLL 和 NVIDIA GPU 环境无 ERROR。

然后执行，例如：

TestSDK.exe Models\best.onnx Models\classes.names D:\Images\001.jpg 1280 512

首次运行因为需要创建 TensorRT Engine，会明显慢于后续运行。


八、检测返回
------------
每个 OBB 检测结果包含：

ClassId
ClassName
Confidence

CenterX
CenterY
Width
Height
AngleRadians
AngleDegrees

P1(X,Y)
P2(X,Y)
P3(X,Y)
P4(X,Y)


九、常见问题
------------
1. YoloDeploy.Native.dll 无法加载
   请确认 Runtime ZIP 中全部 DLL 与宿主 exe 位于同一运行目录。

2. 没有检测到 NVIDIA GPU
   安装/升级兼容的 NVIDIA 显卡驱动。

3. ONNX -> Engine 失败
   检查 ONNX 是否为项目支持的标准 Ultralytics raw-output OBB 模型，
   检查固定 W/H 是否与 ONNX shape 一致。

4. 类别错乱
   classes.names 的数量与顺序必须与训练/导出时一致。

5. Engine 不能在另一台电脑使用
   正常现象。建议分发 ONNX，由目标电脑第一次运行时自动生成自己的 Engine。
