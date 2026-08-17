using System;
using System.IO;
using System.Globalization;
using YoloDeploy.SDK;

Console.OutputEncoding = System.Text.Encoding.UTF8;

static void Usage()
{
    Console.WriteLine("YoloDeploy SDK OBB Test");
    Console.WriteLine();
    Console.WriteLine("用法：");
    Console.WriteLine("  TestSDK.exe <model.onnx> <classes.names> <image> <inputWidth> <inputHeight> [confidence] [nms]");
    Console.WriteLine();
    Console.WriteLine("示例：");
    Console.WriteLine(@"  TestSDK.exe Models\best.onnx Models\classes.names D:\Images\001.jpg 1280 512 0.25 0.45");
}

if (args.Length < 5)
{
    Usage();
    return 2;
}

string model = Path.GetFullPath(args[0]);
string names = Path.GetFullPath(args[1]);
string image = Path.GetFullPath(args[2]);

if (!int.TryParse(args[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out int width) ||
    !int.TryParse(args[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out int height))
{
    Console.Error.WriteLine("[ERROR] inputWidth / inputHeight 必须为整数。");
    return 2;
}

float confidence = 0.25f;
float nms = 0.45f;

if (args.Length >= 6 &&
    !float.TryParse(args[5], NumberStyles.Float, CultureInfo.InvariantCulture, out confidence))
{
    Console.Error.WriteLine("[ERROR] confidence 无效。请使用如 0.25。");
    return 2;
}

if (args.Length >= 7 &&
    !float.TryParse(args[6], NumberStyles.Float, CultureInfo.InvariantCulture, out nms))
{
    Console.Error.WriteLine("[ERROR] nms 无效。请使用如 0.45。");
    return 2;
}

try
{
    Console.WriteLine("=== Initialize OBB SDK ===");

    using var detector = new ObbDetector(new ObbDetectorOptions
    {
        ModelPath = model,
        ClassNamesPath = names,
        InputWidth = width,
        InputHeight = height,
        EnableFp16 = true,
        WorkspaceMiB = 1024,
        ConfidenceThreshold = confidence,
        NmsThreshold = nms
    });

    var init = detector.InitializationInfo;

    Console.WriteLine($"GPU          : {init.Runtime.GpuName}");
    Console.WriteLine($"Compute cap. : {init.Runtime.ComputeCapability}");
    Console.WriteLine($"TensorRT     : {init.Runtime.TensorRtVersion}");
    Console.WriteLine($"Engine       : {init.EnginePath}");
    Console.WriteLine($"Cache hit    : {init.EngineCacheHit}");
    Console.WriteLine($"Built now    : {init.EngineBuiltNow}");
    Console.WriteLine();

    Console.WriteLine("=== Detect ===");
    var result = detector.Detect(image);

    Console.WriteLine($"Image        : {result.ImageWidth} x {result.ImageHeight}");
    Console.WriteLine($"Inference    : {result.InferenceMilliseconds:F2} ms");
    Console.WriteLine($"Detections   : {result.Detections.Count}");
    Console.WriteLine();

    for (int i = 0; i < result.Detections.Count; i++)
    {
        var d = result.Detections[i];

        Console.WriteLine(
            $"#{i + 1} class={d.ClassName} id={d.ClassId} score={d.Confidence:F4}");

        Console.WriteLine(
            $"  center=({d.CenterX:F2},{d.CenterY:F2}) " +
            $"size=({d.Width:F2},{d.Height:F2}) " +
            $"angle={d.AngleDegrees:F2} deg");

        Console.WriteLine(
            $"  P1=({d.P1.X:F2},{d.P1.Y:F2}) " +
            $"P2=({d.P2.X:F2},{d.P2.Y:F2}) " +
            $"P3=({d.P3.X:F2},{d.P3.Y:F2}) " +
            $"P4=({d.P4.X:F2},{d.P4.Y:F2})");
    }

    Console.WriteLine();
    Console.WriteLine("[OK] SDK test completed.");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine();
    Console.Error.WriteLine("[ERROR] SDK test failed.");
    Console.Error.WriteLine(ex);
    return 1;
}
