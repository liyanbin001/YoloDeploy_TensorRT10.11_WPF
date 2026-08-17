using YoloDeploy.SDK;

if (args.Length < 4)
{
    Console.WriteLine(
        "Usage:");
    Console.WriteLine(
        "  YoloDeploy.SDK.Demo.exe <model.onnx> <classes.names> <imagePath> <inputWidth> [inputHeight]");
    Console.WriteLine();
    Console.WriteLine(
        "Example:");
    Console.WriteLine(
        @"  YoloDeploy.SDK.Demo.exe D:\Model\best.onnx D:\Model\classes.names D:\Images\001.jpg 1280 512");
    return;
}

string modelPath = args[0];
string namesPath = args[1];
string imagePath = args[2];

if (!int.TryParse(args[3], out int inputWidth))
{
    Console.WriteLine("inputWidth 无效。");
    return;
}

int inputHeight =
    args.Length >= 5 &&
    int.TryParse(args[4], out int parsedHeight)
        ? parsedHeight
        : 512;

try
{
    using var detector =
        new ObbDetector(
            new ObbDetectorOptions
            {
                ModelPath = modelPath,
                ClassNamesPath = namesPath,
                InputWidth = inputWidth,
                InputHeight = inputHeight,
                EnableFp16 = true,
                WorkspaceMiB = 1024,
                ConfidenceThreshold = 0.25f,
                NmsThreshold = 0.45f
            });

    ObbDetectorInitializationInfo init =
        detector.InitializationInfo;

    Console.WriteLine("=== Initialization ===");
    Console.WriteLine($"GPU        : {init.Runtime.GpuName}");
    Console.WriteLine($"CC         : {init.Runtime.ComputeCapability}");
    Console.WriteLine($"TensorRT   : {init.Runtime.TensorRtVersion}");
    Console.WriteLine($"Engine     : {init.EnginePath}");
    Console.WriteLine($"Cache hit  : {init.EngineCacheHit}");
    Console.WriteLine($"Built now  : {init.EngineBuiltNow}");
    Console.WriteLine();

    ObbDetectionResponse result =
        detector.Detect(imagePath);

    Console.WriteLine("=== Detection ===");
    Console.WriteLine(
        $"Image: {result.ImageWidth}x{result.ImageHeight}");
    Console.WriteLine(
        $"Inference: {result.InferenceMilliseconds:F2} ms");
    Console.WriteLine(
        $"Count: {result.Detections.Count}");
    Console.WriteLine();

    foreach (ObbResult box in result.Detections)
    {
        Console.WriteLine(
            $"[{box.ClassId}] {box.ClassName} "
            + $"score={box.Confidence:F3} "
            + $"center=({box.CenterX:F1},{box.CenterY:F1}) "
            + $"size=({box.Width:F1},{box.Height:F1}) "
            + $"angle={box.AngleDegrees:F2} deg");

        Console.WriteLine(
            $"    P1=({box.P1.X:F1},{box.P1.Y:F1}) "
            + $"P2=({box.P2.X:F1},{box.P2.Y:F1}) "
            + $"P3=({box.P3.X:F1},{box.P3.Y:F1}) "
            + $"P4=({box.P4.X:F1},{box.P4.Y:F1})");
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    Environment.ExitCode = 1;
}
