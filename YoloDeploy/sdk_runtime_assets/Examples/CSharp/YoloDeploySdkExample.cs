using YoloDeploy.SDK;

using var detector = new ObbDetector(new ObbDetectorOptions
{
    ModelPath = @"D:\Model\best.onnx",
    ClassNamesPath = @"D:\Model\classes.names",

    // 固定模型输入尺寸
    InputWidth = 1280,
    InputHeight = 512,

    EnableFp16 = true,
    WorkspaceMiB = 1024,
    ConfidenceThreshold = 0.25f,
    NmsThreshold = 0.45f
});

var result = detector.Detect(
    @"D:\Images",
    "001.jpg");

foreach (var box in result.Detections)
{
    Console.WriteLine(
        $"Class={box.ClassName}, Score={box.Confidence:F3}, " +
        $"P1=({box.P1.X:F1},{box.P1.Y:F1}), " +
        $"P2=({box.P2.X:F1},{box.P2.Y:F1}), " +
        $"P3=({box.P3.X:F1},{box.P3.Y:F1}), " +
        $"P4=({box.P4.X:F1},{box.P4.Y:F1})");
}
