using System;
using System.IO;
using System.Text;

namespace YoloDeploy.SDK;

/// <summary>
/// TensorRT OBB detector.
/// The model is loaded once and remains resident until Dispose().
/// </summary>
public sealed class ObbDetector : IDisposable
{
    private readonly object _syncRoot = new();
    private readonly ObbDetectorOptions _options;
    private readonly string[] _classNames;

    private IntPtr _handle;
    private bool _disposed;

    /// <summary>
    /// Engine/runtime information resolved during construction.
    /// </summary>
    public ObbDetectorInitializationInfo InitializationInfo { get; }

    public int InputWidth => _options.InputWidth;
    public int InputHeight => _options.InputHeight;
    public IReadOnlyList<string> ClassNames => _classNames;

    public ObbDetector(ObbDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        _options = options;
        _classNames = LoadClassNames(options.ClassNamesPath);

        EngineResolveResult engine = EngineProvider.Resolve(options);

        var error = new StringBuilder(8192);

        try
        {
            _handle = NativeMethods.YoloCreate(
                engine.EnginePath,
                options.InputWidth,
                options.InputHeight,
                error,
                error.Capacity);
        }
        catch (DllNotFoundException ex)
        {
            throw new YoloSdkException(
                "加载 YoloDeploy.Native.dll 失败。请使用完整 SDK 运行时目录，"
                + "不要只复制 YoloDeploy.SDK.dll。",
                ex);
        }

        if (_handle == IntPtr.Zero)
        {
            throw new YoloSdkException(
                $"加载 TensorRT Engine 失败：{error}");
        }

        int taskHint =
            NativeMethods.YoloGetTaskHint(
                _handle,
                _classNames.Length);

        if (taskHint != 1)
        {
            NativeMethods.YoloDestroy(_handle);
            _handle = IntPtr.Zero;

            throw new YoloSdkException(
                "当前模型未被识别为标准 OBB 模型。"
                + "请确认 ONNX/Engine 为标准 Ultralytics OBB raw-output，"
                + "且 classes.names 的类别数量与模型一致。");
        }

        InitializationInfo = new ObbDetectorInitializationInfo
        {
            ModelPath = Path.GetFullPath(options.ModelPath),
            EnginePath = engine.EnginePath,
            BuiltFromOnnx = engine.BuiltFromOnnx,
            EngineCacheHit = engine.CacheHit,
            EngineBuiltNow = engine.BuiltNow,
            BuildLog = engine.BuildLog,
            Runtime = engine.Gpu.ToPublic()
        };
    }

    /// <summary>
    /// Detect one image using a full image path.
    /// </summary>
    public ObbDetectionResponse Detect(
        string imagePath,
        float? confidenceThreshold = null,
        float? nmsThreshold = null)
    {
        ThrowIfDisposed();

        float confidence =
            confidenceThreshold ?? _options.ConfidenceThreshold;

        float nms =
            nmsThreshold ?? _options.NmsThreshold;

        ValidateThreshold(
            confidence,
            nameof(confidenceThreshold));

        ValidateThreshold(
            nms,
            nameof(nmsThreshold));

        string fullPath = Path.GetFullPath(imagePath);
        BgraImage image =
            ImageLoader.LoadBgra32(fullPath);

        var nativeResults =
            new NativeMethods.YoloObbDetection[
                _options.MaxResults];

        var error =
            new StringBuilder(8192);

        int count;
        float inferenceMs;

        // One TensorRT execution context/stream is kept per detector.
        // Serialize calls on the same detector instance.
        lock (_syncRoot)
        {
            ThrowIfDisposed();

            count = NativeMethods.YoloDetectObbBgra(
                _handle,
                image.Pixels,
                image.Width,
                image.Height,
                image.Stride,
                confidence,
                nms,
                _classNames.Length,
                nativeResults,
                nativeResults.Length,
                out inferenceMs,
                error,
                error.Capacity);
        }

        if (count < 0)
        {
            throw new YoloSdkException(
                $"OBB 推理失败：{error}");
        }

        var detections =
            new List<ObbResult>(count);

        for (int i = 0; i < count; i++)
        {
            NativeMethods.YoloObbDetection d =
                nativeResults[i];

            string className =
                d.ClassId >= 0 &&
                d.ClassId < _classNames.Length
                    ? _classNames[d.ClassId]
                    : $"class_{d.ClassId}";

            detections.Add(new ObbResult
            {
                ClassId = d.ClassId,
                ClassName = className,
                Confidence = d.Score,

                CenterX = d.CenterX,
                CenterY = d.CenterY,
                Width = d.Width,
                Height = d.Height,
                AngleRadians = d.AngleRadians,

                P1 = new ObbPoint(d.P1X, d.P1Y),
                P2 = new ObbPoint(d.P2X, d.P2Y),
                P3 = new ObbPoint(d.P3X, d.P3Y),
                P4 = new ObbPoint(d.P4X, d.P4Y)
            });
        }

        return new ObbDetectionResponse
        {
            ImagePath = fullPath,
            ImageWidth = image.Width,
            ImageHeight = image.Height,
            InferenceMilliseconds = inferenceMs,
            Detections = detections
        };
    }

    /// <summary>
    /// Detect using "folder path + image file name", matching the requested external SDK API.
    /// </summary>
    public ObbDetectionResponse Detect(
        string imageDirectory,
        string imageName,
        float? confidenceThreshold = null,
        float? nmsThreshold = null)
    {
        if (string.IsNullOrWhiteSpace(imageDirectory))
            throw new ArgumentException(
                "图片目录不能为空。",
                nameof(imageDirectory));

        if (string.IsNullOrWhiteSpace(imageName))
            throw new ArgumentException(
                "图片名称不能为空。",
                nameof(imageName));

        return Detect(
            Path.Combine(imageDirectory, imageName),
            confidenceThreshold,
            nmsThreshold);
    }

    public void Dispose()
    {
        lock (_syncRoot)
        {
            if (_disposed)
                return;

            if (_handle != IntPtr.Zero)
            {
                NativeMethods.YoloDestroy(_handle);
                _handle = IntPtr.Zero;
            }

            _disposed = true;
        }

        GC.SuppressFinalize(this);
    }

    private static string[] LoadClassNames(
        string classNamesPath)
    {
        if (!File.Exists(classNamesPath))
        {
            throw new FileNotFoundException(
                "类别文件不存在。",
                classNamesPath);
        }

        string[] names = File.ReadAllLines(classNamesPath)
            .Select(x => x.Trim())
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .ToArray();

        if (names.Length == 0)
        {
            throw new YoloSdkException(
                "类别文件为空。每行应包含一个类别名称。");
        }

        return names;
    }

    private static void ValidateThreshold(
        float value,
        string name)
    {
        if (float.IsNaN(value) ||
            float.IsInfinity(value) ||
            value < 0 ||
            value > 1)
        {
            throw new ArgumentOutOfRangeException(
                name,
                "阈值必须位于 [0,1]。");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(
                nameof(ObbDetector));
        }
    }
}
