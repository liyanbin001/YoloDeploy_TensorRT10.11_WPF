using Microsoft.Win32;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace YoloDeploy.App;

public partial class MainWindow : Window
{
    private IntPtr _detector = IntPtr.Zero;
    private BitmapSource? _currentBitmap;
    private byte[]? _currentBgra;
    private int _currentStride;
    private string[] _classNames = Array.Empty<string>();

    private GpuInfo? _gpuInfo;
    private bool _uiReady;
    private string? _cachedOnnxHash;
    private string? _cachedOnnxHashPath;
    private long _cachedOnnxLength;
    private DateTime _cachedOnnxLastWriteUtc;

    public MainWindow()
    {
        InitializeComponent();
        LoadClassNames();
        _uiReady = true;
        UpdateCacheUiState();
        UpdateCacheStats();
    }

    private void LoadClassNames()
    {
        string path = System.IO.Path.Combine(
            AppContext.BaseDirectory,
            "coco.names");

        _classNames = File.Exists(path)
            ? File.ReadAllLines(path)
                .Select(x => x.Trim())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .ToArray()
            : Array.Empty<string>();
    }

    // ============================================================
    // Phase 2: GPU information + Engine cache
    // ============================================================

    private async void Window_Loaded(
        object sender,
        RoutedEventArgs e)
    {
        await RefreshGpuInfoAsync();
        await RefreshCachePreviewAsync();
    }

    private async void RefreshGpuInfo_Click(
        object sender,
        RoutedEventArgs e)
    {
        await RefreshGpuInfoAsync();
        await RefreshCachePreviewAsync();
    }

    private async Task RefreshGpuInfoAsync()
    {
        try
        {
            RefreshGpuInfoButton.IsEnabled = false;

            GpuInfoTextBlock.Text =
                "正在读取 GPU / CUDA / TensorRT 信息...";

            _gpuInfo =
                await Task.Run(
                    GpuInfoProvider.Query);

            GpuInfoTextBlock.Text =
                _gpuInfo.DisplayText;

            StatusTextBlock.Text =
                $"GPU 已识别：{_gpuInfo.Name} | CC {_gpuInfo.ComputeCapability}";
        }
        catch (Exception ex)
        {
            _gpuInfo = null;

            GpuInfoTextBlock.Text =
                $"GPU 信息读取失败：{ex.Message}";
        }
        finally
        {
            RefreshGpuInfoButton.IsEnabled = true;
        }
    }

    private async void CacheOption_Changed(
        object sender,
        RoutedEventArgs e)
    {
        if (!_uiReady)
            return;

        UpdateCacheUiState();
        await RefreshCachePreviewAsync();
    }

    private async void CacheParameter_TextChanged(
        object sender,
        TextChangedEventArgs e)
    {
        if (!_uiReady)
            return;

        await RefreshCachePreviewAsync();
    }

    private async void CacheParameter_SelectionChanged(
        object sender,
        SelectionChangedEventArgs e)
    {
        if (!_uiReady)
            return;

        await RefreshCachePreviewAsync();
    }

    private void UpdateCacheUiState()
    {
        bool useCache =
            UseEngineCacheCheckBox.IsChecked == true;

        EngineOutputPathTextBox.IsReadOnly =
            useCache;

        BrowseEngineOutputButton.IsEnabled =
            !useCache;

        ForceRebuildCheckBox.IsEnabled =
            useCache;
    }

    private async Task<string> GetOnnxHashAsync(
        string onnxPath)
    {
        FileInfo file =
            new(onnxPath);

        if (!file.Exists)
        {
            throw new FileNotFoundException(
                "ONNX 文件不存在。",
                onnxPath);
        }

        if (string.Equals(
                _cachedOnnxHashPath,
                file.FullName,
                StringComparison.OrdinalIgnoreCase) &&
            _cachedOnnxHash is not null &&
            _cachedOnnxLength == file.Length &&
            _cachedOnnxLastWriteUtc == file.LastWriteTimeUtc)
        {
            return _cachedOnnxHash;
        }

        string hash =
            await EngineCacheManager.ComputeSha256Async(
                file.FullName);

        _cachedOnnxHashPath =
            file.FullName;

        _cachedOnnxHash =
            hash;

        _cachedOnnxLength =
            file.Length;

        _cachedOnnxLastWriteUtc =
            file.LastWriteTimeUtc;

        return hash;
    }

    private async Task<EngineCacheDescriptor?>
        CreateCurrentCacheDescriptorAsync()
    {
        if (UseEngineCacheCheckBox.IsChecked != true)
            return null;

        string onnxPath =
            OnnxPathTextBox.Text.Trim();

        if (!File.Exists(onnxPath))
            return null;

        if (_gpuInfo is null)
            return null;

        if (!int.TryParse(
                BuildInputWidthTextBox.Text,
                out int inputWidth) ||
            inputWidth <= 0 ||
            !int.TryParse(
                BuildInputHeightTextBox.Text,
                out int inputHeight) ||
            inputHeight <= 0)
        {
            return null;
        }

        if (!int.TryParse(
                WorkspaceTextBox.Text,
                out int workspaceMiB) ||
            workspaceMiB < 64)
        {
            return null;
        }

        string hash =
            await GetOnnxHashAsync(
                onnxPath);

        return EngineCacheManager.CreateDescriptor(
            onnxPath,
            hash,
            _gpuInfo,
            GetSelectedPrecision(),
            inputWidth,
            inputHeight,
            workspaceMiB);
    }

    private async Task RefreshCachePreviewAsync()
    {
        if (!_uiReady)
            return;

        UpdateCacheUiState();
        UpdateCacheStats();

        if (UseEngineCacheCheckBox.IsChecked != true)
        {
            CacheStatusTextBlock.Text =
                "缓存状态：已禁用。Engine 将保存到手动指定的位置。";

            if (File.Exists(
                    OnnxPathTextBox.Text.Trim()) &&
                string.IsNullOrWhiteSpace(
                    EngineOutputPathTextBox.Text))
            {
                EngineOutputPathTextBox.Text =
                    BuildDefaultEnginePath(
                        OnnxPathTextBox.Text.Trim());
            }

            return;
        }

        if (_gpuInfo is null)
        {
            CacheStatusTextBlock.Text =
                "缓存状态：等待 GPU 信息。";

            return;
        }

        string onnxPath =
            OnnxPathTextBox.Text.Trim();

        if (!File.Exists(onnxPath))
        {
            CacheStatusTextBlock.Text =
                "缓存状态：等待选择 ONNX。";

            return;
        }

        if (!int.TryParse(
                BuildInputWidthTextBox.Text,
                out int inputWidth) ||
            inputWidth <= 0 ||
            !int.TryParse(
                BuildInputHeightTextBox.Text,
                out int inputHeight) ||
            inputHeight <= 0 ||
            !int.TryParse(
                WorkspaceTextBox.Text,
                out int workspaceMiB) ||
            workspaceMiB < 64)
        {
            CacheStatusTextBlock.Text =
                "缓存状态：请输入有效的固定输入宽度、高度和 Workspace。";

            return;
        }

        try
        {
            CacheStatusTextBlock.Text =
                "缓存状态：正在计算 ONNX SHA-256...";

            EngineCacheDescriptor? descriptor =
                await CreateCurrentCacheDescriptorAsync();

            if (descriptor is null)
                return;

            EngineOutputPathTextBox.Text =
                descriptor.EnginePath;

            bool hit =
                EngineCacheManager.TryValidate(
                    descriptor,
                    out string reason);

            CacheStatusTextBlock.Text =
                hit
                    ? $"缓存命中：{System.IO.Path.GetFileName(descriptor.EnginePath)}"
                    : $"缓存未命中：{reason}。将生成 {System.IO.Path.GetFileName(descriptor.EnginePath)}";
        }
        catch (Exception ex)
        {
            CacheStatusTextBlock.Text =
                $"缓存状态检查失败：{ex.Message}";
        }
    }

    private void UpdateCacheStats()
    {
        try
        {
            EngineCacheStats stats =
                EngineCacheManager.GetStats();

            CacheStatsTextBlock.Text =
                $"缓存：{stats.DisplayText}";
        }
        catch (Exception ex)
        {
            CacheStatsTextBlock.Text =
                $"缓存统计失败：{ex.Message}";
        }
    }

    private void OpenCache_Click(
        object sender,
        RoutedEventArgs e)
    {
        try
        {
            EngineCacheManager.OpenCacheFolder();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "打开缓存目录失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
    }

    private async void ClearCache_Click(
        object sender,
        RoutedEventArgs e)
    {
        var result =
            MessageBox.Show(
                "确定清空本机全部 TensorRT Engine 缓存吗？\n\n"
                + EngineCacheManager.CacheRoot,
                "清空 Engine 缓存",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning);

        if (result != MessageBoxResult.Yes)
            return;

        try
        {
            if (EngineCacheManager.IsInsideCache(
                    EnginePathTextBox.Text.Trim()))
            {
                DestroyDetector();

                ModelInfoTextBlock.Text =
                    "模型尚未加载。";
            }

            EngineCacheManager.ClearAll();

            UpdateCacheStats();
            await RefreshCachePreviewAsync();

            StatusTextBlock.Text =
                "Engine 缓存已清空";
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "清空缓存失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
    }

    // ============================================================
    // Phase 1: ONNX -> TensorRT Engine
    // ============================================================

    private async void BrowseOnnx_Click(
        object sender,
        RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择 ONNX 模型",
            Filter = "ONNX 模型 (*.onnx)|*.onnx|所有文件 (*.*)|*.*"
        };

        if (dialog.ShowDialog() != true)
            return;

        OnnxPathTextBox.Text =
            dialog.FileName;

        _cachedOnnxHash = null;
        _cachedOnnxHashPath = null;

        if (UseEngineCacheCheckBox.IsChecked != true)
        {
            EngineOutputPathTextBox.Text =
                BuildDefaultEnginePath(
                    dialog.FileName);
        }

        await RefreshCachePreviewAsync();
    }

    private void BrowseEngineOutput_Click(
        object sender,
        RoutedEventArgs e)
    {
        string onnxPath =
            OnnxPathTextBox.Text.Trim();

        var dialog =
            new SaveFileDialog
            {
                Title = "保存 TensorRT Engine",
                Filter = "TensorRT Engine (*.engine)|*.engine|所有文件 (*.*)|*.*",
                AddExtension = true,
                DefaultExt = ".engine",
                OverwritePrompt = true
            };

        if (File.Exists(onnxPath))
        {
            dialog.InitialDirectory =
                System.IO.Path.GetDirectoryName(
                    onnxPath);

            dialog.FileName =
                System.IO.Path.GetFileName(
                    BuildDefaultEnginePath(
                        onnxPath));
        }

        if (dialog.ShowDialog() == true)
        {
            UseEngineCacheCheckBox.IsChecked =
                false;

            EngineOutputPathTextBox.Text =
                dialog.FileName;
        }
    }

    private string BuildDefaultEnginePath(
        string onnxPath)
    {
        string directory =
            System.IO.Path.GetDirectoryName(onnxPath)
            ?? Environment.CurrentDirectory;

        string stem =
            System.IO.Path.GetFileNameWithoutExtension(onnxPath);

        int width =
            int.TryParse(
                BuildInputWidthTextBox.Text,
                out int parsedWidth)
            && parsedWidth > 0
                ? parsedWidth
                : 640;

        int height =
            int.TryParse(
                BuildInputHeightTextBox.Text,
                out int parsedHeight)
            && parsedHeight > 0
                ? parsedHeight
                : 640;

        string precision =
            GetSelectedPrecision();

        return System.IO.Path.Combine(
            directory,
            $"{stem}_trt10_11_{precision.ToLowerInvariant()}_{width}x{height}.engine");
    }

    private string GetSelectedPrecision()
    {
        if (PrecisionComboBox.SelectedItem
            is ComboBoxItem item)
        {
            return item.Content?.ToString()
                   ?? "FP32";
        }

        return "FP32";
    }

    private async void BuildEngine_Click(
        object sender,
        RoutedEventArgs e)
    {
        try
        {
            string onnxPath =
                OnnxPathTextBox.Text.Trim();

            if (!File.Exists(onnxPath))
            {
                throw new FileNotFoundException(
                    "请选择有效的 .onnx 文件。",
                    onnxPath);
            }

            if (!int.TryParse(
                    BuildInputWidthTextBox.Text,
                    out int inputWidth) ||
                inputWidth <= 0)
            {
                throw new InvalidOperationException(
                    "固定输入宽度必须是正整数，例如 1280。");
            }

            if (!int.TryParse(
                    BuildInputHeightTextBox.Text,
                    out int inputHeight) ||
                inputHeight <= 0)
            {
                throw new InvalidOperationException(
                    "固定输入高度必须是正整数，例如 512。");
            }

            if (!int.TryParse(
                    WorkspaceTextBox.Text,
                    out int workspaceMiB) ||
                workspaceMiB < 64)
            {
                throw new InvalidOperationException(
                    "Workspace 至少设置为 64 MiB；建议 2048。");
            }

            if (_gpuInfo is null)
            {
                await RefreshGpuInfoAsync();
            }

            if (_gpuInfo is null)
            {
                throw new InvalidOperationException(
                    "无法读取当前 GPU 信息，不能安全创建 Engine 缓存。");
            }

            bool enableFp16 =
                string.Equals(
                    GetSelectedPrecision(),
                    "FP16",
                    StringComparison.OrdinalIgnoreCase);

            EngineCacheDescriptor? cacheDescriptor =
                null;

            string enginePath;

            if (UseEngineCacheCheckBox.IsChecked == true)
            {
                cacheDescriptor =
                    await CreateCurrentCacheDescriptorAsync()
                    ?? throw new InvalidOperationException(
                        "无法生成 Engine 缓存描述。");

                enginePath =
                    cacheDescriptor.EnginePath;

                EngineOutputPathTextBox.Text =
                    enginePath;

                bool forceRebuild =
                    ForceRebuildCheckBox.IsChecked == true;

                if (!forceRebuild &&
                    EngineCacheManager.TryValidate(
                        cacheDescriptor,
                        out string cacheReason))
                {
                    EnginePathTextBox.Text =
                        enginePath;

                    InputWidthTextBox.Text =
                        inputWidth.ToString(
                            CultureInfo.InvariantCulture);

                    InputHeightTextBox.Text =
                        inputHeight.ToString(
                            CultureInfo.InvariantCulture);

                    BuildLogTextBox.Text =
                        "Result: CACHE HIT"
                        + $"Engine: {enginePath}"
                        + $"GPU: {_gpuInfo.Name} | CC {_gpuInfo.ComputeCapability}"
                        + $"TensorRT: {_gpuInfo.TensorRtVersion}"
                        + $"Precision: {GetSelectedPrecision()}"
                        + $"Input: {inputWidth}x{inputHeight}\n"
                        + $"Workspace: {workspaceMiB} MiB"
                        + $"Validation: {cacheReason}";

                    CacheStatusTextBlock.Text =
                        $"缓存命中：{System.IO.Path.GetFileName(enginePath)}";

                    StatusTextBlock.Text =
                        "已复用本机 Engine 缓存，无需重新构建。";

                    return;
                }
            }
            else
            {
                enginePath =
                    EngineOutputPathTextBox.Text.Trim();

                if (string.IsNullOrWhiteSpace(
                        enginePath))
                {
                    enginePath =
                        BuildDefaultEnginePath(
                            onnxPath);

                    EngineOutputPathTextBox.Text =
                        enginePath;
                }

                if (!string.Equals(
                        System.IO.Path.GetExtension(enginePath),
                        ".engine",
                        StringComparison.OrdinalIgnoreCase))
                {
                    enginePath += ".engine";

                    EngineOutputPathTextBox.Text =
                        enginePath;
                }

                if (File.Exists(enginePath))
                {
                    var overwrite =
                        MessageBox.Show(
                            $"Engine 已存在：{enginePath}是否覆盖？",
                            "确认覆盖",
                            MessageBoxButton.YesNo,
                            MessageBoxImage.Question);

                    if (overwrite !=
                        MessageBoxResult.Yes)
                    {
                        return;
                    }
                }
            }

            string? outputDirectory =
                System.IO.Path.GetDirectoryName(
                    enginePath);

            if (!string.IsNullOrWhiteSpace(
                    outputDirectory))
            {
                Directory.CreateDirectory(
                    outputDirectory);
            }

            DestroyDetector();

            ModelInfoTextBlock.Text =
                "模型尚未加载。";

            BuildEngineButton.IsEnabled =
                false;

            LoadModelButton.IsEnabled =
                false;

            DetectButton.IsEnabled =
                false;

            RefreshGpuInfoButton.IsEnabled =
                false;

            StatusTextBlock.Text =
                "正在从 ONNX 构建 TensorRT Engine，请勿关闭程序...";

            BuildLogTextBox.Text =
                $"开始构建..."
                + $"GPU: {_gpuInfo.Name} | CC {_gpuInfo.ComputeCapability}"
                + $"TensorRT: {_gpuInfo.TensorRtVersion}"
                + $"ONNX: {onnxPath}"
                + $"输出: {enginePath}"
                + $"精度: {GetSelectedPrecision()}"
                + $"输入: {inputWidth}x{inputHeight}\n"
                + $"Workspace: {workspaceMiB} MiB";

            var timer =
                Stopwatch.StartNew();

            var result =
                await Task.Run(() =>
                {
                    var log =
                        new StringBuilder(32768);

                    var error =
                        new StringBuilder(8192);

                    int code =
                        NativeMethods.YoloBuildEngineFromOnnx(
                            onnxPath,
                            enginePath,
                            inputWidth,
                            inputHeight,
                            enableFp16 ? 1 : 0,
                            workspaceMiB,
                            log,
                            log.Capacity,
                            error,
                            error.Capacity);

                    return (
                        code,
                        log: log.ToString(),
                        error: error.ToString());
                });

            timer.Stop();

            BuildLogTextBox.Text =
                result.log;

            if (result.code != 0)
            {
                throw new InvalidOperationException(
                    $"ONNX → Engine 构建失败：{result.error}");
            }

            if (!File.Exists(enginePath))
            {
                throw new InvalidOperationException(
                    "TensorRT 返回成功，但未找到输出 Engine 文件。");
            }

            if (cacheDescriptor is not null)
            {
                await EngineCacheManager.WriteMetadataAsync(
                    cacheDescriptor,
                    result.log);

                CacheStatusTextBlock.Text =
                    $"缓存已写入：{System.IO.Path.GetFileName(enginePath)}";

                UpdateCacheStats();
            }

            EnginePathTextBox.Text =
                enginePath;

            InputWidthTextBox.Text =
                inputWidth.ToString(
                    CultureInfo.InvariantCulture);

            InputHeightTextBox.Text =
                inputHeight.ToString(
                    CultureInfo.InvariantCulture);

            StatusTextBlock.Text =
                $"Engine 构建成功 | 总耗时 {timer.Elapsed.TotalSeconds:0.0} s";

            MessageBox.Show(
                $"Engine 构建成功！{enginePath}"
                + (cacheDescriptor is not null
                    ? "已写入与当前 GPU / TensorRT / 模型配置绑定的本机缓存。"
                    : "")
                + "已自动填入下方 Engine 路径，可以直接点击“加载模型”。",
                "构建完成",
                MessageBoxButton.OK,
                MessageBoxImage.Information);
        }
        catch (DllNotFoundException ex)
        {
            MessageBox.Show(
                "无法加载 Native/TensorRT/CUDA DLL。"
                + "ONNX 构建还要求 nvonnxparser_10.dll 可被 Windows 找到。"
                + ex.Message,
                "DLL 加载失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            StatusTextBlock.Text =
                "Engine 构建失败";
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "Engine 构建失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            StatusTextBlock.Text =
                "Engine 构建失败";
        }
        finally
        {
            BuildEngineButton.IsEnabled =
                true;

            LoadModelButton.IsEnabled =
                true;

            DetectButton.IsEnabled =
                true;

            RefreshGpuInfoButton.IsEnabled =
                true;

            UpdateCacheStats();
        }
    }

    // ============================================================
    // Existing Engine inference workflow
    // ============================================================

    private void BrowseEngine_Click(
        object sender,
        RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择 TensorRT Engine",
            Filter = "TensorRT Engine (*.engine)|*.engine|所有文件 (*.*)|*.*"
        };

        if (dialog.ShowDialog() == true)
            EnginePathTextBox.Text =
                dialog.FileName;
    }

    private void BrowseImage_Click(
        object sender,
        RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择图片",
            Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff|所有文件 (*.*)|*.*"
        };

        if (dialog.ShowDialog() == true)
        {
            ImagePathTextBox.Text =
                dialog.FileName;

            LoadImage(dialog.FileName);
            ClearOverlay();
            DetectionGrid.ItemsSource = null;

            StatusTextBlock.Text =
                $"已加载图片：{_currentBitmap!.PixelWidth} × {_currentBitmap.PixelHeight}";
        }
    }

    private void LoadImage(
        string path)
    {
        using var stream =
            new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite);

        var decoder =
            BitmapDecoder.Create(
                stream,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad);

        BitmapSource source =
            decoder.Frames[0];

        var converted =
            new FormatConvertedBitmap();

        converted.BeginInit();
        converted.Source = source;
        converted.DestinationFormat =
            PixelFormats.Bgra32;
        converted.EndInit();
        converted.Freeze();

        _currentBitmap = converted;
        _currentStride =
            converted.PixelWidth * 4;

        _currentBgra =
            new byte[
                _currentStride
                * converted.PixelHeight];

        converted.CopyPixels(
            _currentBgra,
            _currentStride,
            0);

        PreviewImage.Source =
            converted;

        ImageSurface.Width =
            converted.PixelWidth;

        ImageSurface.Height =
            converted.PixelHeight;

        OverlayCanvas.Width =
            converted.PixelWidth;

        OverlayCanvas.Height =
            converted.PixelHeight;

        MaskOverlayImage.Width =
            converted.PixelWidth;

        MaskOverlayImage.Height =
            converted.PixelHeight;

        MaskOverlayImage.Source =
            null;
    }

    private void LoadModel_Click(
        object sender,
        RoutedEventArgs e)
    {
        try
        {
            LoadModel();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "加载失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            StatusTextBlock.Text =
                "模型加载失败";
        }
    }

    private void LoadModel()
    {
        string enginePath =
            EnginePathTextBox.Text.Trim();

        if (!File.Exists(enginePath))
        {
            throw new FileNotFoundException(
                "请选择有效的 .engine 文件。",
                enginePath);
        }

        if (!int.TryParse(
                InputWidthTextBox.Text,
                out int inputWidth)
            || inputWidth <= 0)
        {
            throw new InvalidOperationException(
                "Engine 输入宽度必须是正整数，例如 1280。");
        }

        if (!int.TryParse(
                InputHeightTextBox.Text,
                out int inputHeight)
            || inputHeight <= 0)
        {
            throw new InvalidOperationException(
                "Engine 输入高度必须是正整数，例如 512。");
        }

        DestroyDetector();

        var error =
            new StringBuilder(4096);

        _detector =
            NativeMethods.YoloCreate(
                enginePath,
                inputWidth,
                inputHeight,
                error,
                error.Capacity);

        if (_detector == IntPtr.Zero)
        {
            throw new InvalidOperationException(
                $"TensorRT 模型加载失败：\n{error}");
        }

        var info =
            new StringBuilder(4096);

        NativeMethods.YoloGetModelInfo(
            _detector,
            info,
            info.Capacity);

        int taskHint =
            NativeMethods.YoloGetTaskHint(
                _detector,
                _classNames.Length);

        string taskText;

        if (taskHint == 2)
        {
            ModelTaskComboBox.SelectedIndex = 2;
            taskText = "YOLO26 实例分割 Seg（prediction + proto 自动识别）";
        }
        else if (taskHint == 1)
        {
            ModelTaskComboBox.SelectedIndex = 1;
            taskText = "OBB 旋转框（根据输出通道自动识别）";
        }
        else if (taskHint == 0)
        {
            ModelTaskComboBox.SelectedIndex = 0;
            taskText = "普通 Detect（根据输出通道自动识别）";
        }
        else
        {
            taskText = "未自动识别，请手动选择 Detect / OBB / Seg";
        }

        ModelInfoTextBlock.Text =
            info.ToString()
            + $"\nTask: {taskText}";

        StatusTextBlock.Text =
            "模型加载成功";
    }

    private bool IsObbTask()
    {
        return ModelTaskComboBox.SelectedIndex == 1;
    }

    private bool IsSegTask()
    {
        return ModelTaskComboBox.SelectedIndex == 2;
    }

    private async void Detect_Click(
        object sender,
        RoutedEventArgs e)
    {
        try
        {
            if (_currentBitmap is null
                || _currentBgra is null)
            {
                throw new InvalidOperationException(
                    "请先选择一张图片。");
            }

            if (_detector == IntPtr.Zero)
                LoadModel();

            if (!float.TryParse(
                    ConfidenceTextBox.Text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float confidence))
            {
                throw new InvalidOperationException(
                    "置信度格式错误，例如 0.25。");
            }

            if (!float.TryParse(
                    NmsTextBox.Text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float nms))
            {
                throw new InvalidOperationException(
                    "NMS 阈值格式错误，例如 0.45。");
            }

            if (!float.TryParse(
                    MaskThresholdTextBox.Text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float maskThreshold)
                || maskThreshold <= 0.0f
                || maskThreshold >= 1.0f)
            {
                throw new InvalidOperationException(
                    "Mask 阈值必须在 0 和 1 之间，例如 0.50。");
            }

            DetectButton.IsEnabled = false;
            LoadModelButton.IsEnabled = false;
            BuildEngineButton.IsEnabled = false;

            bool segTask =
                IsSegTask();

            bool obbTask =
                IsObbTask();

            StatusTextBlock.Text =
                segTask
                    ? "正在执行 YOLO26 实例分割并计算最小外接矩形..."
                    : obbTask
                        ? "正在执行 OBB 旋转框推理..."
                        : "正在执行普通目标检测...";

            byte[] imageBytes =
                _currentBgra;

            int width =
                _currentBitmap.PixelWidth;

            int height =
                _currentBitmap.PixelHeight;

            int stride =
                _currentStride;

            IntPtr detector =
                _detector;

            var totalTimer =
                Stopwatch.StartNew();

            if (segTask)
            {
                NativeMethods.YoloSegDetection[] buffer =
                    new NativeMethods.YoloSegDetection[2048];

                // Original-resolution uint16 instance-id map.
                // 0 = background, 1..N = result.MaskId.
                ushort[] instanceMask =
                    new ushort[
                        width
                        * height];

                int expectedClassCount =
                    _classNames.Length;

                var result =
                    await Task.Run(() =>
                    {
                        var error =
                            new StringBuilder(8192);

                        int count =
                            NativeMethods.YoloDetectSegBgra(
                                detector,
                                imageBytes,
                                width,
                                height,
                                stride,
                                confidence,
                                nms,
                                maskThreshold,
                                expectedClassCount,
                                buffer,
                                buffer.Length,
                                instanceMask,
                                width,
                                out float inferenceMs,
                                error,
                                error.Capacity);

                        if (count < 0)
                        {
                            throw new InvalidOperationException(
                                $"Seg 推理失败：{error}");
                        }

                        return (
                            count,
                            inferenceMs);
                    });

                totalTimer.Stop();

                var detections =
                    buffer
                    .Take(result.count)
                    .ToArray();

                DrawSegDetections(
                    detections,
                    instanceMask,
                    width,
                    height);

                DetectionGrid.ItemsSource =
                    detections
                    .Select(
                        (d, i) =>
                        {
                            double angleDegrees =
                                d.AngleRadians
                                * 180.0
                                / Math.PI;

                            return new DetectionRow
                            {
                                Index = i + 1,
                                ClassName =
                                    GetClassName(
                                        d.ClassId),
                                ScoreText =
                                    d.Score.ToString("0.000"),
                                BoxText =
                                    $"MaskBox({d.X1:0},{d.Y1:0})-({d.X2:0},{d.Y2:0}) "
                                    + $"R={d.RotatedWidth:0}×{d.RotatedHeight:0}",
                                AngleText =
                                    $"{angleDegrees:0.0}°",
                                AreaText =
                                    $"{d.MaskAreaPixels:0} px"
                            };
                        })
                    .ToList();

                string classSummary =
                    string.Join(
                        ", ",
                        detections
                        .GroupBy(d => GetClassName(d.ClassId))
                        .Select(group => $"{group.Key}×{group.Count()}"));

                StatusTextBlock.Text =
                    $"Seg 完成：{result.count} 个实例"
                    + (string.IsNullOrWhiteSpace(classSummary)
                        ? ""
                        : $" | 分类：{classSummary}")
                    + $" | TensorRT {result.inferenceMs:0.00} ms"
                    + $" | 总耗时 {totalTimer.Elapsed.TotalMilliseconds:0.00} ms";
            }
            else if (obbTask)
            {
                NativeMethods.YoloObbDetection[] buffer =
                    new NativeMethods.YoloObbDetection[2048];

                int expectedClassCount =
                    _classNames.Length;

                var result =
                    await Task.Run(() =>
                    {
                        var error =
                            new StringBuilder(4096);

                        int count =
                            NativeMethods.YoloDetectObbBgra(
                                detector,
                                imageBytes,
                                width,
                                height,
                                stride,
                                confidence,
                                nms,
                                expectedClassCount,
                                buffer,
                                buffer.Length,
                                out float inferenceMs,
                                error,
                                error.Capacity);

                        if (count < 0)
                        {
                            throw new InvalidOperationException(
                                $"OBB 推理失败：{error}");
                        }

                        return (
                            count,
                            inferenceMs);
                    });

                totalTimer.Stop();

                var detections =
                    buffer.Take(
                        result.count)
                    .ToArray();

                DrawObbDetections(
                    detections);

                DetectionGrid.ItemsSource =
                    detections
                    .Select(
                        (d, i) =>
                        {
                            double angleDegrees =
                                d.AngleRadians
                                * 180.0
                                / Math.PI;

                            return new DetectionRow
                            {
                                Index = i + 1,
                                ClassName =
                                    GetClassName(
                                        d.ClassId),
                                ScoreText =
                                    d.Score.ToString("0.000"),
                                BoxText =
                                    $"C({d.CenterX:0},{d.CenterY:0}) "
                                    + $"{d.Width:0}×{d.Height:0}",
                                AngleText =
                                    $"{angleDegrees:0.0}°",
                                AreaText =
                                    "-"
                            };
                        })
                    .ToList();

                StatusTextBlock.Text =
                    $"OBB 检测完成：{result.count} 个目标"
                    + $" | TensorRT {result.inferenceMs:0.00} ms"
                    + $" | 总耗时 {totalTimer.Elapsed.TotalMilliseconds:0.00} ms";
            }
            else
            {
                NativeMethods.YoloDetection[] buffer =
                    new NativeMethods.YoloDetection[2048];

                var result =
                    await Task.Run(() =>
                    {
                        var error =
                            new StringBuilder(4096);

                        int count =
                            NativeMethods.YoloDetectBgra(
                                detector,
                                imageBytes,
                                width,
                                height,
                                stride,
                                confidence,
                                nms,
                                buffer,
                                buffer.Length,
                                out float inferenceMs,
                                error,
                                error.Capacity);

                        if (count < 0)
                        {
                            throw new InvalidOperationException(
                                $"推理失败：{error}");
                        }

                        return (
                            count,
                            inferenceMs);
                    });

                totalTimer.Stop();

                var detections =
                    buffer.Take(
                        result.count)
                    .ToArray();

                DrawDetections(
                    detections);

                DetectionGrid.ItemsSource =
                    detections
                    .Select(
                        (d, i) =>
                            new DetectionRow
                            {
                                Index = i + 1,
                                ClassName =
                                    GetClassName(
                                        d.ClassId),
                                ScoreText =
                                    d.Score.ToString("0.000"),
                                BoxText =
                                    $"{d.X1:0},{d.Y1:0} - {d.X2:0},{d.Y2:0}",
                                AngleText =
                                    "-",
                                AreaText =
                                    "-"
                            })
                    .ToList();

                StatusTextBlock.Text =
                    $"检测完成：{result.count} 个目标"
                    + $" | TensorRT {result.inferenceMs:0.00} ms"
                    + $" | 总耗时 {totalTimer.Elapsed.TotalMilliseconds:0.00} ms";
            }
        }
        catch (DllNotFoundException ex)
        {
            MessageBox.Show(
                "无法加载 YoloDeploy.Native.dll 或它依赖的 TensorRT/CUDA DLL。"
                + "请检查发布目录中的 TensorRT/CUDA Runtime。"
                + ex.Message,
                "DLL 加载失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        catch (BadImageFormatException ex)
        {
            MessageBox.Show(
                "DLL 位数不一致。请确保 WPF、Native DLL、TensorRT、CUDA 都是 x64。"
                + ex.Message,
                "架构错误",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.Message,
                "执行失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);

            StatusTextBlock.Text =
                "执行失败";
        }
        finally
        {
            DetectButton.IsEnabled = true;
            LoadModelButton.IsEnabled = true;
            BuildEngineButton.IsEnabled = true;
        }
    }

    private void DrawDetections(
        IEnumerable<NativeMethods.YoloDetection> detections)
    {
        ClearOverlay();

        double fontSize =
            Math.Max(
                12,
                Math.Min(
                    24,
                    ImageSurface.Width / 55.0));

        foreach (var d in detections)
        {
            double x = d.X1;
            double y = d.Y1;
            double w =
                Math.Max(
                    1,
                    d.X2 - d.X1);

            double h =
                Math.Max(
                    1,
                    d.Y2 - d.Y1);

            var rect =
                new Rectangle
                {
                    Width = w,
                    Height = h,
                    Stroke = Brushes.Lime,
                    StrokeThickness =
                        Math.Max(
                            2,
                            ImageSurface.Width / 500.0)
                };

            Canvas.SetLeft(rect, x);
            Canvas.SetTop(rect, y);

            OverlayCanvas.Children.Add(
                rect);

            string caption =
                $"{GetClassName(d.ClassId)} {d.Score:0.00}";

            var label =
                new Border
                {
                    Background = Brushes.Black,
                    Opacity = 0.80,
                    Padding =
                        new Thickness(
                            4, 1, 4, 1),
                    Child =
                        new TextBlock
                        {
                            Text = caption,
                            Foreground = Brushes.Lime,
                            FontSize = fontSize
                        }
                };

            Canvas.SetLeft(label, x);

            Canvas.SetTop(
                label,
                Math.Max(
                    0,
                    y - fontSize - 8));

            OverlayCanvas.Children.Add(
                label);
        }
    }

    private void DrawObbDetections(
        IEnumerable<NativeMethods.YoloObbDetection> detections)
    {
        ClearOverlay();

        double fontSize =
            Math.Max(
                12,
                Math.Min(
                    24,
                    ImageSurface.Width / 55.0));

        double strokeThickness =
            Math.Max(
                2,
                ImageSurface.Width / 500.0);

        foreach (var d in detections)
        {
            var points =
                new PointCollection
                {
                    new Point(d.P1X, d.P1Y),
                    new Point(d.P2X, d.P2Y),
                    new Point(d.P3X, d.P3Y),
                    new Point(d.P4X, d.P4Y)
                };

            var polygon =
                new Polygon
                {
                    Points = points,
                    Stroke = Brushes.Lime,
                    StrokeThickness = strokeThickness,
                    Fill = Brushes.Transparent
                };

            OverlayCanvas.Children.Add(
                polygon);

            double minX =
                points.Min(
                    p => p.X);

            double minY =
                points.Min(
                    p => p.Y);

            double angleDegrees =
                d.AngleRadians
                * 180.0
                / Math.PI;

            string caption =
                $"{GetClassName(d.ClassId)} "
                + $"{d.Score:0.00} "
                + $"{angleDegrees:0.0}°";

            var label =
                new Border
                {
                    Background = Brushes.Black,
                    Opacity = 0.80,
                    Padding =
                        new Thickness(
                            4, 1, 4, 1),
                    Child =
                        new TextBlock
                        {
                            Text = caption,
                            Foreground = Brushes.Lime,
                            FontSize = fontSize
                        }
                };

            Canvas.SetLeft(
                label,
                Math.Max(
                    0,
                    minX));

            Canvas.SetTop(
                label,
                Math.Max(
                    0,
                    minY - fontSize - 8));

            OverlayCanvas.Children.Add(
                label);
        }
    }

    private void DrawSegDetections(
        IReadOnlyList<NativeMethods.YoloSegDetection> detections,
        ushort[] instanceMask,
        int width,
        int height)
    {
        ClearOverlay();

        if (ShowMaskCheckBox.IsChecked == true)
        {
            DrawInstanceMask(
                instanceMask,
                width,
                height);
        }

        double fontSize =
            Math.Max(
                12,
                Math.Min(
                    24,
                    ImageSurface.Width / 55.0));

        double strokeThickness =
            Math.Max(
                2,
                ImageSurface.Width / 500.0);

        foreach (var d in detections)
        {
            if (ShowBBoxCheckBox.IsChecked == true)
            {
                var rect =
                    new Rectangle
                    {
                        Width =
                            Math.Max(
                                1,
                                d.X2 - d.X1),
                        Height =
                            Math.Max(
                                1,
                                d.Y2 - d.Y1),
                        Stroke =
                            Brushes.DeepSkyBlue,
                        StrokeThickness =
                            strokeThickness
                    };

                Canvas.SetLeft(
                    rect,
                    d.X1);

                Canvas.SetTop(
                    rect,
                    d.Y1);

                OverlayCanvas.Children.Add(
                    rect);
            }

            var rotatedPoints =
                new PointCollection
                {
                    new Point(d.P1X, d.P1Y),
                    new Point(d.P2X, d.P2Y),
                    new Point(d.P3X, d.P3Y),
                    new Point(d.P4X, d.P4Y)
                };

            if (ShowMinRectCheckBox.IsChecked == true)
            {
                var polygon =
                    new Polygon
                    {
                        Points =
                            rotatedPoints,
                        Stroke =
                            Brushes.Lime,
                        StrokeThickness =
                            strokeThickness,
                        Fill =
                            Brushes.Transparent
                    };

                OverlayCanvas.Children.Add(
                    polygon);
            }

            double minX =
                Math.Min(
                    d.X1,
                    rotatedPoints.Min(
                        point => point.X));

            double minY =
                Math.Min(
                    d.Y1,
                    rotatedPoints.Min(
                        point => point.Y));

            double angleDegrees =
                d.AngleRadians
                * 180.0
                / Math.PI;

            string caption =
                $"{GetClassName(d.ClassId)} "
                + $"{d.Score:0.00} "
                + $"A={d.MaskAreaPixels:0}px "
                + $"{angleDegrees:0.0}°";

            var label =
                new Border
                {
                    Background =
                        Brushes.Black,
                    Opacity =
                        0.82,
                    Padding =
                        new Thickness(
                            4,
                            1,
                            4,
                            1),
                    Child =
                        new TextBlock
                        {
                            Text =
                                caption,
                            Foreground =
                                Brushes.Lime,
                            FontSize =
                                fontSize
                        }
                };

            Canvas.SetLeft(
                label,
                Math.Max(
                    0,
                    minX));

            Canvas.SetTop(
                label,
                Math.Max(
                    0,
                    minY - fontSize - 8));

            OverlayCanvas.Children.Add(
                label);
        }
    }

    private void DrawInstanceMask(
        ushort[] instanceMask,
        int width,
        int height)
    {
        if (instanceMask.Length <
            width * height)
        {
            throw new InvalidOperationException(
                "实例 Mask 缓冲区大小不正确。");
        }

        int stride =
            width * 4;

        byte[] bgra =
            new byte[
                stride
                * height];

        for (int y = 0;
             y < height;
             ++y)
        {
            int maskRow =
                y * width;

            int pixelRow =
                y * stride;

            for (int x = 0;
                 x < width;
                 ++x)
            {
                ushort id =
                    instanceMask[
                        maskRow + x];

                if (id == 0)
                    continue;

                // Deterministic, dependency-free instance palette.
                byte red =
                    (byte)(
                        64
                        + (id * 73) % 192);

                byte green =
                    (byte)(
                        64
                        + (id * 151) % 192);

                byte blue =
                    (byte)(
                        64
                        + (id * 199) % 192);

                int offset =
                    pixelRow
                    + x * 4;

                bgra[offset + 0] =
                    blue;

                bgra[offset + 1] =
                    green;

                bgra[offset + 2] =
                    red;

                bgra[offset + 3] =
                    220;
            }
        }

        BitmapSource overlay =
            BitmapSource.Create(
                width,
                height,
                96,
                96,
                PixelFormats.Bgra32,
                null,
                bgra,
                stride);

        overlay.Freeze();

        MaskOverlayImage.Source =
            overlay;
    }

    private string GetClassName(
        int classId)
    {
        if (classId >= 0
            && classId < _classNames.Length)
        {
            return _classNames[classId];
        }

        return $"class_{classId}";
    }

    private void ClearOverlay()
    {
        OverlayCanvas.Children.Clear();
        MaskOverlayImage.Source = null;
    }

    private void DestroyDetector()
    {
        if (_detector != IntPtr.Zero)
        {
            NativeMethods.YoloDestroy(
                _detector);

            _detector =
                IntPtr.Zero;
        }
    }

    private void Window_Closing(
        object? sender,
        System.ComponentModel.CancelEventArgs e)
    {
        DestroyDetector();
    }
}
