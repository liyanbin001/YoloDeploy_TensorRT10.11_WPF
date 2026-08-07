using Microsoft.Win32;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Controls;


namespace YoloDeploy.App;

public partial class MainWindow : Window
{
    private IntPtr _detector = IntPtr.Zero;
    private BitmapSource? _currentBitmap;
    private byte[]? _currentBgra;
    private int _currentStride;
    private string[] _classNames = Array.Empty<string>();

    public MainWindow()
    {
        InitializeComponent();
        LoadClassNames();
    }

    private void LoadClassNames()
    {
        string path = System.IO.Path.Combine(AppContext.BaseDirectory, "coco.names");
        _classNames = File.Exists(path)
            ? File.ReadAllLines(path)
                .Select(x => x.Trim())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .ToArray()
            : Array.Empty<string>();
    }

    private void BrowseEngine_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择 TensorRT Engine",
            Filter = "TensorRT Engine (*.engine)|*.engine|所有文件 (*.*)|*.*"
        };

        if (dialog.ShowDialog() == true)
            EnginePathTextBox.Text = dialog.FileName;
    }

    private void BrowseImage_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择图片",
            Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff|所有文件 (*.*)|*.*"
        };

        if (dialog.ShowDialog() == true)
        {
            ImagePathTextBox.Text = dialog.FileName;
            LoadImage(dialog.FileName);
            ClearOverlay();
            DetectionGrid.ItemsSource = null;
            StatusTextBlock.Text = $"已加载图片：{_currentBitmap!.PixelWidth} × {_currentBitmap.PixelHeight}";
        }
    }

    private void LoadImage(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);

        var decoder = BitmapDecoder.Create(
            stream,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad);

        BitmapSource source = decoder.Frames[0];

        var converted = new FormatConvertedBitmap();
        converted.BeginInit();
        converted.Source = source;
        converted.DestinationFormat = PixelFormats.Bgra32;
        converted.EndInit();
        converted.Freeze();

        _currentBitmap = converted;
        _currentStride = converted.PixelWidth * 4;
        _currentBgra = new byte[_currentStride * converted.PixelHeight];
        converted.CopyPixels(_currentBgra, _currentStride, 0);

        PreviewImage.Source = converted;
        ImageSurface.Width = converted.PixelWidth;
        ImageSurface.Height = converted.PixelHeight;
        OverlayCanvas.Width = converted.PixelWidth;
        OverlayCanvas.Height = converted.PixelHeight;
    }

    private void LoadModel_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            LoadModel();
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "加载失败", MessageBoxButton.OK, MessageBoxImage.Error);
            StatusTextBlock.Text = "模型加载失败";
        }
    }

    private void LoadModel()
    {
        string enginePath = EnginePathTextBox.Text.Trim();
        if (!File.Exists(enginePath))
            throw new FileNotFoundException("请选择有效的 .engine 文件。", enginePath);

        if (!int.TryParse(InputSizeTextBox.Text, out int dynamicSize) || dynamicSize <= 0)
            throw new InvalidOperationException("动态输入尺寸必须是正整数，例如 640。");

        DestroyDetector();

        var error = new StringBuilder(4096);
        _detector = NativeMethods.YoloCreate(
            enginePath,
            dynamicSize,
            dynamicSize,
            error,
            error.Capacity);

        if (_detector == IntPtr.Zero)
            throw new InvalidOperationException($"TensorRT 模型加载失败：\n{error}");

        var info = new StringBuilder(4096);
        NativeMethods.YoloGetModelInfo(_detector, info, info.Capacity);
        ModelInfoTextBlock.Text = info.ToString();

        StatusTextBlock.Text = "模型加载成功";
    }

    private async void Detect_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_currentBitmap is null || _currentBgra is null)
                throw new InvalidOperationException("请先选择一张图片。");

            if (_detector == IntPtr.Zero)
                LoadModel();

            if (!float.TryParse(
                    ConfidenceTextBox.Text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float confidence))
                throw new InvalidOperationException("置信度格式错误，例如 0.25。");

            if (!float.TryParse(
                    NmsTextBox.Text,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float nms))
                throw new InvalidOperationException("NMS IoU 格式错误，例如 0.45。");

            DetectButton.IsEnabled = false;
            LoadModelButton.IsEnabled = false;
            StatusTextBlock.Text = "正在推理...";

            var imageBytes = _currentBgra;
            int width = _currentBitmap.PixelWidth;
            int height = _currentBitmap.PixelHeight;
            int stride = _currentStride;
            IntPtr detector = _detector;

            NativeMethods.YoloDetection[] buffer = new NativeMethods.YoloDetection[2048];

            var totalTimer = Stopwatch.StartNew();

            var result = await Task.Run(() =>
            {
                var error = new StringBuilder(4096);

                int count = NativeMethods.YoloDetectBgra(
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
                    throw new InvalidOperationException($"推理失败：\n{error}");

                return (count, inferenceMs);
            });

            totalTimer.Stop();

            var detections = buffer.Take(result.count).ToArray();
            DrawDetections(detections);

            DetectionGrid.ItemsSource = detections
                .Select((d, i) => new DetectionRow
                {
                    Index = i + 1,
                    ClassName = GetClassName(d.ClassId),
                    ScoreText = d.Score.ToString("0.000"),
                    BoxText = $"{d.X1:0},{d.Y1:0} - {d.X2:0},{d.Y2:0}"
                })
                .ToList();

            StatusTextBlock.Text =
                $"检测完成：{result.count} 个目标 | TensorRT {result.inferenceMs:0.00} ms | 总耗时 {totalTimer.Elapsed.TotalMilliseconds:0.00} ms";
        }
        catch (DllNotFoundException ex)
        {
            MessageBox.Show(
                "无法加载 YoloDeploy.Native.dll 或它依赖的 TensorRT/CUDA DLL。\n\n" +
                "请检查 TENSORRT_ROOT、CUDA_PATH 和 PATH。\n\n" + ex.Message,
                "DLL 加载失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        catch (BadImageFormatException ex)
        {
            MessageBox.Show(
                "DLL 位数不一致。请确保 WPF、Native DLL、TensorRT、CUDA 都是 x64。\n\n" + ex.Message,
                "架构错误",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "执行失败", MessageBoxButton.OK, MessageBoxImage.Error);
            StatusTextBlock.Text = "执行失败";
        }
        finally
        {
            DetectButton.IsEnabled = true;
            LoadModelButton.IsEnabled = true;
        }
    }

    private void DrawDetections(IEnumerable<NativeMethods.YoloDetection> detections)
    {
        ClearOverlay();

        double fontSize = Math.Max(12, Math.Min(24, ImageSurface.Width / 55.0));

        foreach (var d in detections)
        {
            double x = d.X1;
            double y = d.Y1;
            double w = Math.Max(1, d.X2 - d.X1);
            double h = Math.Max(1, d.Y2 - d.Y1);

            var rect = new Rectangle
            {
                Width = w,
                Height = h,
                Stroke = Brushes.Lime,
                StrokeThickness = Math.Max(2, ImageSurface.Width / 500.0)
            };
            Canvas.SetLeft(rect, x);
            Canvas.SetTop(rect, y);
            OverlayCanvas.Children.Add(rect);

            string caption = $"{GetClassName(d.ClassId)} {d.Score:0.00}";
            var label = new Border
            {
                Background = Brushes.Black,
                Opacity = 0.80,
                Padding = new Thickness(4, 1, 4, 1),
                Child = new TextBlock
                {
                    Text = caption,
                    Foreground = Brushes.Lime,
                    FontSize = fontSize
                }
            };
            Canvas.SetLeft(label, x);
            Canvas.SetTop(label, Math.Max(0, y - fontSize - 8));
            OverlayCanvas.Children.Add(label);
        }
    }

    private string GetClassName(int classId)
    {
        if (classId >= 0 && classId < _classNames.Length)
            return _classNames[classId];

        return $"class_{classId}";
    }

    private void ClearOverlay() => OverlayCanvas.Children.Clear();

    private void DestroyDetector()
    {
        if (_detector != IntPtr.Zero)
        {
            NativeMethods.YoloDestroy(_detector);
            _detector = IntPtr.Zero;
        }
    }

    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        DestroyDetector();
    }
}
