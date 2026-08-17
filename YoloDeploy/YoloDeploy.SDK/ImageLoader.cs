using System;
using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace YoloDeploy.SDK;

internal sealed record BgraImage(
    byte[] Pixels,
    int Width,
    int Height,
    int Stride);

internal static class ImageLoader
{
    internal static BgraImage LoadBgra32(string imagePath)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException(
                "待检测图片不存在。",
                imagePath);
        }

        using FileStream stream = new(
            imagePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite);

        BitmapDecoder decoder = BitmapDecoder.Create(
            stream,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad);

        if (decoder.Frames.Count == 0)
        {
            throw new YoloSdkException(
                $"无法从图片读取像素：{imagePath}");
        }

        BitmapSource source = decoder.Frames[0];

        var converted = new FormatConvertedBitmap();
        converted.BeginInit();
        converted.Source = source;
        converted.DestinationFormat = PixelFormats.Bgra32;
        converted.EndInit();
        converted.Freeze();

        int width = converted.PixelWidth;
        int height = converted.PixelHeight;

        if (width <= 0 || height <= 0)
        {
            throw new YoloSdkException(
                $"图片尺寸无效：{width}x{height}");
        }

        int stride = checked(width * 4);
        byte[] bgra = new byte[checked(stride * height)];

        converted.CopyPixels(
            bgra,
            stride,
            0);

        return new BgraImage(
            bgra,
            width,
            height,
            stride);
    }
}
