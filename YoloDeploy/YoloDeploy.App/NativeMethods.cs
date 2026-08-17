using System.Runtime.InteropServices;
using System.Text;

namespace YoloDeploy.App;

internal static class NativeMethods
{
    private const string DllName = "YoloDeploy.Native.dll";

    [StructLayout(LayoutKind.Sequential)]
    internal struct YoloDetection
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public float Score;
        public int ClassId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct YoloObbDetection
    {
        public float CenterX;
        public float CenterY;
        public float Width;
        public float Height;
        public float AngleRadians;
        public float Score;
        public int ClassId;

        public float P1X;
        public float P1Y;
        public float P2X;
        public float P2Y;
        public float P3X;
        public float P3Y;
        public float P4X;
        public float P4Y;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct YoloSegDetection
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;

        public float Score;
        public int ClassId;

        public float MaskAreaPixels;

        public float CenterX;
        public float CenterY;
        public float RotatedWidth;
        public float RotatedHeight;
        public float AngleRadians;

        public float P1X;
        public float P1Y;
        public float P2X;
        public float P2Y;
        public float P3X;
        public float P3Y;
        public float P4X;
        public float P4Y;

        public int MaskId;
    }

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern IntPtr YoloCreate(
        string enginePath,
        int dynamicInputWidth,
        int dynamicInputHeight,
        StringBuilder errorBuffer,
        int errorCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl)]
    internal static extern int YoloGetTaskHint(
        IntPtr handle,
        int expectedClassCount);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloGetModelInfo(
        IntPtr handle,
        StringBuilder infoBuffer,
        int infoCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloDetectBgra(
        IntPtr handle,
        byte[] bgra,
        int width,
        int height,
        int stride,
        float confidenceThreshold,
        float nmsThreshold,
        [Out] YoloDetection[] results,
        int resultCapacity,
        out float inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloDetectObbBgra(
        IntPtr handle,
        byte[] bgra,
        int width,
        int height,
        int stride,
        float confidenceThreshold,
        float nmsThreshold,
        int expectedClassCount,
        [Out] YoloObbDetection[] results,
        int resultCapacity,
        out float inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloDetectSegBgra(
        IntPtr handle,
        byte[] bgra,
        int width,
        int height,
        int stride,
        float confidenceThreshold,
        float nmsThreshold,
        float maskThreshold,
        int expectedClassCount,
        [Out] YoloSegDetection[] results,
        int resultCapacity,
        [Out] ushort[] instanceMask,
        int maskStride,
        out float inferenceMilliseconds,
        StringBuilder errorBuffer,
        int errorCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl)]
    internal static extern void YoloDestroy(IntPtr handle);


    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloGetGpuInfoJson(
        StringBuilder jsonBuffer,
        int jsonCapacity,
        StringBuilder errorBuffer,
        int errorCapacity);

    [DllImport(
        DllName,
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Unicode)]
    internal static extern int YoloBuildEngineFromOnnx(
        string onnxPath,
        string enginePath,
        int inputWidth,
        int inputHeight,
        int enableFp16,
        int workspaceMiB,
        StringBuilder logBuffer,
        int logCapacity,
        StringBuilder errorBuffer,
        int errorCapacity);
}
