using System.Runtime.InteropServices;
using System.Text;

namespace YoloDeploy.SDK;

internal static class NativeMethods
{
    private const string DllName = "YoloDeploy.Native.dll";

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
