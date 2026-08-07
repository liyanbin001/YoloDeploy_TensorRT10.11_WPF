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
