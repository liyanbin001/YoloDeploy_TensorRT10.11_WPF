using System.Text;
using System.Text.Json;

namespace YoloDeploy.SDK;

internal sealed class GpuInfo
{
    public int DeviceIndex { get; set; }
    public int DeviceCount { get; set; }
    public string Name { get; set; } = "";
    public int ComputeCapabilityMajor { get; set; }
    public int ComputeCapabilityMinor { get; set; }
    public ulong TotalGlobalMemoryBytes { get; set; }
    public int MultiProcessorCount { get; set; }
    public int CudaRuntimeVersion { get; set; }
    public int CudaDriverVersion { get; set; }
    public int TensorRtMajor { get; set; }
    public int TensorRtMinor { get; set; }
    public int TensorRtPatch { get; set; }
    public int TensorRtBuild { get; set; }

    public YoloRuntimeInfo ToPublic() => new()
    {
        GpuName = Name,
        ComputeCapabilityMajor = ComputeCapabilityMajor,
        ComputeCapabilityMinor = ComputeCapabilityMinor,
        TotalGlobalMemoryBytes = TotalGlobalMemoryBytes,
        MultiProcessorCount = MultiProcessorCount,
        CudaRuntimeVersion = CudaRuntimeVersion,
        CudaDriverVersion = CudaDriverVersion,
        TensorRtMajor = TensorRtMajor,
        TensorRtMinor = TensorRtMinor,
        TensorRtPatch = TensorRtPatch,
        TensorRtBuild = TensorRtBuild
    };
}

internal static class GpuInfoProvider
{
    internal static GpuInfo Query()
    {
        try
        {
            var json = new StringBuilder(8192);
            var error = new StringBuilder(4096);

            int code = NativeMethods.YoloGetGpuInfoJson(
                json,
                json.Capacity,
                error,
                error.Capacity);

            if (code != 0)
            {
                throw new YoloSdkException(
                    $"读取 GPU / CUDA / TensorRT 信息失败：{error}");
            }

            GpuInfo? info = JsonSerializer.Deserialize<GpuInfo>(
                json.ToString(),
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

            return info ?? throw new YoloSdkException(
                "YoloDeploy.Native.dll 返回的 GPU 信息 JSON 无法解析。");
        }
        catch (DllNotFoundException ex)
        {
            throw new YoloSdkException(
                "无法加载 YoloDeploy.Native.dll 或其 TensorRT/CUDA 依赖。"
                + "请确保 SDK 发布目录中的原生 DLL 与宿主程序 exe 位于同一目录，"
                + "并确保目标机安装了兼容的 NVIDIA 驱动。",
                ex);
        }
        catch (BadImageFormatException ex)
        {
            throw new YoloSdkException(
                "Native DLL 位数不匹配。当前 SDK 只支持 Windows x64。",
                ex);
        }
    }
}
