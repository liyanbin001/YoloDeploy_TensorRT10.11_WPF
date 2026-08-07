using System.Text;
using System.Text.Json;

namespace YoloDeploy.App;

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

    public string ComputeCapability =>
        $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

    public string TensorRtVersion =>
        $"{TensorRtMajor}.{TensorRtMinor}.{TensorRtPatch}.{TensorRtBuild}";

    public double TotalMemoryGiB =>
        TotalGlobalMemoryBytes
        / 1024.0
        / 1024.0
        / 1024.0;

    public string CudaRuntimeDisplay =>
        FormatCudaVersion(
            CudaRuntimeVersion);

    public string CudaDriverDisplay =>
        FormatCudaVersion(
            CudaDriverVersion);

    public string DisplayText =>
        $"{Name} | CC {ComputeCapability} | "
        + $"{TotalMemoryGiB:0.0} GiB | SM {MultiProcessorCount} | "
        + $"CUDA Runtime {CudaRuntimeDisplay} | "
        + $"Driver API {CudaDriverDisplay} | "
        + $"TensorRT {TensorRtVersion}";

    public static string FormatCudaVersion(
        int version)
    {
        if (version <= 0)
            return "unknown";

        int major =
            version / 1000;

        int minor =
            (version % 1000) / 10;

        int patch =
            version % 10;

        return patch == 0
            ? $"{major}.{minor}"
            : $"{major}.{minor}.{patch}";
    }
}

internal static class GpuInfoProvider
{
    internal static GpuInfo Query()
    {
        var json =
            new StringBuilder(8192);

        var error =
            new StringBuilder(4096);

        int code =
            NativeMethods.YoloGetGpuInfoJson(
                json,
                json.Capacity,
                error,
                error.Capacity);

        if (code != 0)
        {
            throw new InvalidOperationException(
                $"读取 GPU 信息失败：\n{error}");
        }

        GpuInfo? info =
            JsonSerializer.Deserialize<GpuInfo>(
                json.ToString(),
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

        return info
            ?? throw new InvalidOperationException(
                "Native DLL 返回的 GPU 信息 JSON 无法解析。");
    }
}
