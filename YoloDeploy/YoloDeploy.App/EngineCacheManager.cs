using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.IO;

namespace YoloDeploy.App;

internal sealed class EngineCacheDescriptor
{
    public required string OnnxPath { get; init; }
    public required string OnnxSha256 { get; init; }
    public required string Precision { get; init; }
    public required int InputWidth { get; init; }
    public required int InputHeight { get; init; }
    public required int WorkspaceMiB { get; init; }
    public required GpuInfo Gpu { get; init; }
    public required string CacheKey { get; init; }
    public required string EnginePath { get; init; }
    public required string MetadataPath { get; init; }
}

internal sealed class EngineCacheMetadata
{
    public int SchemaVersion { get; set; } = 2;
    public string SourceOnnxPath { get; set; } = "";
    public string OnnxSha256 { get; set; } = "";
    public string Precision { get; set; } = "";
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public int WorkspaceMiB { get; set; }

    public string GpuName { get; set; } = "";
    public int ComputeCapabilityMajor { get; set; }
    public int ComputeCapabilityMinor { get; set; }
    public int MultiProcessorCount { get; set; }
    public ulong TotalGlobalMemoryBytes { get; set; }

    public int CudaRuntimeVersion { get; set; }
    public int CudaDriverVersion { get; set; }

    public int TensorRtMajor { get; set; }
    public int TensorRtMinor { get; set; }
    public int TensorRtPatch { get; set; }
    public int TensorRtBuild { get; set; }

    public long EngineLengthBytes { get; set; }
    public DateTime CreatedUtc { get; set; }
    public string BuildLog { get; set; } = "";
}

internal readonly record struct EngineCacheStats(
    int EngineCount,
    long TotalBytes)
{
    public string DisplayText =>
        $"{EngineCount} 个 Engine，{FormatBytes(TotalBytes)}";

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024)
            return $"{bytes} B";

        double value = bytes;
        string[] units = ["KiB", "MiB", "GiB", "TiB"];
        int index = -1;

        do
        {
            value /= 1024.0;
            index++;
        }
        while (value >= 1024.0 &&
               index < units.Length - 1);

        return $"{value:0.0} {units[index]}";
    }
}

internal static class EngineCacheManager
{
    private const int CurrentSchemaVersion = 2;

    private static readonly JsonSerializerOptions JsonOptions =
        new()
        {
            WriteIndented = true
        };

    internal static string CacheRoot { get; } =
        Path.Combine(
            Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData),
            "YoloDeploy",
            "EngineCache");

    internal static async Task<string> ComputeSha256Async(
        string filePath)
    {
        await using FileStream stream = new(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);

        byte[] hash =
            await SHA256.HashDataAsync(stream);

        return Convert.ToHexString(hash)
            .ToLowerInvariant();
    }

    internal static EngineCacheDescriptor CreateDescriptor(
        string onnxPath,
        string onnxSha256,
        GpuInfo gpu,
        string precision,
        int inputWidth,
        int inputHeight,
        int workspaceMiB)
    {
        Directory.CreateDirectory(CacheRoot);

        string stem =
            SanitizeToken(
                Path.GetFileNameWithoutExtension(
                    onnxPath));

        string gpuToken =
            SanitizeToken(gpu.Name);

        string hashToken =
            onnxSha256[
                ..Math.Min(
                    16,
                    onnxSha256.Length)];

        string precisionToken =
            SanitizeToken(
                precision.ToLowerInvariant());

        string cacheKey =
            $"{stem}_"
            + $"{hashToken}_"
            + $"{gpuToken}_"
            + $"cc{gpu.ComputeCapabilityMajor}{gpu.ComputeCapabilityMinor}_"
            + $"sm{gpu.MultiProcessorCount}_"
            + $"trt{gpu.TensorRtMajor}_{gpu.TensorRtMinor}_{gpu.TensorRtPatch}_{gpu.TensorRtBuild}_"
            + $"{precisionToken}_"
            + $"{inputWidth}x{inputHeight}_"
            + $"ws{workspaceMiB}";

        string enginePath =
            Path.Combine(
                CacheRoot,
                cacheKey + ".engine");

        return new EngineCacheDescriptor
        {
            OnnxPath =
                Path.GetFullPath(onnxPath),

            OnnxSha256 =
                onnxSha256,

            Precision =
                precision.ToUpperInvariant(),

            InputWidth =
                inputWidth,

            InputHeight =
                inputHeight,

            WorkspaceMiB =
                workspaceMiB,

            Gpu =
                gpu,

            CacheKey =
                cacheKey,

            EnginePath =
                enginePath,

            MetadataPath =
                enginePath + ".json"
        };
    }

    internal static bool TryValidate(
        EngineCacheDescriptor descriptor,
        out string reason)
    {
        reason = "";

        if (!File.Exists(
                descriptor.EnginePath))
        {
            reason =
                "Engine 文件不存在";

            return false;
        }

        if (!File.Exists(
                descriptor.MetadataPath))
        {
            reason =
                "缓存元数据不存在";

            return false;
        }

        try
        {
            string json =
                File.ReadAllText(
                    descriptor.MetadataPath);

            EngineCacheMetadata? metadata =
                JsonSerializer.Deserialize<EngineCacheMetadata>(
                    json,
                    JsonOptions);

            if (metadata is null)
            {
                reason =
                    "缓存元数据无法解析";

                return false;
            }

            if (metadata.SchemaVersion !=
                CurrentSchemaVersion)
            {
                reason =
                    "缓存格式版本已变化";

                return false;
            }

            if (!string.Equals(
                    metadata.OnnxSha256,
                    descriptor.OnnxSha256,
                    StringComparison.OrdinalIgnoreCase))
            {
                reason =
                    "ONNX 内容已变化";

                return false;
            }

            if (!string.Equals(
                    metadata.Precision,
                    descriptor.Precision,
                    StringComparison.OrdinalIgnoreCase))
            {
                reason =
                    "精度配置不同";

                return false;
            }

            if (metadata.InputWidth !=
                    descriptor.InputWidth ||
                metadata.InputHeight !=
                    descriptor.InputHeight)
            {
                reason =
                    "输入尺寸不同";

                return false;
            }

            if (metadata.WorkspaceMiB !=
                descriptor.WorkspaceMiB)
            {
                reason =
                    "Workspace 配置不同";

                return false;
            }

            GpuInfo gpu =
                descriptor.Gpu;

            if (!string.Equals(
                    metadata.GpuName,
                    gpu.Name,
                    StringComparison.OrdinalIgnoreCase) ||
                metadata.ComputeCapabilityMajor !=
                    gpu.ComputeCapabilityMajor ||
                metadata.ComputeCapabilityMinor !=
                    gpu.ComputeCapabilityMinor ||
                metadata.MultiProcessorCount !=
                    gpu.MultiProcessorCount)
            {
                reason =
                    "GPU 配置不同";

                return false;
            }

            if (metadata.TensorRtMajor !=
                    gpu.TensorRtMajor ||
                metadata.TensorRtMinor !=
                    gpu.TensorRtMinor ||
                metadata.TensorRtPatch !=
                    gpu.TensorRtPatch ||
                metadata.TensorRtBuild !=
                    gpu.TensorRtBuild)
            {
                reason =
                    "TensorRT 版本不同";

                return false;
            }

            long currentLength =
                new FileInfo(
                    descriptor.EnginePath)
                .Length;

            if (currentLength <= 0)
            {
                reason =
                    "Engine 文件为空";

                return false;
            }

            if (metadata.EngineLengthBytes > 0 &&
                metadata.EngineLengthBytes !=
                    currentLength)
            {
                reason =
                    "Engine 文件大小与元数据不一致";

                return false;
            }

            reason =
                "缓存有效";

            return true;
        }
        catch (Exception ex)
        {
            reason =
                $"缓存检查失败：{ex.Message}";

            return false;
        }
    }

    internal static async Task WriteMetadataAsync(
        EngineCacheDescriptor descriptor,
        string buildLog)
    {
        FileInfo engineFile =
            new(descriptor.EnginePath);

        if (!engineFile.Exists ||
            engineFile.Length <= 0)
        {
            throw new InvalidOperationException(
                "不能为不存在或为空的 Engine 写入缓存元数据。");
        }

        GpuInfo gpu =
            descriptor.Gpu;

        var metadata =
            new EngineCacheMetadata
            {
                SchemaVersion =
                    CurrentSchemaVersion,

                SourceOnnxPath =
                    descriptor.OnnxPath,

                OnnxSha256 =
                    descriptor.OnnxSha256,

                Precision =
                    descriptor.Precision,

                InputWidth =
                    descriptor.InputWidth,

                InputHeight =
                    descriptor.InputHeight,

                WorkspaceMiB =
                    descriptor.WorkspaceMiB,

                GpuName =
                    gpu.Name,

                ComputeCapabilityMajor =
                    gpu.ComputeCapabilityMajor,

                ComputeCapabilityMinor =
                    gpu.ComputeCapabilityMinor,

                MultiProcessorCount =
                    gpu.MultiProcessorCount,

                TotalGlobalMemoryBytes =
                    gpu.TotalGlobalMemoryBytes,

                CudaRuntimeVersion =
                    gpu.CudaRuntimeVersion,

                CudaDriverVersion =
                    gpu.CudaDriverVersion,

                TensorRtMajor =
                    gpu.TensorRtMajor,

                TensorRtMinor =
                    gpu.TensorRtMinor,

                TensorRtPatch =
                    gpu.TensorRtPatch,

                TensorRtBuild =
                    gpu.TensorRtBuild,

                EngineLengthBytes =
                    engineFile.Length,

                CreatedUtc =
                    DateTime.UtcNow,

                BuildLog =
                    buildLog
            };

        string json =
            JsonSerializer.Serialize(
                metadata,
                JsonOptions);

        await File.WriteAllTextAsync(
            descriptor.MetadataPath,
            json,
            Encoding.UTF8);
    }

    internal static EngineCacheStats GetStats()
    {
        Directory.CreateDirectory(
            CacheRoot);

        FileInfo[] engines =
            new DirectoryInfo(
                CacheRoot)
            .GetFiles(
                "*.engine",
                SearchOption.TopDirectoryOnly);

        return new EngineCacheStats(
            engines.Length,
            engines.Sum(
                x => x.Length));
    }

    internal static void ClearAll()
    {
        if (Directory.Exists(
                CacheRoot))
        {
            Directory.Delete(
                CacheRoot,
                recursive: true);
        }

        Directory.CreateDirectory(
            CacheRoot);
    }

    internal static void OpenCacheFolder()
    {
        Directory.CreateDirectory(
            CacheRoot);

        Process.Start(
            new ProcessStartInfo
            {
                FileName =
                    "explorer.exe",

                Arguments =
                    $"\"{CacheRoot}\"",

                UseShellExecute =
                    true
            });
    }

    internal static bool IsInsideCache(
        string? path)
    {
        if (string.IsNullOrWhiteSpace(
                path))
        {
            return false;
        }

        string cache =
            Path.GetFullPath(
                CacheRoot)
            .TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;

        string target =
            Path.GetFullPath(path);

        return target.StartsWith(
            cache,
            StringComparison.OrdinalIgnoreCase);
    }

    private static string SanitizeToken(
        string text)
    {
        string sanitized =
            Regex.Replace(
                text.Trim(),
                @"[^A-Za-z0-9._-]+",
                "_");

        sanitized =
            sanitized.Trim(
                '_',
                '.',
                '-');

        if (string.IsNullOrWhiteSpace(
                sanitized))
        {
            sanitized =
                "model";
        }

        return sanitized.Length <= 48
            ? sanitized
            : sanitized[..48];
    }
}
