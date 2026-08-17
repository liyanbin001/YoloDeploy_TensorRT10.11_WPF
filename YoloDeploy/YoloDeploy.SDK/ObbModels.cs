namespace YoloDeploy.SDK;

public readonly record struct ObbPoint(float X, float Y);

public sealed record ObbResult
{
    public required int ClassId { get; init; }
    public required string ClassName { get; init; }
    public required float Confidence { get; init; }

    public required float CenterX { get; init; }
    public required float CenterY { get; init; }
    public required float Width { get; init; }
    public required float Height { get; init; }

    public required float AngleRadians { get; init; }
    public float AngleDegrees => AngleRadians * 180.0f / MathF.PI;

    public required ObbPoint P1 { get; init; }
    public required ObbPoint P2 { get; init; }
    public required ObbPoint P3 { get; init; }
    public required ObbPoint P4 { get; init; }
}

public sealed record ObbDetectionResponse
{
    public required string ImagePath { get; init; }
    public required int ImageWidth { get; init; }
    public required int ImageHeight { get; init; }
    public required float InferenceMilliseconds { get; init; }
    public required IReadOnlyList<ObbResult> Detections { get; init; }
}

public sealed record YoloRuntimeInfo
{
    public required string GpuName { get; init; }
    public required int ComputeCapabilityMajor { get; init; }
    public required int ComputeCapabilityMinor { get; init; }
    public required ulong TotalGlobalMemoryBytes { get; init; }
    public required int MultiProcessorCount { get; init; }
    public required int CudaRuntimeVersion { get; init; }
    public required int CudaDriverVersion { get; init; }
    public required int TensorRtMajor { get; init; }
    public required int TensorRtMinor { get; init; }
    public required int TensorRtPatch { get; init; }
    public required int TensorRtBuild { get; init; }

    public string ComputeCapability =>
        $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

    public string TensorRtVersion =>
        $"{TensorRtMajor}.{TensorRtMinor}.{TensorRtPatch}.{TensorRtBuild}";

    public double TotalMemoryGiB =>
        TotalGlobalMemoryBytes / 1024.0 / 1024.0 / 1024.0;
}

public sealed record ObbDetectorInitializationInfo
{
    public required string ModelPath { get; init; }
    public required string EnginePath { get; init; }
    public required bool BuiltFromOnnx { get; init; }
    public required bool EngineCacheHit { get; init; }
    public required bool EngineBuiltNow { get; init; }
    public required string BuildLog { get; init; }
    public required YoloRuntimeInfo Runtime { get; init; }
}
