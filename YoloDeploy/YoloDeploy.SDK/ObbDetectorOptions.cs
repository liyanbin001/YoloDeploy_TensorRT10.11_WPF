using System;
using System.IO;
namespace YoloDeploy.SDK;

/// <summary>
/// OBB detector configuration. InputWidth/InputHeight are fixed for one detector instance.
/// </summary>
public sealed class ObbDetectorOptions
{
    /// <summary>
    /// Recommended: an ONNX model. A local TensorRT engine will be built/cached automatically.
    /// Existing .engine files are also accepted.
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// One class name per line, in the exact training/export order.
    /// </summary>
    public required string ClassNamesPath { get; init; }

    /// <summary>TensorRT model input width.</summary>
    public int InputWidth { get; init; } = 1280;

    /// <summary>TensorRT model input height.</summary>
    public int InputHeight { get; init; } = 512;

    /// <summary>Use TensorRT FP16 builder mode when building from ONNX.</summary>
    public bool EnableFp16 { get; init; } = true;

    /// <summary>TensorRT builder workspace limit in MiB.</summary>
    public int WorkspaceMiB { get; init; } = 1024;

    /// <summary>Default confidence threshold used by Detect().</summary>
    public float ConfidenceThreshold { get; init; } = 0.25f;

    /// <summary>Default rotated-NMS threshold used by Detect().</summary>
    public float NmsThreshold { get; init; } = 0.45f;

    /// <summary>Maximum number of detections copied out of the native layer.</summary>
    public int MaxResults { get; init; } = 2048;

    /// <summary>Ignore a valid cache and rebuild the engine from ONNX.</summary>
    public bool ForceRebuildEngine { get; init; } = false;
}
