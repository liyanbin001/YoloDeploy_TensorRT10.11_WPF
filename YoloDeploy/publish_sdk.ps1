param(
    [string]$Configuration = "Release",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Root "dist\YoloDeploy.SDK_win-x64"
}

if (-not $env:TENSORRT_ROOT) {
    throw "TENSORRT_ROOT is not set."
}

if (-not $env:CUDA_PATH) {
    throw "CUDA_PATH is not set."
}

$NativeDll = Join-Path $Root "artifacts\native\$Configuration\YoloDeploy.Native.dll"
if (-not (Test-Path $NativeDll)) {
    throw "Native DLL not found: $NativeDll`nBuild YoloDeploy.Native ($Configuration|x64) first."
}

Write-Host "[1/5] Build managed SDK..."
dotnet build `
    (Join-Path $Root "YoloDeploy.SDK\YoloDeploy.SDK.csproj") `
    -c $Configuration `
    -p:Platform=x64

if ($LASTEXITCODE -ne 0) {
    throw "dotnet build failed."
}

$SdkOut = Join-Path $Root "YoloDeploy.SDK\bin\x64\$Configuration\net8.0-windows"
if (-not (Test-Path $SdkOut)) {
    # SDK-style projects may omit the platform folder depending on local MSBuild settings.
    $SdkOut = Join-Path $Root "YoloDeploy.SDK\bin\$Configuration\net8.0-windows"
}

if (-not (Test-Path $SdkOut)) {
    throw "SDK build output not found."
}

Write-Host "[2/5] Prepare output folder..."
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

Copy-Item (Join-Path $SdkOut "YoloDeploy.SDK.dll") $OutputDir -Force
$Xml = Join-Path $SdkOut "YoloDeploy.SDK.xml"
if (Test-Path $Xml) {
    Copy-Item $Xml $OutputDir -Force
}
Copy-Item $NativeDll $OutputDir -Force

function Copy-Patterns {
    param(
        [string[]]$Directories,
        [string[]]$Patterns
    )

    foreach ($dir in $Directories) {
        if (-not (Test-Path $dir)) { continue }

        foreach ($pattern in $Patterns) {
            Get-ChildItem -Path $dir -Filter $pattern -File -ErrorAction SilentlyContinue |
                ForEach-Object {
                    Copy-Item $_.FullName $OutputDir -Force
                }
        }
    }
}

Write-Host "[3/5] Copy TensorRT runtime + ONNX parser..."
Copy-Patterns `
    -Directories @(
        (Join-Path $env:TENSORRT_ROOT "lib"),
        (Join-Path $env:TENSORRT_ROOT "bin")
    ) `
    -Patterns @(
        "nvinfer*.dll",
        "nvonnxparser*.dll"
    )

foreach ($required in @(
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll"
)) {
    $path = Join-Path $OutputDir $required
    if (-not (Test-Path $path)) {
        throw "Required TensorRT runtime missing after copy: $required"
    }
}

Write-Host "[4/5] Copy CUDA user-mode runtime..."
$CudaBin = Join-Path $env:CUDA_PATH "bin"

Copy-Patterns `
    -Directories @($CudaBin) `
    -Patterns @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll",
        "cufft64_*.dll",
        "curand64_*.dll"
    )

$Cudart = Get-ChildItem `
    -Path $OutputDir `
    -Filter "cudart64_*.dll" `
    -File `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $Cudart) {
    throw "CUDA Runtime cudart64_*.dll was not copied."
}

$Readme = @'
YoloDeploy.SDK Runtime
======================

调用方：
1. .NET 8 Windows x64 项目引用 YoloDeploy.SDK.dll
2. 运行时请把本目录所有 DLL 放到宿主 exe 同一目录
3. 模型建议只提供 .onnx + classes.names
4. 第一次初始化会在目标 GPU 上构建并缓存 TensorRT Engine
5. 后续相同模型/GPU/TensorRT/输入尺寸/精度将直接复用 Engine

目标机仍必须安装：
- 兼容的 NVIDIA display driver
- Microsoft Visual C++ 2015-2022 Redistributable x64（建议）
- 如果宿主应用不是 self-contained：.NET 8 Windows Desktop Runtime

不需要：
- Python
- PyTorch
- Ultralytics
- Visual Studio
- trtexec
- 手工生成 .engine

Engine Cache:
%LOCALAPPDATA%\YoloDeploy\EngineCache
'@

Set-Content `
    -Path (Join-Path $OutputDir "SDK_DEPLOY_README_CN.txt") `
    -Value $Readme `
    -Encoding UTF8

Write-Host "[5/5] Done."
Write-Host "SDK Runtime: $OutputDir"
