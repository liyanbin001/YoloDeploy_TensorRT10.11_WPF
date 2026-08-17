[CmdletBinding()]
param(
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$PackageName = "YoloDeploy_v6_seg-minrect_win-x64",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Solution = Join-Path $Root "YoloDeploy.sln"
$NativeProject = Join-Path $Root "YoloDeploy.Native\YoloDeploy.Native.vcxproj"
$AppProject = Join-Path $Root "YoloDeploy.App\YoloDeploy.App.csproj"
$NativeDll = Join-Path $Root "artifacts\native\$Configuration\YoloDeploy.Native.dll"

$DistRoot = Join-Path $Root "dist"
$StagingRoot = Join-Path $DistRoot "_staging"
$PackageDir = Join-Path $DistRoot $PackageName
$ZipPath = Join-Path $DistRoot ($PackageName + ".zip")

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Require-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path $Path)) {
        throw "$Description not found: $Path"
    }
}

function Find-MSBuild {
    $cmd = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $vswhereCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($vswhere in $vswhereCandidates) {
        if (-not (Test-Path $vswhere)) {
            continue
        }

        $result = & $vswhere `
            -latest `
            -products * `
            -requires Microsoft.Component.MSBuild `
            -find "MSBuild\**\Bin\MSBuild.exe" |
            Select-Object -First 1

        if ($result -and (Test-Path $result)) {
            return $result
        }
    }

    throw "MSBuild.exe was not found. Install Visual Studio 2022 with Desktop development with C++."
}

function Copy-UniqueDlls {
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$Directories,

        [Parameter(Mandatory=$true)]
        [string[]]$Patterns,

        [Parameter(Mandatory=$true)]
        [string]$Destination,

        [Parameter(Mandatory=$true)]
        [string]$GroupName
    )

    $copied = @{}
    $count = 0

    foreach ($directory in $Directories) {
        if ([string]::IsNullOrWhiteSpace($directory) -or
            -not (Test-Path $directory)) {
            continue
        }

        foreach ($pattern in $Patterns) {
            Get-ChildItem `
                -Path $directory `
                -Filter $pattern `
                -File `
                -ErrorAction SilentlyContinue |
            ForEach-Object {
                $key = $_.Name.ToLowerInvariant()

                if (-not $copied.ContainsKey($key)) {
                    Copy-Item $_.FullName $Destination -Force
                    $copied[$key] = $_.FullName
                    $count++
                    Write-Host "  + $($_.Name)"
                }
            }
        }
    }

    if ($count -eq 0) {
        throw "No $GroupName runtime DLLs were copied."
    }

    Write-Host "$GroupName DLL count: $count"
    return $copied
}

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $null
    }

    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-GitCommit {
    try {
        $git = Get-Command git.exe -ErrorAction SilentlyContinue
        if (-not $git) {
            return $null
        }

        $value = (& git -C $Root rev-parse HEAD 2>$null).Trim()
        if ($LASTEXITCODE -eq 0 -and $value) {
            return $value
        }
    }
    catch {
    }

    return $null
}

Write-Step "1/8 Validate environment"

Require-Path $Solution "Solution"
Require-Path $NativeProject "Native project"
Require-Path $AppProject "WPF project"

if ([string]::IsNullOrWhiteSpace($env:TENSORRT_ROOT)) {
    throw "TENSORRT_ROOT is not set. Edit release_env.bat or set the environment variable."
}

if ([string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
    throw "CUDA_PATH is not set. Edit release_env.bat or set the environment variable."
}

Require-Path $env:TENSORRT_ROOT "TensorRT root"
Require-Path $env:CUDA_PATH "CUDA root"
Require-Path (Join-Path $env:TENSORRT_ROOT "include\NvInfer.h") "TensorRT header NvInfer.h"
Require-Path (Join-Path $env:TENSORRT_ROOT "include\NvOnnxParser.h") "TensorRT header NvOnnxParser.h"
Require-Path (Join-Path $env:CUDA_PATH "include\cuda_runtime.h") "CUDA header cuda_runtime.h"

$DotNet = (Get-Command dotnet.exe -ErrorAction Stop).Source
$MSBuild = Find-MSBuild

Write-Host "dotnet:  $DotNet"
Write-Host "MSBuild: $MSBuild"
Write-Host "TensorRT: $env:TENSORRT_ROOT"
Write-Host "CUDA:     $env:CUDA_PATH"

Write-Step "2/8 Clean previous release"

if (Test-Path $StagingRoot) {
    Remove-Item $StagingRoot -Recurse -Force
}

if (Test-Path $PackageDir) {
    Remove-Item $PackageDir -Recurse -Force
}

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

New-Item -ItemType Directory -Path $StagingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

Write-Step "3/8 Build native TensorRT DLL ($Configuration | x64)"

& $MSBuild `
    $NativeProject `
    "/t:Rebuild" `
    "/p:Configuration=$Configuration" `
    "/p:Platform=x64" `
    "/m" `
    "/nologo"

if ($LASTEXITCODE -ne 0) {
    throw "Native build failed with exit code $LASTEXITCODE."
}

Require-Path $NativeDll "YoloDeploy.Native.dll"
Write-Host "Native DLL: $NativeDll"

Write-Step "4/8 Publish WPF as .NET 8 self-contained win-x64"

& $DotNet publish `
    $AppProject `
    -c $Configuration `
    -r $RuntimeIdentifier `
    --self-contained true `
    --nologo `
    -p:PublishSingleFile=false `
    -p:PublishTrimmed=false `
    -p:PublishReadyToRun=false `
    -o $StagingRoot

if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE."
}

Require-Path (Join-Path $StagingRoot "YoloDeploy.App.exe") "Published application"

Copy-Item $NativeDll $StagingRoot -Force

Write-Step "5/8 Collect TensorRT and CUDA runtime DLLs"

# TensorRT 10.11 Windows ZIP commonly stores DLLs in lib.
# TensorRT 10.14+ moved Windows runtime DLLs to bin.
# Search both to make the release script layout-tolerant.
$TrtDirs = @(
    (Join-Path $env:TENSORRT_ROOT "lib"),
    (Join-Path $env:TENSORRT_ROOT "bin")
)

$TrtCopied = Copy-UniqueDlls `
    -Directories $TrtDirs `
    -Patterns @(
        "nvinfer*.dll",
        "nvonnxparser*.dll"
    ) `
    -Destination $StagingRoot `
    -GroupName "TensorRT"

$RequiredTrt = @(
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll"
)

foreach ($name in $RequiredTrt) {
    Require-Path (Join-Path $StagingRoot $name) "Required TensorRT runtime $name"
}

$CudaBin = Join-Path $env:CUDA_PATH "bin"
Require-Path $CudaBin "CUDA bin"

# Portable runtime set.  cudart is directly linked by YoloDeploy.Native.
# cublas/nvrtc/FFT/random DLLs are included because TensorRT builder/tactics
# can require additional CUDA user-mode libraries depending on the network.
$CudaCopied = Copy-UniqueDlls `
    -Directories @($CudaBin) `
    -Patterns @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll",
        "cufft64_*.dll",
        "curand64_*.dll"
    ) `
    -Destination $StagingRoot `
    -GroupName "CUDA"

$Cudart = Get-ChildItem `
    -Path $StagingRoot `
    -Filter "cudart64_*.dll" `
    -File `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $Cudart) {
    throw "cudart64_*.dll was not copied."
}

# Optional cuDNN deployment support.
if (-not [string]::IsNullOrWhiteSpace($env:CUDNN_ROOT) -and
    (Test-Path $env:CUDNN_ROOT)) {

    Write-Host ""
    Write-Host "Optional cuDNN runtime detected: $env:CUDNN_ROOT"

    $CudnnDirs = @(
        (Join-Path $env:CUDNN_ROOT "bin"),
        (Join-Path $env:CUDNN_ROOT "lib"),
        $env:CUDNN_ROOT
    )

    Copy-UniqueDlls `
        -Directories $CudnnDirs `
        -Patterns @("cudnn*.dll") `
        -Destination $StagingRoot `
        -GroupName "cuDNN" |
        Out-Null
}

Write-Step "6/8 Add models, launcher, verifier, docs and manifest"

$ModelsDir = Join-Path $StagingRoot "Models"
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

# Copy ONNX files if the repository contains a models/ or Models/ folder.
# Existing .engine files are intentionally NOT shipped by default:
# Phase 2 builds a GPU-specific engine on the target machine.
$ModelSources = @(
    (Join-Path $Root "models"),
    (Join-Path $Root "Models")
)

foreach ($modelSource in $ModelSources) {
    if (-not (Test-Path $modelSource)) {
        continue
    }

    Get-ChildItem `
        -Path $modelSource `
        -File `
        -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Extension -in @(".onnx", ".names", ".txt")
    } |
    ForEach-Object {
        Copy-Item $_.FullName $ModelsDir -Force
        Write-Host "  model + $($_.Name)"
    }
}

$ModelReadme = @'
把准备部署的 YOLO Detect、YOLO OBB 或 YOLO26-Seg ONNX 放到此目录。

推荐目标机第一次运行：
1. 双击 ..\run_YoloDeploy.bat
2. 选择本目录中的 .onnx
3. 保持“使用本机 Engine 缓存”勾选
4. 设置固定输入宽度和固定输入高度，例如 1280 × 512
5. 首次先用 FP32，确认检测正确后再测试 FP16
6. Workspace 建议 2048 MiB

不要默认把开发机生成的 .engine 当作跨 GPU 通用模型。
程序会在目标电脑的 %LOCALAPPDATA%\YoloDeploy\EngineCache 中生成并缓存本机 Engine。
'@
Set-Content `
    -Path (Join-Path $ModelsDir "README_MODEL_CN.txt") `
    -Value $ModelReadme `
    -Encoding UTF8

$Launcher = @'
@echo off
setlocal
cd /d "%~dp0"

rem Put the portable runtime folder first for this process only.
set "PATH=%~dp0;%PATH%"

start "" "%~dp0YoloDeploy.App.exe"
'@
Set-Content `
    -Path (Join-Path $StagingRoot "run_YoloDeploy.bat") `
    -Value $Launcher `
    -Encoding ASCII

$Verifier = @'
@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================================
echo YoloDeploy runtime verification
echo ================================================================
echo.

set "FAIL=0"

if exist "YoloDeploy.App.exe" (
  echo [OK] YoloDeploy.App.exe
) else (
  echo [ERROR] YoloDeploy.App.exe missing
  set "FAIL=1"
)

if exist "YoloDeploy.Native.dll" (
  echo [OK] YoloDeploy.Native.dll
) else (
  echo [ERROR] YoloDeploy.Native.dll missing
  set "FAIL=1"
)

if exist "nvinfer_10.dll" (
  echo [OK] nvinfer_10.dll
) else (
  echo [ERROR] nvinfer_10.dll missing
  set "FAIL=1"
)

if exist "nvinfer_plugin_10.dll" (
  echo [OK] nvinfer_plugin_10.dll
) else (
  echo [ERROR] nvinfer_plugin_10.dll missing
  set "FAIL=1"
)

if exist "nvonnxparser_10.dll" (
  echo [OK] nvonnxparser_10.dll
) else (
  echo [ERROR] nvonnxparser_10.dll missing
  set "FAIL=1"
)

dir /b cudart64_*.dll >nul 2>nul
if errorlevel 1 (
  echo [ERROR] CUDA Runtime cudart64_*.dll missing
  set "FAIL=1"
) else (
  echo [OK] CUDA Runtime
)

echo.
echo --- NVIDIA driver / GPU ---
where nvidia-smi.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nvidia-smi.exe not found.
  echo         Install a compatible NVIDIA display driver.
  set "FAIL=1"
) else (
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

echo.
echo --- VC++ runtime ---
if exist "%WINDIR%\System32\vcruntime140.dll" (
  echo [OK] vcruntime140.dll
) else (
  echo [WARNING] vcruntime140.dll not found in System32.
  echo           Install Microsoft Visual C++ 2015-2022 Redistributable x64.
)

echo.
if "%FAIL%"=="0" (
  echo BASIC CHECK PASSED.
  echo Run run_YoloDeploy.bat and build an Engine from ONNX for final validation.
) else (
  echo BASIC CHECK FAILED. Fix the errors above before launching the app.
)

echo.
pause
exit /b %FAIL%
'@
Set-Content `
    -Path (Join-Path $StagingRoot "verify_runtime.bat") `
    -Value $Verifier `
    -Encoding UTF8

$DeployReadme = @'
YoloDeploy Phase 3 - 目标电脑部署说明
======================================

目标：
- Windows x64
- NVIDIA GPU
- 通过 ONNX 在目标 GPU 上生成/缓存 TensorRT Engine
- 不要求目标电脑安装 Python / PyTorch / Ultralytics / Visual Studio / .NET 8

推荐使用步骤：
1. 整个发布目录原样复制/解压到目标电脑。
2. 安装兼容的 NVIDIA 驱动。
3. 建议安装 Microsoft Visual C++ 2015-2022 Redistributable x64。
4. 双击 verify_runtime.bat。
5. 检查无 ERROR。
6. 双击 run_YoloDeploy.bat。
7. 选择 Models 目录中的 .onnx。
8. 保持“使用本机 Engine 缓存”。
9. 第一次点击“生成 / 使用 Engine”会针对当前 GPU 构建 Engine。
10. 第二次相同模型/GPU/参数会直接命中缓存。

缓存位置：
%LOCALAPPDATA%\YoloDeploy\EngineCache

发布包已包含：
- .NET 8 win-x64 self-contained 文件
- YoloDeploy.Native.dll
- TensorRT runtime / ONNX parser DLL
- CUDA 用户态运行库（由发布脚本从 CUDA_PATH\bin 收集）
- coco.names
- Models 目录
- 运行/检查脚本

没有包含：
- NVIDIA 内核显示驱动；目标电脑必须安装 NVIDIA Driver。
- 你的训练环境；Python/PyTorch/Ultralytics 不需要。
- 默认不携带开发机 .engine；目标机应该从 ONNX 构建自己的 Engine。

注意：
- 请整体分发目录，不要只复制 YoloDeploy.App.exe。
- 发布包体积会明显大于普通 WPF 程序，因为包含 .NET Runtime、TensorRT 和部分 CUDA Runtime。
- 如果某个模型构建时报告缺少额外 CUDA/cuDNN DLL，可在开发机 release_env.bat 配置 CUDNN_ROOT 后重新发布，或补充相应 NVIDIA 可再分发运行库。
'@
Set-Content `
    -Path (Join-Path $StagingRoot "DEPLOY_README_CN.txt") `
    -Value $DeployReadme `
    -Encoding UTF8

$Commit = Get-GitCommit
$ImportantFiles = @(
    "YoloDeploy.App.exe",
    "YoloDeploy.Native.dll",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    $Cudart.Name
)

$FileEntries = @()

foreach ($fileName in $ImportantFiles | Select-Object -Unique) {
    $path = Join-Path $StagingRoot $fileName
    if (Test-Path $path) {
        $item = Get-Item $path
        $FileEntries += [ordered]@{
            name = $item.Name
            bytes = $item.Length
            sha256 = Get-FileSha256 $item.FullName
        }
    }
}

$Manifest = [ordered]@{
    schemaVersion = 1
    package = $PackageName
    createdUtc = [DateTime]::UtcNow.ToString("o")
    configuration = $Configuration
    runtimeIdentifier = $RuntimeIdentifier
    dotnetSelfContained = $true
    publishSingleFile = $false
    sourceGitCommit = $Commit
    tensorRtRootAtBuild = $env:TENSORRT_ROOT
    cudaPathAtBuild = $env:CUDA_PATH
    cudnnRootAtBuild = $env:CUDNN_ROOT
    importantFiles = $FileEntries
}

$Manifest |
    ConvertTo-Json -Depth 8 |
    Set-Content `
        -Path (Join-Path $StagingRoot "publish_manifest.json") `
        -Encoding UTF8

Write-Step "7/8 Finalize release folder"

Copy-Item `
    -Path (Join-Path $StagingRoot "*") `
    -Destination $PackageDir `
    -Recurse `
    -Force

$PackageSize = (
    Get-ChildItem $PackageDir -Recurse -File |
    Measure-Object Length -Sum
).Sum

Write-Host "Release folder: $PackageDir"
Write-Host ("Release size:   {0:N1} MiB" -f ($PackageSize / 1MB))

Write-Step "8/8 Create ZIP"

if (-not $NoZip) {
    Compress-Archive `
        -Path $PackageDir `
        -DestinationPath $ZipPath `
        -CompressionLevel Optimal `
        -Force

    Write-Host "ZIP: $ZipPath"
    Write-Host ("ZIP size: {0:N1} MiB" -f ((Get-Item $ZipPath).Length / 1MB))
}
else {
    Write-Host "ZIP creation skipped."
}

Remove-Item $StagingRoot -Recurse -Force

Write-Host ""
Write-Host "Phase 3 release completed successfully." -ForegroundColor Green
Write-Host "Distribute the entire release folder or ZIP, not only YoloDeploy.App.exe." -ForegroundColor Green
