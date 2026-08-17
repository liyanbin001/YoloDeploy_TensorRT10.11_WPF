param(
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$PackageName = "YoloDeploy.SDK.Runtime",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$NativeProject = Join-Path $Root "YoloDeploy.Native\YoloDeploy.Native.vcxproj"
$SdkProject = Join-Path $Root "YoloDeploy.SDK\YoloDeploy.SDK.csproj"
$TestProject = Join-Path $Root "YoloDeploy.SDK.Test\YoloDeploy.SDK.Test.csproj"
$AssetsDir = Join-Path $Root "sdk_runtime_assets"

$NativeDll = Join-Path $Root "artifacts\native\$Configuration\YoloDeploy.Native.dll"

$DistRoot = Join-Path $Root "dist"
$StagingRoot = Join-Path $DistRoot "_sdk_runtime_staging"
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

function Find-DotNet {
    $cmd = Get-Command dotnet.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidate = Join-Path $env:ProgramFiles "dotnet\dotnet.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    throw ".NET 8 SDK was not found."
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
                    $copied[$key] = $true
                    $count++
                }
            }
        }
    }

    Write-Host "$GroupName DLLs copied: $count"
    return $count
}

function Get-GitCommit {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue

    if (-not $git) {
        return ""
    }

    try {
        $commit = (& git -C $Root rev-parse HEAD 2>$null).Trim()
        return $commit
    }
    catch {
        return ""
    }
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-AnyFile([string]$Directory, [string]$Pattern, [string]$Description) {
    $item = Get-ChildItem `
        -Path $Directory `
        -Filter $Pattern `
        -File `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if (-not $item) {
        throw "$Description missing in package: $Pattern"
    }

    return $item
}

Write-Step "1/10 Validate SDK source and build environment"

Require-Path $NativeProject "Native project"
Require-Path $SdkProject "YoloDeploy.SDK project"
Require-Path $TestProject "TestSDK project"
Require-Path $AssetsDir "SDK runtime assets"

if ([string]::IsNullOrWhiteSpace($env:TENSORRT_ROOT)) {
    throw "TENSORRT_ROOT is not set. Example: D:\TensorRT-10.11.0.33"
}

if ([string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
    throw "CUDA_PATH is not set. Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
}

Require-Path $env:TENSORRT_ROOT "TENSORRT_ROOT"
Require-Path $env:CUDA_PATH "CUDA_PATH"

$MSBuild = Find-MSBuild
$DotNet = Find-DotNet

Write-Host "MSBuild       : $MSBuild"
Write-Host "dotnet        : $DotNet"
Write-Host "TensorRT root : $env:TENSORRT_ROOT"
Write-Host "CUDA path     : $env:CUDA_PATH"

Write-Step "2/10 Build YoloDeploy.Native ($Configuration | x64)"

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

Write-Step "3/10 Build YoloDeploy.SDK"

& $DotNet build `
    $SdkProject `
    -c $Configuration `
    -p:Platform=x64 `
    --nologo

if ($LASTEXITCODE -ne 0) {
    throw "YoloDeploy.SDK build failed with exit code $LASTEXITCODE."
}

Write-Step "4/10 Publish TestSDK"

$TestPublish = Join-Path $DistRoot "_sdk_test_publish"

if (Test-Path $TestPublish) {
    Remove-Item $TestPublish -Recurse -Force
}

& $DotNet publish `
    $TestProject `
    -c $Configuration `
    -r $RuntimeIdentifier `
    --self-contained false `
    --nologo `
    -p:Platform=x64 `
    -p:PublishSingleFile=false `
    -o $TestPublish

if ($LASTEXITCODE -ne 0) {
    throw "TestSDK publish failed with exit code $LASTEXITCODE."
}

Require-Path (Join-Path $TestPublish "TestSDK.exe") "TestSDK.exe"
Require-Path (Join-Path $TestPublish "YoloDeploy.SDK.dll") "YoloDeploy.SDK.dll"

Write-Step "5/10 Prepare customer package staging"

if (Test-Path $StagingRoot) {
    Remove-Item $StagingRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $StagingRoot | Out-Null

# Managed SDK + TestSDK managed launcher/runtimeconfig.
# Native runtime will be copied from the development installations below.
$ManagedAllowList = @(
    "YoloDeploy.SDK.dll",
    "YoloDeploy.SDK.xml",
    "TestSDK.exe",
    "TestSDK.dll",
    "TestSDK.deps.json",
    "TestSDK.runtimeconfig.json"
)

foreach ($name in $ManagedAllowList) {
    $source = Join-Path $TestPublish $name
    if (Test-Path $source) {
        Copy-Item $source $StagingRoot -Force
    }
}

Copy-Item $NativeDll $StagingRoot -Force

Copy-Item `
    -Path (Join-Path $AssetsDir "*") `
    -Destination $StagingRoot `
    -Recurse `
    -Force

Write-Step "6/10 Collect TensorRT runtime + ONNX parser DLLs"

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

foreach ($required in @(
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll"
)) {
    Require-Path `
        (Join-Path $StagingRoot $required) `
        "Required TensorRT DLL"
}

Write-Step "7/10 Collect portable CUDA user-mode runtime"

$CudaBin = Join-Path $env:CUDA_PATH "bin"
Require-Path $CudaBin "CUDA bin directory"

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

# Optional cuDNN redistribution, if the Native/TensorRT build in a specific
# environment needs it and CUDNN_ROOT is configured.
if (-not [string]::IsNullOrWhiteSpace($env:CUDNN_ROOT) -and
    (Test-Path $env:CUDNN_ROOT)) {

    $CudnnDirs = @(
        (Join-Path $env:CUDNN_ROOT "bin"),
        (Join-Path $env:CUDNN_ROOT "bin\12.0"),
        (Join-Path $env:CUDNN_ROOT "bin\11.0")
    )

    [void](Copy-UniqueDlls `
        -Directories $CudnnDirs `
        -Patterns @("cudnn*.dll") `
        -Destination $StagingRoot `
        -GroupName "cuDNN")
}

$Cudart = Require-AnyFile `
    $StagingRoot `
    "cudart64_*.dll" `
    "CUDA Runtime"

Write-Step "8/10 Validate final package contents"

foreach ($required in @(
    "YoloDeploy.SDK.dll",
    "YoloDeploy.Native.dll",
    "TestSDK.exe",
    "TestSDK.runtimeconfig.json",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    "verify_runtime.bat",
    "README_CUSTOMER_CN.txt"
)) {
    Require-Path `
        (Join-Path $StagingRoot $required) `
        "Runtime file"
}

# Make sure no compiled Engine is accidentally shipped from the developer PC.
$engines = Get-ChildItem `
    -Path $StagingRoot `
    -Recurse `
    -Filter "*.engine" `
    -File `
    -ErrorAction SilentlyContinue

if ($engines) {
    throw "One or more .engine files are present in the package. Remove them; customer machines should build Engine from ONNX."
}

Write-Step "9/10 Generate package manifest + SHA256 list"

$Commit = Get-GitCommit

$Files = Get-ChildItem `
    -Path $StagingRoot `
    -Recurse `
    -File |
    Sort-Object FullName

$ManifestEntries = @()
$ShaLines = @()

foreach ($file in $Files) {
    $relative = [IO.Path]::GetRelativePath(
        $StagingRoot,
        $file.FullName
    ).Replace("\", "/")

    $hash = Get-FileSha256 $file.FullName

    $ManifestEntries += [ordered]@{
        path = $relative
        bytes = $file.Length
        sha256 = $hash
    }

    $ShaLines += "$hash  $relative"
}

$Manifest = [ordered]@{
    schemaVersion = 1
    package = $PackageName
    createdUtc = [DateTime]::UtcNow.ToString("o")
    configuration = $Configuration
    runtimeIdentifier = $RuntimeIdentifier

    managedSdk = "YoloDeploy.SDK.dll"
    nativeBridge = "YoloDeploy.Native.dll"
    modelDelivery = "ONNX"
    enginePolicy = "Build and cache on target machine"
    engineCache = "%LOCALAPPDATA%\YoloDeploy\EngineCache"

    testSdkFrameworkDependent = $true
    requiredDotNetRuntime = ".NET 8 Windows Desktop Runtime x64"
    requiredTargetDriver = "Compatible NVIDIA display driver"

    sourceGitCommit = $Commit
    tensorRtRootAtBuild = $env:TENSORRT_ROOT
    cudaPathAtBuild = $env:CUDA_PATH
    cudnnRootAtBuild = $env:CUDNN_ROOT

    files = $ManifestEntries
}

$Manifest |
    ConvertTo-Json -Depth 8 |
    Set-Content `
        -Path (Join-Path $StagingRoot "runtime_manifest.json") `
        -Encoding UTF8

$ShaLines |
    Set-Content `
        -Path (Join-Path $StagingRoot "SHA256SUMS.txt") `
        -Encoding ASCII

Write-Step "10/10 Create customer directory and ZIP"

if (Test-Path $PackageDir) {
    Remove-Item $PackageDir -Recurse -Force
}

Copy-Item `
    -Path $StagingRoot `
    -Destination $PackageDir `
    -Recurse `
    -Force

if (-not $NoZip) {
    if (Test-Path $ZipPath) {
        Remove-Item $ZipPath -Force
    }

    Compress-Archive `
        -Path $PackageDir `
        -DestinationPath $ZipPath `
        -CompressionLevel Optimal `
        -Force

    Write-Host ""
    Write-Host "Customer ZIP created:" -ForegroundColor Green
    Write-Host $ZipPath -ForegroundColor Green
    Write-Host ("ZIP size: {0:N1} MiB" -f ((Get-Item $ZipPath).Length / 1MB))
}

Remove-Item $StagingRoot -Recurse -Force
Remove-Item $TestPublish -Recurse -Force

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YoloDeploy.SDK.Runtime publish completed successfully." -ForegroundColor Green
Write-Host "Distribute the entire Runtime ZIP. Do NOT send only YoloDeploy.SDK.dll." -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
