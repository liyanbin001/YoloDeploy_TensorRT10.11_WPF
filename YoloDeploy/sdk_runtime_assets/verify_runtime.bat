@echo off
setlocal
cd /d "%~dp0"

set "FAIL=0"

echo ================================================================
echo YoloDeploy SDK Runtime Verification
echo ================================================================
echo.

call :check_file "YoloDeploy.SDK.dll"
call :check_file "YoloDeploy.Native.dll"
call :check_file "nvinfer_10.dll"
call :check_file "nvinfer_plugin_10.dll"
call :check_file "nvonnxparser_10.dll"
call :check_file "TestSDK.exe"
call :check_file "TestSDK.runtimeconfig.json"

dir /b cudart64_*.dll >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cudart64_*.dll missing
  set "FAIL=1"
) else (
  echo [OK] CUDA Runtime
)

echo.
echo --- NVIDIA GPU / Driver ---
where nvidia-smi.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nvidia-smi.exe not found.
  echo         Install a compatible NVIDIA display driver.
  set "FAIL=1"
) else (
  nvidia-smi.exe --query-gpu=name,driver_version --format=csv,noheader
)

echo.
echo --- .NET 8 Desktop Runtime ---
where dotnet.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] dotnet.exe not found.
  echo         Install .NET 8 Windows Desktop Runtime x64.
  set "FAIL=1"
) else (
  dotnet --list-runtimes | findstr /i "Microsoft.WindowsDesktop.App 8." >nul
  if errorlevel 1 (
    echo [ERROR] Microsoft.WindowsDesktop.App 8.x not found.
    echo         Install .NET 8 Windows Desktop Runtime x64.
    set "FAIL=1"
  ) else (
    echo [OK] .NET 8 Windows Desktop Runtime
  )
)

echo.
if "%FAIL%"=="0" (
  echo ================================================================
  echo [OK] Basic runtime verification passed.
  echo ================================================================
) else (
  echo ================================================================
  echo [ERROR] Runtime verification failed.
  echo ================================================================
)

echo.
pause
exit /b %FAIL%

:check_file
if exist "%~1" (
  echo [OK] %~1
) else (
  echo [ERROR] %~1 missing
  set "FAIL=1"
)
exit /b 0
