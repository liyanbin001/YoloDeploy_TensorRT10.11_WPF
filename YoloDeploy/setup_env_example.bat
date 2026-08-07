@echo off
chcp 65001 >nul

echo ==========================================
echo   YoloDeploy Environment Setup
echo ==========================================
echo.

rem ============================================================
rem TensorRT 10.11
rem ============================================================
set "TENSORRT_ROOT=D:\TensorRT-10.11.0.33\TensorRT-10.11.0.33"

rem ============================================================
rem CUDA 12.3
rem ============================================================
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"

rem ============================================================
rem Optional: OpenCV 4.12
rem The current YoloDeploy project does NOT require OpenCV.
rem Keep these variables only for future C++ projects if needed.
rem ============================================================
set "OPENCV_ROOT=D:\opencv\build"
set "OPENCV_INCLUDE=D:\opencv\build\include"
set "OPENCV_LIB=D:\opencv\build\x64\vc16\lib"
set "OPENCV_BIN=D:\opencv\build\x64\vc16\bin"

rem ============================================================
rem Runtime PATH
rem ============================================================
set "PATH=%TENSORRT_ROOT%\bin;%TENSORRT_ROOT%\lib;%CUDA_PATH%\bin;%OPENCV_BIN%;%PATH%"

echo.
echo [TensorRT]
echo TENSORRT_ROOT=%TENSORRT_ROOT%
echo.

echo [CUDA]
echo CUDA_PATH=%CUDA_PATH%
echo.

echo [OpenCV]
echo OPENCV_ROOT=%OPENCV_ROOT%
echo.

echo ==========================================
echo Checking environment...
echo ==========================================
echo.

if exist "%TENSORRT_ROOT%\include\NvInfer.h" (
    echo [OK] TensorRT NvInfer.h found
) else (
    echo [ERROR] TensorRT NvInfer.h NOT found
)

if exist "%TENSORRT_ROOT%\lib\nvinfer_10.lib" (
    echo [OK] TensorRT nvinfer_10.lib found
) else (
    echo [ERROR] TensorRT nvinfer_10.lib NOT found
)

if exist "%TENSORRT_ROOT%\lib\nvonnxparser_10.lib" (
    echo [OK] TensorRT nvonnxparser_10.lib found
) else (
    echo [ERROR] TensorRT nvonnxparser_10.lib NOT found
)

if exist "%TENSORRT_ROOT%\lib\nvinfer_plugin_10.lib" (
    echo [OK] TensorRT nvinfer_plugin_10.lib found
) else (
    echo [ERROR] TensorRT nvinfer_plugin_10.lib NOT found
)

if exist "%CUDA_PATH%\include\cuda_runtime.h" (
    echo [OK] CUDA headers found
) else (
    echo [ERROR] CUDA headers NOT found
)

if exist "%CUDA_PATH%\lib\x64\cudart.lib" (
    echo [OK] CUDA cudart.lib found
) else (
    echo [ERROR] CUDA cudart.lib NOT found
)

if exist "%OPENCV_INCLUDE%\opencv2\opencv.hpp" (
    echo [OK] OpenCV headers found
) else (
    echo [WARNING] OpenCV headers NOT found
)

if exist "%OPENCV_LIB%\opencv_world4120.lib" (
    echo [OK] OpenCV Release library found
) else (
    echo [WARNING] opencv_world4120.lib NOT found
)

echo.
echo ==========================================
echo TensorRT version:
echo ==========================================

if exist "%TENSORRT_ROOT%\bin\trtexec.exe" (
    "%TENSORRT_ROOT%\bin\trtexec.exe" --version
) else (
    echo [ERROR] trtexec.exe NOT found
)

echo.
echo ==========================================
echo CUDA version:
echo ==========================================
nvcc --version

echo.
echo ==========================================
echo Environment setup finished.
echo Keep this window open when launching VS
echo if variables are not set permanently.
echo ==========================================
echo.

pause
echo.
echo Starting Visual Studio...

start "" "D:\visualstudio\Common7\IDE\devenv.exe" "%~dp0YoloDeploy.sln"

exit
