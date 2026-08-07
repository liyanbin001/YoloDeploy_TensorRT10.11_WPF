@echo off
rem ================================================================
rem Phase 3 build environment
rem
rem These values match the environment used during development.
rem Edit them only if your local install paths change.
rem ================================================================

if "%TENSORRT_ROOT%"=="" (
    set "TENSORRT_ROOT=D:\TensorRT-10.11.0.33\TensorRT-10.11.0.33"
)

if "%CUDA_PATH%"=="" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
)

rem Optional cuDNN root. Leave empty when not required.
if "%CUDNN_ROOT%"=="" (
    set "CUDNN_ROOT="
)

echo TENSORRT_ROOT=%TENSORRT_ROOT%
echo CUDA_PATH=%CUDA_PATH%
if not "%CUDNN_ROOT%"=="" echo CUDNN_ROOT=%CUDNN_ROOT%
