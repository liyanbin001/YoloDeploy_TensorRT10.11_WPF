@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo YoloDeploy SDK OBB Test
echo ================================================================
echo.
echo 请先将：
echo   Models\best.onnx
echo   Models\classes.names
echo 放入 Models 目录。
echo.
echo 然后把下面 TEST_IMAGE 改成真实图片路径。
echo.

set "TEST_IMAGE=D:\Images\001.jpg"
set "INPUT_W=1280"
set "INPUT_H=512"

if not exist "Models\best.onnx" (
  echo [ERROR] Models\best.onnx not found.
  pause
  exit /b 1
)

if not exist "Models\classes.names" (
  echo [ERROR] Models\classes.names not found.
  pause
  exit /b 1
)

if not exist "%TEST_IMAGE%" (
  echo [ERROR] Test image not found: %TEST_IMAGE%
  echo Please edit TEST_IMAGE in run_test_example.bat.
  pause
  exit /b 1
)

TestSDK.exe "Models\best.onnx" "Models\classes.names" "%TEST_IMAGE%" %INPUT_W% %INPUT_H% 0.25 0.45

echo.
pause
