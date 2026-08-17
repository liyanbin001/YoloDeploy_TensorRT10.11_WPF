@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo Build YoloDeploy.SDK.Runtime customer package
echo ================================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish_sdk_runtime.ps1"

if errorlevel 1 (
    echo.
    echo [ERROR] SDK Runtime publish failed.
    pause
    exit /b 1
)

echo.
echo [OK] Customer package:
echo     dist\YoloDeploy.SDK.Runtime.zip
echo.
pause
