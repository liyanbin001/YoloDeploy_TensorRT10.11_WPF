@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

call "%~dp0release_env.bat"
if errorlevel 1 exit /b %errorlevel%

echo.
echo ================================================================
echo YoloDeploy Phase 3 - One-click Release
echo ================================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%~dp0publish_release.ps1"

set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo ================================================================
    echo RELEASE SUCCESS
    echo See the dist folder.
    echo ================================================================
) else (
    echo ================================================================
    echo RELEASE FAILED - exit code %EXITCODE%
    echo ================================================================
)

echo.
pause
exit /b %EXITCODE%
