@echo off
setlocal
cd /d "%~dp0"

dotnet sln YoloDeploy.sln add YoloDeploy.SDK\YoloDeploy.SDK.csproj
if errorlevel 1 exit /b %errorlevel%

dotnet sln YoloDeploy.sln add YoloDeploy.SDK.Demo\YoloDeploy.SDK.Demo.csproj
if errorlevel 1 exit /b %errorlevel%

echo.
echo SDK projects added to YoloDeploy.sln.
echo Build YoloDeploy.Native first, then build Release ^| x64.
pause
