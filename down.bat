@echo off
REM ---------------------------------------------------------------------------
REM  down.bat - stop the knee OA inference API (Windows)
REM  Counterpart of up.bat. Removes the container; keeps images and the
REM  shared knee-oa-net network so the other services stay unaffected.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set CONTAINER=knee-oa-ai

echo.
echo === Knee OA API - stopping ===
echo.

REM --- Docker must be running ----------------------------------------------
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Nothing to stop.
    goto :fail
)

REM --- Report what is there before touching it ------------------------------
docker ps -a --filter "name=%CONTAINER%" --format "  found: {{.Names}} ({{.Status}})"
docker ps -aq --filter "name=%CONTAINER%" >nul 2>&1

REM --- Stop and remove the container ----------------------------------------
REM  "docker compose down" leaves images alone, and never removes a network
REM  declared external, so be/fe/rag keep working.
docker compose down
if errorlevel 1 (
    echo [ERROR] docker compose down failed.
    goto :fail
)

echo.
for /f %%c in ('docker ps -aq --filter "name=%CONTAINER%" 2^>nul ^| find /c /v ""') do set LEFT=%%c
if "%LEFT%"=="0" (
    echo === Stopped. Container removed. ===
) else (
    echo [WARN] A container named %CONTAINER% still exists.
    echo        Force it with:  docker rm -f %CONTAINER%
)

echo.
echo   Images kept.  Remove this one with:  docker rmi ml-ai
echo   Start again:                         up.bat
echo.
goto :end

:fail
echo.
endlocal
pause
exit /b 1

:end
endlocal
pause
exit /b 0
