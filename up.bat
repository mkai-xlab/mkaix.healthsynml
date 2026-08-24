@echo off
REM ---------------------------------------------------------------------------
REM  up.bat - start the knee OA inference API with Docker Compose (Windows)
REM  Equivalent of "make up". Double-click it, or run it from a terminal.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0"

set CONTAINER=knee-oa-ai
set NETWORK=knee-oa-net
set ENVFILE=..\env\ai.env
if "%AI_PORT%"=="" set AI_PORT=8005

echo.
echo === Knee OA API - starting ===
echo.

REM --- 1. Docker must be running -------------------------------------------
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Start Docker Desktop and try again.
    goto :fail
)

REM --- 2. The env file lives outside the repo, next to it -------------------
if not exist "%ENVFILE%" (
    echo [ERROR] Missing %ENVFILE%
    echo         It must sit beside this repo, i.e. ^<parent^>\env\ai.env
    goto :fail
)

REM --- 3. Compose declares the network as external, so create it if absent --
docker network inspect %NETWORK% >nul 2>&1
if errorlevel 1 (
    echo [ .. ] Creating network %NETWORK%
    docker network create %NETWORK% >nul
    if errorlevel 1 (
        echo [ERROR] Could not create network %NETWORK%
        goto :fail
    )
)

REM --- 4. Build and start ---------------------------------------------------
echo [ .. ] Building and starting the container.
echo        The first build downloads PyTorch and takes about 10 minutes.
echo.
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo [ERROR] docker compose failed. Full log:  docker compose logs ai
    goto :fail
)

REM --- 5. Wait for the healthcheck to pass ----------------------------------
echo.
echo [ .. ] Waiting for the service to report healthy.
set STATUS=
for /l %%i in (1,1,60) do (
    for /f "tokens=*" %%h in ('docker inspect -f "{{.State.Health.Status}}" %CONTAINER% 2^>nul') do set STATUS=%%h
    if "!STATUS!"=="healthy" goto :ready
    if "!STATUS!"=="unhealthy" (
        echo [ERROR] Container reported unhealthy. Log:  docker compose logs ai
        goto :fail
    )
    timeout /t 5 /nobreak >nul
)
echo [WARN] Still not healthy after 5 minutes. Check:  docker compose logs ai
goto :end

:ready
echo.
echo === Ready ===
echo   Health : http://localhost:%AI_PORT%/api/v1/health
echo   Docs   : http://localhost:%AI_PORT%/docs
echo.
echo   Logs   : docker compose logs -f ai
echo   Stop   : docker compose down
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
