@echo off
REM =============================================================================
REM RIPPLE in-place updater. Spawned by the app after the user confirms Update.
REM Waits for the Java process to exit, fast-forwards from official GitHub main,
REM rebuilds with quickstart, then relaunches.
REM =============================================================================
setlocal EnableDelayedExpansion

set "PID=%~1"
set "MODE=%~2"
if /i not "%MODE%"=="gpu" set "MODE=cpu"
if not defined RIPPLE_UPDATE_REMOTE set "RIPPLE_UPDATE_REMOTE=https://github.com/Le0nZim/ripple.git"
if not defined RIPPLE_UPDATE_BRANCH set "RIPPLE_UPDATE_BRANCH=main"
if not defined RIPPLE_UPDATE_WAIT_SECONDS set "RIPPLE_UPDATE_WAIT_SECONDS=120"
if not defined RIPPLE_UPDATE_SKIP_QUICKSTART set "RIPPLE_UPDATE_SKIP_QUICKSTART=0"
if not defined RIPPLE_UPDATE_SKIP_RELAUNCH set "RIPPLE_UPDATE_SKIP_RELAUNCH=0"

cd /d "%~dp0.."
set "REPO_ROOT=%cd%"
if not exist "tools" mkdir tools
set "LOG=%REPO_ROOT%\tools\ripple-update.log"

echo === RIPPLE update %DATE% %TIME% ===
echo === RIPPLE update %DATE% %TIME% ===>> "%LOG%"
echo Install: %REPO_ROOT%
echo Install: %REPO_ROOT%>> "%LOG%"
echo Mode: %MODE%
echo Mode: %MODE%>> "%LOG%"

if "%PID%"=="" goto wait_done
echo Waiting for RIPPLE (PID %PID%) to exit...
echo Waiting for RIPPLE (PID %PID%) to exit...>> "%LOG%"
set WAITED=0

:wait_loop
tasklist /FI "PID eq %PID%" 2>nul | findstr /I /C:" %PID% " >nul 2>&1
if errorlevel 1 goto wait_done
timeout /t 1 /nobreak >nul
set /a WAITED+=1
if !WAITED! GEQ %RIPPLE_UPDATE_WAIT_SECONDS% (
    echo Timed out waiting for RIPPLE to exit.
    echo Timed out waiting for RIPPLE to exit.>> "%LOG%"
    exit /b 1
)
goto wait_loop

:wait_done
echo RIPPLE exited.
echo RIPPLE exited.>> "%LOG%"

where git >nul 2>&1
if errorlevel 1 (
    echo git was not found on PATH.
    echo git was not found on PATH.>> "%LOG%"
    exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo This folder is not a git checkout. Re-clone from https://github.com/Le0nZim/ripple.git
    echo This folder is not a git checkout.>> "%LOG%"
    exit /b 1
)

git status --porcelain --untracked-files=no > "%TEMP%\ripple-git-status.txt" 2>> "%LOG%"
for %%A in ("%TEMP%\ripple-git-status.txt") do if %%~zA GTR 0 (
    echo Working tree has local source changes; aborting update.
    echo Working tree has local source changes; aborting update.>> "%LOG%"
    type "%TEMP%\ripple-git-status.txt"
    type "%TEMP%\ripple-git-status.txt">> "%LOG%"
    exit /b 1
)

echo Fetching %RIPPLE_UPDATE_REMOTE% %RIPPLE_UPDATE_BRANCH%...
echo Fetching %RIPPLE_UPDATE_REMOTE% %RIPPLE_UPDATE_BRANCH%...>> "%LOG%"
git fetch "%RIPPLE_UPDATE_REMOTE%" "%RIPPLE_UPDATE_BRANCH%"
if errorlevel 1 (
    echo git fetch failed.
    echo git fetch failed.>> "%LOG%"
    exit /b 1
)

echo Fast-forwarding...
echo Fast-forwarding...>> "%LOG%"
git merge --ff-only FETCH_HEAD
if errorlevel 1 (
    echo Fast-forward merge failed.
    echo Fast-forward merge failed.>> "%LOG%"
    exit /b 1
)

if not "%RIPPLE_UPDATE_SKIP_QUICKSTART%"=="1" (
    if /i "%MODE%"=="gpu" (
        call "%REPO_ROOT%\quickstart.bat" --yes --gpu --no-launch
    ) else (
        call "%REPO_ROOT%\quickstart.bat" --yes --cpu --no-launch
    )
    if errorlevel 1 (
        echo Rebuild failed.
        echo Rebuild failed.>> "%LOG%"
        exit /b 1
    )
) else (
    echo Skipping quickstart (RIPPLE_UPDATE_SKIP_QUICKSTART=1).
    echo Skipping quickstart (RIPPLE_UPDATE_SKIP_QUICKSTART=1).>> "%LOG%"
)

if "%RIPPLE_UPDATE_SKIP_RELAUNCH%"=="1" (
    echo Skipping relaunch (RIPPLE_UPDATE_SKIP_RELAUNCH=1).
    echo Skipping relaunch (RIPPLE_UPDATE_SKIP_RELAUNCH=1).>> "%LOG%"
    exit /b 0
)

if exist "%REPO_ROOT%\RIPPLE.bat" (
    echo Relaunching RIPPLE...
    echo Relaunching RIPPLE...>> "%LOG%"
    call "%REPO_ROOT%\RIPPLE.bat"
    exit /b %ERRORLEVEL%
)

if exist "%REPO_ROOT%\target\ripple.jar" (
    echo RIPPLE.bat missing; launching JAR directly.
    echo RIPPLE.bat missing; launching JAR directly.>> "%LOG%"
    java -jar "%REPO_ROOT%\target\ripple.jar"
    exit /b %ERRORLEVEL%
)

echo Update finished but RIPPLE could not be relaunched. Start it with RIPPLE.bat
echo Update finished but RIPPLE could not be relaunched.>> "%LOG%"
exit /b 1
