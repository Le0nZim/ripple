@echo off
REM =============================================================================
REM Persistent RAFT Tracking Service Management Interface for Windows
REM =============================================================================
REM 
REM Windows equivalent of run_persistent_tracking.sh
REM Uses TCP socket instead of Unix socket for Windows compatibility
REM
REM Supported operations:
REM   start              - Initialize persistent tracking server process
REM   stop               - Terminate server with graceful shutdown sequence
REM   status             - Query server operational state
REM   restart            - Perform stop-start cycle with cleanup
REM   compute_flow       - Compute optical flow for video
REM   track_seed         - Track a single seed point
REM   track_anchors      - Track with multiple anchor points
REM   visualize_flow     - Generate flow visualization
REM   load_flow          - Load cached flow into memory
REM   compress_video     - Compress video for faster processing
REM   clear_cache        - Clear server's flow cache
REM   clear_memory       - Clear server's memory
REM   preview_dog_detection     - Preview DoG detection
REM   preview_trackpy_trajectories - Preview TrackPy trajectories
REM   optimize_tracks    - Optimize track positions
REM   ping               - Test server connection
REM =============================================================================

setlocal EnableDelayedExpansion

REM Get script directory for relative paths
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

REM Configuration with defaults (can be overridden by environment variables)
if not defined SOCKET_HOST set "SOCKET_HOST=127.0.0.1"
if not defined SOCKET_PORT set "SOCKET_PORT=9876"
if not defined SERVER_SCRIPT set "SERVER_SCRIPT=%PROJECT_DIR%\src\main\python\tracking_server.py"
if not defined SEND_CMD_SCRIPT set "SEND_CMD_SCRIPT=%PROJECT_DIR%\src\main\python\send_command.py"
if not defined CONDA_ENV set "CONDA_ENV=ripple-env"
if not defined MODEL_SIZE set "MODEL_SIZE=large"
if not defined LOG_FILE set "LOG_FILE=%TEMP%\ripple-env.log"

REM Find conda Python executable
set "CONDA_PYTHON="
call :find_conda_python
if not defined CONDA_PYTHON (
    echo [ERROR] Could not find Python in conda environment: %CONDA_ENV%
    echo [ERROR] Make sure the environment exists and has Python installed.
    exit /b 1
)

REM Check arguments
if "%~1"=="" (
    echo Usage: %~nx0 ^<command^> [args...]
    echo Commands: start, stop, status, restart, compute_flow, track_seed, etc.
    exit /b 1
)

set "COMMAND=%~1"

REM Route to appropriate function based on command
if /i "%COMMAND%"=="start" goto :start_server
if /i "%COMMAND%"=="stop" goto :stop_server
if /i "%COMMAND%"=="status" goto :check_status
if /i "%COMMAND%"=="status_json" goto :status_json
if /i "%COMMAND%"=="restart" goto :restart_server

REM For all other commands, forward to the Python send_command.py script
REM This handles: compute_flow, track_seed, track_anchors, visualize_flow, 
REM               load_flow, compress_video, clear_cache, clear_memory,
REM               preview_dog_detection, preview_trackpy_trajectories,
REM               optimize_tracks, ping, etc.

REM First ensure server is running
call :is_server_running
if %errorlevel% neq 0 (
    echo [INFO] Server not running, starting it now...
    call :start_server
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to start server
        exit /b 1
    )
)

REM Build full argument list for send_command.py
REM Pass all arguments including the command name
set "ALL_ARGS="
for %%A in (%*) do set "ALL_ARGS=!ALL_ARGS! %%A"

REM Set environment variables for the Python script
set "SOCKET_HOST=%SOCKET_HOST%"
set "SOCKET_PORT=%SOCKET_PORT%"

REM Run the command via send_command.py
"%CONDA_PYTHON%" "%SEND_CMD_SCRIPT%" %ALL_ARGS%
exit /b %errorlevel%

REM =============================================================================
REM FIND CONDA PYTHON
REM =============================================================================
:find_conda_python
REM Search for Python in the conda environment

REM Check common conda installation locations
if exist "%USERPROFILE%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%USERPROFILE%\miniconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)
if exist "%USERPROFILE%\Miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%USERPROFILE%\Miniconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)
if exist "%USERPROFILE%\anaconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%USERPROFILE%\anaconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)
if exist "%USERPROFILE%\Anaconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%USERPROFILE%\Anaconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)
if exist "%LOCALAPPDATA%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%LOCALAPPDATA%\miniconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)
if exist "%PROGRAMDATA%\miniconda3\envs\%CONDA_ENV%\python.exe" (
    set "CONDA_PYTHON=%PROGRAMDATA%\miniconda3\envs\%CONDA_ENV%\python.exe"
    goto :eof
)

REM Try using CONDA_EXE to find conda location
if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do set "CONDA_BASE=%%~dpI.."
    if exist "!CONDA_BASE!\envs\%CONDA_ENV%\python.exe" (
        set "CONDA_PYTHON=!CONDA_BASE!\envs\%CONDA_ENV%\python.exe"
        goto :eof
    )
)

goto :eof

REM =============================================================================
REM SERVER LIFECYCLE
REM =============================================================================
:start_server
call :is_server_running
if %errorlevel%==0 (
    echo [INFO] Server is already running
    exit /b 0
)

REM First ensure any old server processes are killed
call :kill_server_processes

echo [INFO] Starting persistent tracking server...
echo [INFO] Using Python: %CONDA_PYTHON%
echo [INFO] Script: %SERVER_SCRIPT%

REM Ensure log file directory exists
for %%F in ("%LOG_FILE%") do if not exist "%%~dpF" mkdir "%%~dpF" 2>nul

REM Try to clear old log - if it fails, use a timestamped log file
set "ACTUAL_LOG_FILE=%LOG_FILE%"
if exist "%LOG_FILE%" (
    del "%LOG_FILE%" 2>nul
    if exist "%LOG_FILE%" (
        REM Log file is locked, use a timestamped version
        for /f "tokens=2 delims==" %%T in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set "TIMESTAMP=%%T"
        set "ACTUAL_LOG_FILE=%LOG_FILE%.!TIMESTAMP:~0,14!"
        echo [WARN] Log file locked, using: !ACTUAL_LOG_FILE!
    )
)

REM Start server in background - use PowerShell Start-Process to fully detach
REM This creates a truly independent process that won't be killed when parent exits
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%CONDA_PYTHON%' -ArgumentList '\"%SERVER_SCRIPT%\" --tcp-host %SOCKET_HOST% --tcp-port %SOCKET_PORT% --model %MODEL_SIZE% --device auto' -RedirectStandardOutput '!ACTUAL_LOG_FILE!' -RedirectStandardError '!ACTUAL_LOG_FILE!.err' -WindowStyle Hidden"

REM Give it time to start (use ping instead of timeout to avoid stdin issues)
echo [INFO] Waiting for server to start...
set /a WAIT_COUNT=0
:wait_loop
if %WAIT_COUNT% geq 15 goto :wait_done
ping -n 2 127.0.0.1 >nul
call :is_server_running
if %errorlevel%==0 goto :server_started
set /a WAIT_COUNT+=1
goto :wait_loop

:server_started
echo [SUCCESS] Server started
echo   Host: %SOCKET_HOST%:%SOCKET_PORT%
exit /b 0

:wait_done
echo [ERROR] Server failed to start within timeout
echo   Check log file: %LOG_FILE%
echo   Check if Python and dependencies are installed in %CONDA_ENV%
type "%LOG_FILE%" 2>nul
exit /b 1

:stop_server
echo [INFO] Stopping server...

REM Try graceful shutdown by sending stop command
call :send_stop_command 2>nul

REM Give it a moment (use ping instead of timeout)
ping -n 3 127.0.0.1 >nul

REM Kill any remaining python processes running tracking_server
call :kill_server_processes

echo [SUCCESS] Server stopped
exit /b 0

:kill_server_processes
REM Kill all python processes running tracking_server
for /f "tokens=2 delims=," %%p in ('tasklist /fi "imagename eq python.exe" /fo csv /nh 2^>nul') do (
    set "PID=%%~p"
    wmic process where "ProcessId=!PID!" get CommandLine 2>nul | findstr /i "tracking_server" >nul
    if not errorlevel 1 (
        echo [INFO] Killing process !PID!
        taskkill /pid !PID! /f >nul 2>&1
    )
)
REM Wait briefly for processes to terminate and release file handles
ping -n 2 127.0.0.1 >nul
goto :eof

:restart_server
call :stop_server
ping -n 2 127.0.0.1 >nul
call :start_server
exit /b %errorlevel%

:check_status
call :is_server_running
if %errorlevel%==0 (
    echo [SUCCESS] Server is RUNNING
    echo   Host: %SOCKET_HOST%:%SOCKET_PORT%
    
    REM Try to ping the server
    "%CONDA_PYTHON%" "%SEND_CMD_SCRIPT%" ping 2>nul
    if !errorlevel!==0 (
        echo   Status: Responding to ping
    ) else (
        echo   Status: Not responding to ping
    )
    exit /b 0
) else (
    echo [WARNING] Server is NOT RUNNING
    echo   Use '%~nx0 start' to start the server
    exit /b 1
)

:status_json
call :is_server_running
if %errorlevel%==0 (
    REM Output ONLY JSON for programmatic clients
    "%CONDA_PYTHON%" "%SEND_CMD_SCRIPT%" status
    exit /b %errorlevel%
) else (
    echo {"status":"error","message":"Server not running","busy":false}
    exit /b 1
)

:is_server_running
REM Check if server is responding on TCP port using PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('%SOCKET_HOST%', %SOCKET_PORT%); $c.Close(); exit 0 } catch { exit 1 }"
exit /b %errorlevel%

:send_stop_command
REM Send stop command to server using PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('%SOCKET_HOST%', %SOCKET_PORT%); $s = $c.GetStream(); $w = New-Object IO.StreamWriter($s); $w.WriteLine('{\"command\":\"stop\"}'); $w.Flush(); $c.Close() } catch {}"
exit /b 0

endlocal
