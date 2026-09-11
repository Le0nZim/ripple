@echo off
REM =============================================================================
REM RIPPLE Quick Start Script for Windows
REM =============================================================================
REM One-command setup and launch for biology labs.
REM Missing JDK 17+, Maven, and Miniconda are downloaded after confirmation.
REM =============================================================================

setlocal EnableDelayedExpansion

set "ASSUME_YES=%RIPPLE_ASSUME_YES%"
if "%ASSUME_YES%"=="" set "ASSUME_YES=0"
set "FORCE_CPU=0"
set "FORCE_GPU=0"
set "CHECK_ONLY=0"
set "NO_LAUNCH=%RIPPLE_NO_LAUNCH%"
if "%NO_LAUNCH%"=="" set "NO_LAUNCH=0"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--yes" set "ASSUME_YES=1"
if /i "%~1"=="-y" set "ASSUME_YES=1"
if /i "%~1"=="--cpu" set "FORCE_CPU=1"
if /i "%~1"=="--gpu" set "FORCE_GPU=1"
if /i "%~1"=="--check" set "CHECK_ONLY=1"
if /i "%~1"=="--no-launch" set "NO_LAUNCH=1"
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
shift
goto parse_args

:show_help
echo Usage: quickstart.bat [options]
echo.
echo One-command RIPPLE setup for Windows. Missing JDK 17+, Maven, and
echo Miniconda are downloaded into tools\ or %%USERPROFILE%%\miniconda3.
echo.
echo   --yes, -y       Install missing tools without prompting
echo   --cpu           Force CPU mode
echo   --gpu           Prefer GPU mode when an NVIDIA GPU is available
echo   --check         Doctor mode: report tools, do not install or launch
echo   --no-launch     Set up but do not start RIPPLE
echo   --help, -h      Show this help
echo.
exit /b 0

:args_done

echo.
echo +==========================================================+
echo ^|                     RIPPLE                               ^|
echo ^|        Video Annotation Tool for Biology                 ^|
echo +==========================================================+
echo.

cd /d "%~dp0"
set "PROJECT_DIR=%cd%"
set "CONDA_ENV_NAME=ripple-env"
set "PS_BOOTSTRAP=%PROJECT_DIR%\scripts\lib\bootstrap_tools.ps1"
set "TOOLCHAIN_CMD=%PROJECT_DIR%\tools\ripple-toolchain.cmd"
set "PREFLIGHT_ENV=%PROJECT_DIR%\tools\ripple-preflight.env"

echo [1/6] Checking system requirements...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action Preflight -ProjectDir "%PROJECT_DIR%"
set "PREFLIGHT_RC=%ERRORLEVEL%"

set "NEED_JDK=1"
set "NEED_MAVEN=1"
set "NEED_CONDA=1"
if exist "%PREFLIGHT_ENV%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%PREFLIGHT_ENV%") do (
        if /i "%%A"=="NEED_JDK" set "NEED_JDK=%%B"
        if /i "%%A"=="NEED_MAVEN" set "NEED_MAVEN=%%B"
        if /i "%%A"=="NEED_CONDA" set "NEED_CONDA=%%B"
    )
)

if "%CHECK_ONLY%"=="1" (
    if "%PREFLIGHT_RC%"=="0" (
        powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action ValidateJava -ProjectDir "%PROJECT_DIR%"
        echo   [OK] Doctor check passed
        exit /b 0
    )
    echo   [!] Doctor check found missing tools. Re-run without --check to install.
    pause
    exit /b 1
)

if not "%NEED_JDK%"=="0" goto need_install
if not "%NEED_MAVEN%"=="0" goto need_install
if not "%NEED_CONDA%"=="0" goto need_install
goto apply_toolchain

:need_install
echo.
if not "%ASSUME_YES%"=="1" (
    set /p "INSTALL_MISSING=  Install missing tools now? [Y/n]: "
    if /i "!INSTALL_MISSING!"=="n" (
        echo.
        echo   Portable install skipped. You can also install tools yourself:
        echo     JDK 17+ ^(full JDK with javac^): https://adoptium.net/
        echo     Maven 3.8+: https://maven.apache.org/download.cgi
        echo     Miniconda: https://docs.conda.io/en/latest/miniconda.html
        echo.
        echo   Then re-run: quickstart.bat
        pause
        exit /b 1
    )
) else (
    echo   Installing missing tools (--yes)...
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action Install -ProjectDir "%PROJECT_DIR%"
if errorlevel 1 (
    echo   [ERROR] Portable tool install failed.
    echo   Check your network, firewall, or proxy, then re-run:
    echo     quickstart.bat --yes
    pause
    exit /b 1
)

:apply_toolchain
if exist "%TOOLCHAIN_CMD%" call "%TOOLCHAIN_CMD%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action ValidateJava -ProjectDir "%PROJECT_DIR%"
if errorlevel 1 (
    echo   [ERROR] Compatible JDK 17+ not found after setup
    echo   Re-run: quickstart.bat --yes
    echo   Or install a full JDK 17+ from https://adoptium.net/
    pause
    exit /b 1
)
if exist "%TOOLCHAIN_CMD%" call "%TOOLCHAIN_CMD%"

REM =============================================================================
REM Initialize Conda for batch scripts
REM =============================================================================
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3"
) else if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Miniconda3\Scripts\activate.bat" "%USERPROFILE%\Miniconda3"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
) else if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Anaconda3\Scripts\activate.bat" "%USERPROFILE%\Anaconda3"
) else if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" "%LOCALAPPDATA%\miniconda3"
) else if exist "%PROGRAMDATA%\miniconda3\Scripts\activate.bat" (
    call "%PROGRAMDATA%\miniconda3\Scripts\activate.bat" "%PROGRAMDATA%\miniconda3"
) else if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do set "CONDA_DIR=%%~dpI.."
    if exist "!CONDA_DIR!\Scripts\activate.bat" (
        call "!CONDA_DIR!\Scripts\activate.bat" "!CONDA_DIR!"
    )
)

where conda >nul 2>&1
if errorlevel 1 (
    call conda --version >nul 2>&1
    if errorlevel 1 (
        echo   [ERROR] Conda not found after setup
        echo   Re-run: quickstart.bat --yes
        echo   Or install Miniconda from https://docs.conda.io/en/latest/miniconda.html
        pause
        exit /b 1
    )
)
echo   [OK] Conda found

where mvn >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Maven not found after setup
    echo   Re-run: quickstart.bat --yes
    pause
    exit /b 1
)
echo   [OK] Maven found

REM =============================================================================
REM STEP 2: Detect GPU
REM =============================================================================
echo.
echo [2/6] Detecting GPU...

set GPU_AVAILABLE=false
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo   [OK] NVIDIA GPU detected
    set GPU_AVAILABLE=true
) else (
    echo   [INFO] No NVIDIA GPU detected
)

REM =============================================================================
REM STEP 3: User Selection
REM =============================================================================
echo.
echo [3/6] Installation mode selection...

if "%FORCE_CPU%"=="1" (
    set GPU_MODE=cpu
    echo   [OK] CPU mode selected (--cpu)
    goto conda_setup
)
if "%FORCE_GPU%"=="1" (
    if "%GPU_AVAILABLE%"=="true" (
        set GPU_MODE=gpu
        echo   [OK] GPU mode selected (--gpu)
    ) else (
        set GPU_MODE=cpu
        echo   [!] --gpu requested but no NVIDIA GPU found; using CPU mode
    )
    goto conda_setup
)

if "%GPU_AVAILABLE%"=="true" (
    echo.
    echo   Please select installation mode:
    echo.
    echo     [1] GPU mode ^(recommended^)
    echo         - Full functionality: RAFT, LocoTrack, TrackPy, DIS
    echo         - Requires NVIDIA GPU with CUDA support
    echo.
    echo     [2] CPU mode
    echo         - Limited functionality: TrackPy, DIS optical flow
    echo         - Works on any system
    echo.

    :choice_loop
    set /p "choice=  Enter your choice [1/2]: "
    if "!choice!"=="1" (
        set GPU_MODE=gpu
        echo.
        echo   [OK] GPU mode selected
    ) else if "!choice!"=="2" (
        set GPU_MODE=cpu
        echo.
        echo   [OK] CPU mode selected
    ) else (
        echo   [ERROR] Invalid choice. Please enter 1 or 2.
        goto choice_loop
    )
) else (
    set GPU_MODE=cpu
    echo   CPU mode will be used ^(no NVIDIA GPU available^)
    echo   Note: RAFT and LocoTrack require an NVIDIA GPU
)

:conda_setup
REM =============================================================================
REM STEP 4: Setup Conda Environment
REM =============================================================================
echo.
echo [4/6] Setting up conda environment...

call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1

call conda env list | findstr /C:"%CONDA_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo   Creating new environment '%CONDA_ENV_NAME%'...
    if "%GPU_MODE%"=="gpu" (
        if exist "conda\environment.yml" (
            call conda env create -f conda\environment.yml -n %CONDA_ENV_NAME% -y
        ) else (
            call conda create -n %CONDA_ENV_NAME% python=3.11 pip -y
        )
    ) else (
        if exist "conda\environment-cpu.yml" (
            call conda env create -f conda\environment-cpu.yml -n %CONDA_ENV_NAME% -y
        ) else (
            call conda create -n %CONDA_ENV_NAME% python=3.11 pip -y
        )
    )
    if errorlevel 1 (
        echo   [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
    echo   [OK] Environment created
) else (
    echo   [OK] Environment '%CONDA_ENV_NAME%' already exists
)

echo   Activating environment '%CONDA_ENV_NAME%'...
call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo   [ERROR] Failed to activate conda environment.
    echo   Try running this script from Anaconda Prompt instead.
    pause
    exit /b 1
)

REM =============================================================================
REM STEP 5: Install Dependencies
REM =============================================================================
echo.
echo [5/6] Installing dependencies...

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action ValidateConda -ProjectDir "%PROJECT_DIR%" -GpuMode "%GPU_MODE%"
set "ENV_STATUS=%ERRORLEVEL%"
if "%ENV_STATUS%"=="0" (
    echo   [OK] Dependencies already installed
) else if "%ENV_STATUS%"=="2" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action RepairConda -ProjectDir "%PROJECT_DIR%" -GpuMode "%GPU_MODE%"
    if errorlevel 1 (
        echo   [ERROR] Failed to install Python dependencies.
        pause
        exit /b 1
    )
) else (
    echo   [ERROR] Conda environment validation failed
    pause
    exit /b 1
)

REM =============================================================================
REM STEP 6: Build and Launch
REM =============================================================================
echo.
echo [6/6] Building and launching RIPPLE...

if exist "%TOOLCHAIN_CMD%" call "%TOOLCHAIN_CMD%"

set "FORCE_REBUILD=%RIPPLE_FORCE_REBUILD%"
if "%FORCE_REBUILD%"=="" set "FORCE_REBUILD=0"

set "NEED_BUILD=0"
if "%FORCE_REBUILD%"=="1" set "NEED_BUILD=1"
if not exist "target\ripple.jar" set "NEED_BUILD=1"

if "%NEED_BUILD%"=="0" (
    for /f "usebackq delims=" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $paths=@('pom.xml','src\main\java','src\main\resources','src\main\python'); $mt=@(); foreach($p in $paths){ if(Test-Path $p){ $it=Get-Item $p; if($it.PSIsContainer){ $f=Get-ChildItem $p -Recurse -File | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1; if($f){ $mt += $f.LastWriteTimeUtc.ToFileTimeUtc() } } else { $mt += $it.LastWriteTimeUtc.ToFileTimeUtc() } } } $src=0; if($mt.Count -gt 0){ $src = ($mt | Measure-Object -Maximum).Maximum } $jar=0; if(Test-Path 'target\ripple.jar'){ $jar=(Get-Item 'target\ripple.jar').LastWriteTimeUtc.ToFileTimeUtc() } if($src -gt $jar){ '1' } else { '0' }"`) do set "NEED_BUILD=%%A"
)

if "%NEED_BUILD%"=="1" (
    echo   Building Java application...
    if "%FORCE_REBUILD%"=="1" (
        call mvn clean package -DskipTests -q
    ) else (
        call mvn package -DskipTests -q
    )
    if errorlevel 1 (
        echo   [ERROR] Maven build failed.
        echo   Check that Maven is properly installed and try again.
        pause
        exit /b 1
    )
    echo   [OK] Build complete
) else (
    echo   [OK] JAR already up to date
)

echo.
echo   Creating launch shortcuts...

if not defined RIPPLE_RESOLVED_JAVA_HOME if defined JAVA_HOME set "RIPPLE_RESOLVED_JAVA_HOME=%JAVA_HOME%"

(
echo @echo off
echo REM RIPPLE Launcher - Auto-generated by quickstart
echo setlocal EnableDelayedExpansion
echo cd /d "%%~dp0"
echo.
echo if exist "%%~dp0tools\jdk\bin\java.exe" ^(
echo     set "JAVA_HOME=%%~dp0tools\jdk"
echo     set "PATH=%%JAVA_HOME%%\bin;%%PATH%%"
echo ^) else if exist "%RIPPLE_RESOLVED_JAVA_HOME%\bin\java.exe" ^(
echo     set "JAVA_HOME=%RIPPLE_RESOLVED_JAVA_HOME%"
echo     set "PATH=%%JAVA_HOME%%\bin;%%PATH%%"
echo ^)
echo if exist "%%~dp0tools\maven\bin\mvn.cmd" set "PATH=%%~dp0tools\maven\bin;%%PATH%%"
echo.
echo REM Initialize Conda
echo if exist "%%USERPROFILE%%\miniconda3\Scripts\activate.bat" ^(
echo     call "%%USERPROFILE%%\miniconda3\Scripts\activate.bat" "%%USERPROFILE%%\miniconda3"
echo ^) else if exist "%%USERPROFILE%%\Miniconda3\Scripts\activate.bat" ^(
echo     call "%%USERPROFILE%%\Miniconda3\Scripts\activate.bat" "%%USERPROFILE%%\Miniconda3"
echo ^) else if exist "%%USERPROFILE%%\anaconda3\Scripts\activate.bat" ^(
echo     call "%%USERPROFILE%%\anaconda3\Scripts\activate.bat" "%%USERPROFILE%%\anaconda3"
echo ^) else if exist "%%USERPROFILE%%\Anaconda3\Scripts\activate.bat" ^(
echo     call "%%USERPROFILE%%\Anaconda3\Scripts\activate.bat" "%%USERPROFILE%%\Anaconda3"
echo ^) else if exist "%%LOCALAPPDATA%%\miniconda3\Scripts\activate.bat" ^(
echo     call "%%LOCALAPPDATA%%\miniconda3\Scripts\activate.bat" "%%LOCALAPPDATA%%\miniconda3"
echo ^) else if exist "%%PROGRAMDATA%%\miniconda3\Scripts\activate.bat" ^(
echo     call "%%PROGRAMDATA%%\miniconda3\Scripts\activate.bat" "%%PROGRAMDATA%%\miniconda3"
echo ^) else if defined CONDA_EXE ^(
echo     for %%%%I in ^("%%CONDA_EXE%%"^) do set "CONDA_DIR=%%%%~dpI.."
echo     if exist "!CONDA_DIR!\Scripts\activate.bat" call "!CONDA_DIR!\Scripts\activate.bat" "!CONDA_DIR!"
echo ^)
echo.
echo call conda activate %CONDA_ENV_NAME%
echo set RIPPLE_MODE=%GPU_MODE%
echo java -jar target\ripple.jar
echo endlocal
) > "%~dp0RIPPLE.bat"

echo   [OK] Created RIPPLE.bat

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_BOOTSTRAP%" -Action Shortcut -ProjectDir "%PROJECT_DIR%" -ShortcutTarget "%PROJECT_DIR%\RIPPLE.bat"

echo.
echo +==========================================================+
echo ^|                 RIPPLE Setup Complete!                   ^|
echo +==========================================================+
echo.
echo   Launch options:
echo     1. Double-click RIPPLE.bat in this folder
echo     2. Use the desktop shortcut
echo.
echo   Mode: %GPU_MODE%
echo.

if "%NO_LAUNCH%"=="1" (
    echo Setup finished without launching (--no-launch^).
    echo.
    exit /b 0
)

echo RIPPLE is ready! Launching now...
echo.

set RIPPLE_MODE=%GPU_MODE%
java -jar target\ripple.jar

if errorlevel 1 (
    echo.
    echo [ERROR] RIPPLE exited with an error.
)

echo.
echo Press any key to close this window...
pause >nul

endlocal
