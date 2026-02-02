@echo off
REM =============================================================================
REM RIPPLE Quick Start Script for Windows
REM =============================================================================
REM One-command setup and launch for biology labs.
REM
REM This script:
REM   1. Checks system requirements (Java, Conda, Maven - must be pre-installed)
REM   2. Detects NVIDIA GPU availability
REM   3. Asks user to choose CPU or GPU version
REM   4. Creates/activates ripple-env conda environment
REM   5. Installs all dependencies
REM   6. Builds and launches RIPPLE
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo +==========================================================+
echo ^|                     RIPPLE                               ^|
echo ^|        Video Annotation Tool for Biology                 ^|
echo +==========================================================+
echo.

cd /d "%~dp0"

set CONDA_ENV_NAME=ripple-env

REM =============================================================================
REM STEP 0: Initialize Conda for batch scripts
REM =============================================================================
REM Find and run conda hook for proper activation support
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
    REM Use CONDA_EXE to find the conda installation
    for %%I in ("%CONDA_EXE%") do set "CONDA_DIR=%%~dpI.."
    if exist "!CONDA_DIR!\Scripts\activate.bat" (
        call "!CONDA_DIR!\Scripts\activate.bat" "!CONDA_DIR!"
    )
)

REM =============================================================================
REM STEP 1: Check Java
REM =============================================================================
echo [1/6] Checking system requirements...

java -version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Java not found!
    echo   Please install Java 11 or newer from https://adoptium.net/
    echo   Make sure Java is added to your PATH.
    pause
    exit /b 1
)
echo   [OK] Java installed

REM Check Conda
where conda >nul 2>&1
if errorlevel 1 (
    call conda --version >nul 2>&1
    if errorlevel 1 (
        echo   [ERROR] Conda not found!
        echo   Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
        echo   Make sure to check "Add to PATH" during installation or run from Anaconda Prompt.
        pause
        exit /b 1
    )
)
echo   [OK] Conda found

REM Check Maven
where mvn >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Maven not found!
    echo   Please install Maven 3.8+ from https://maven.apache.org/download.cgi
    echo   Make sure Maven is added to your PATH.
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

REM =============================================================================
REM STEP 4: Setup Conda Environment
REM =============================================================================
echo.
echo [4/6] Setting up conda environment...

REM Accept conda Terms of Service for default channels (required for non-interactive mode)
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1

REM Check if environment exists
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

REM Activate environment
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

REM Install dependencies if needed
python -c "import trackpy" >nul 2>&1
if errorlevel 1 (
    pip install --upgrade pip wheel setuptools -q
    if "%GPU_MODE%"=="gpu" (
        if exist "requirements\requirements-gpu.txt" (
            echo   Installing GPU packages ^(this may take a few minutes^)...
            pip install -r requirements\requirements-gpu.txt -q
        ) else (
            pip install -r requirements\requirements-cpu.txt -q
        )
    ) else (
        echo   Installing CPU packages...
        pip install -r requirements\requirements-cpu.txt -q
    )
)
echo   [OK] Dependencies installed

REM =============================================================================
REM STEP 6: Build and Launch
REM =============================================================================
echo.
echo [6/6] Building and launching RIPPLE...

REM Build Java application
REM Previous behavior only checked for existence of target\ripple.jar, which can be stale
REM if sources changed after the last build. We now compare timestamps.

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

REM =============================================================================
REM Create launch shortcuts
REM =============================================================================
echo.
echo   Creating launch shortcuts...

REM Create the launcher batch file
(
echo @echo off
echo REM RIPPLE Launcher - Auto-generated by quickstart
echo setlocal EnableDelayedExpansion
echo cd /d "%%~dp0"
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
