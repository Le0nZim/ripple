<# 
.SYNOPSIS
    RIPPLE Build Script for Windows

.DESCRIPTION
    Creates a distributable package for Windows including:
    - Java application (JAR or native installer)
    - Bundled Python environment
    - Pre-downloaded models
    - MSI or EXE installer

.PARAMETER PackageType
    Type of package to create: msi, exe, or zip

.EXAMPLE
    .\build_windows.ps1 -PackageType zip
#>

param(
    [ValidateSet("msi", "exe", "zip")]
    [string]$PackageType = "zip"
)

# Configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $ProjectDir "dist\windows"
$AppName = "RIPPLE"
$AppVersion = "1.0.0"
$MainClass = "com.ripple.VideoAnnotationTool"
$MainJar = "ripple.jar"

# Python embed URL (Windows embeddable package)
$PythonVersion = "3.10.11"
$PythonEmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

# =============================================================================
# Check Prerequisites
# =============================================================================
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Java
    try {
        $javaVersion = & java -version 2>&1 | Select-String "version" | ForEach-Object { $_ -match '"(\d+)' | Out-Null; $matches[1] }
        if ([int]$javaVersion -lt 17) {
            Write-Error "Java 17+ required. Found version $javaVersion."
            exit 1
        }
        Write-Info "Java version: $javaVersion"
    } catch {
        Write-Error "Java not found. Please install JDK 17+."
        exit 1
    }
    
    # Check Maven
    try {
        $mvnVersion = & mvn --version 2>&1 | Select-String "Apache Maven"
        Write-Info "Maven: $mvnVersion"
    } catch {
        Write-Error "Maven not found. Please install Maven 3.8+."
        exit 1
    }
}

# =============================================================================
# Build Java Application
# =============================================================================
function Build-JavaApp {
    Write-Info "Building Java application..."
    
    Push-Location $ProjectDir
    try {
        if (Test-Path "pom.xml") {
            & mvn clean package -DskipTests -q
            
            $jarPath = Get-ChildItem -Path "target" -Filter "*.jar" -Exclude "*-sources.jar","*-javadoc.jar" | Select-Object -First 1
            if ($jarPath) {
                Write-Info "JAR built: $($jarPath.FullName)"
                return $jarPath.FullName
            }
        }
        Write-Warning "pom.xml not found or build failed"
        return $null
    } finally {
        Pop-Location
    }
}

# =============================================================================
# Create Python Bundle (Embedded Python)
# =============================================================================
function New-PythonBundle {
    Write-Info "Creating Python bundle..."
    
    $PythonDir = Join-Path $BuildDir "app\python"
    $TempDir = Join-Path $BuildDir "temp"
    
    New-Item -ItemType Directory -Path $PythonDir -Force | Out-Null
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    # Download Python embeddable package
    $zipPath = Join-Path $TempDir "python-embed.zip"
    
    Write-Info "Downloading Python $PythonVersion embeddable..."
    Invoke-WebRequest -Uri $PythonEmbedUrl -OutFile $zipPath
    
    # Extract
    Expand-Archive -Path $zipPath -DestinationPath $PythonDir -Force
    
    # Download get-pip.py
    $getPipPath = Join-Path $TempDir "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
    
    # Enable pip in embedded Python
    # Modify python310._pth to allow imports
    $pthFile = Get-ChildItem -Path $PythonDir -Filter "python*._pth" | Select-Object -First 1
    if ($pthFile) {
        $content = Get-Content $pthFile.FullName
        $content = $content -replace "#import site", "import site"
        $content += "`nLib\site-packages"
        Set-Content -Path $pthFile.FullName -Value $content
    }
    
    # Install pip
    $pythonExe = Join-Path $PythonDir "python.exe"
    & $pythonExe $getPipPath --no-warn-script-location
    
    # Install packages (CPU version for Windows compatibility)
    Write-Info "Installing Python packages..."
    $requirementsFile = Join-Path $ProjectDir "requirements\requirements-cpu.txt"
    if (Test-Path $requirementsFile) {
        & $pythonExe -m pip install -r $requirementsFile --no-warn-script-location -q
    } else {
        Write-Warning "Requirements file not found: $requirementsFile"
    }
    
    Write-Success "Python bundle created"
}

# =============================================================================
# Copy Runtime Files
# =============================================================================
function Copy-RuntimeFiles {
    param([string]$JarPath)
    
    Write-Info "Copying runtime files..."
    
    $AppDir = Join-Path $BuildDir "app"
    $RuntimeDir = Join-Path $AppDir "runtime"
    
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
    
    # Copy Python scripts
    $scripts = @(
        "tracking_server.py",
        "locotrack_flow.py",
        "trackmate_dog.py",
        "trackpy_flow.py",
        "runtime_config.py",
        "send_command.py"
    )
    
    foreach ($script in $scripts) {
        $srcPath = Join-Path $ProjectDir ("src\main\python\" + $script)
        if (Test-Path $srcPath) {
            Copy-Item $srcPath -Destination $RuntimeDir
        }
    }
    
    # Copy launcher
    Copy-Item (Join-Path $ProjectDir "src\main\python\ripple_launcher.py") -Destination $AppDir
    
    # Copy JAR if built
    if ($JarPath -and (Test-Path $JarPath)) {
        Copy-Item $JarPath -Destination (Join-Path $AppDir $MainJar)
    }
    
    # Create batch launcher
    $batchContent = @"
@echo off
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%python\python.exe" (
    "%SCRIPT_DIR%python\python.exe" "%SCRIPT_DIR%ripple_launcher.py" %*
) else (
    python "%SCRIPT_DIR%ripple_launcher.py" %*
)

endlocal
"@
    
    Set-Content -Path (Join-Path $AppDir "$AppName.bat") -Value $batchContent
    
    Write-Success "Runtime files copied"
}

# =============================================================================
# Create ZIP Distribution
# =============================================================================
function New-ZipDistribution {
    Write-Info "Creating ZIP distribution..."
    
    $AppDir = Join-Path $BuildDir "app"
    $ZipName = "$AppName-$AppVersion-windows-x64.zip"
    $ZipPath = Join-Path $BuildDir $ZipName
    
    Compress-Archive -Path "$AppDir\*" -DestinationPath $ZipPath -Force
    
    Write-Success "Created: $ZipPath"
}

# =============================================================================
# Create Inno Setup Installer
# =============================================================================
function New-InnoSetupInstaller {
    Write-Info "Creating Inno Setup installer..."
    
    # Check for Inno Setup
    $innoPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $innoPath)) {
        $innoPath = "C:\Program Files\Inno Setup 6\ISCC.exe"
    }
    
    if (-not (Test-Path $innoPath)) {
        Write-Warning "Inno Setup not found. Creating ZIP instead."
        New-ZipDistribution
        return
    }
    
    $AppDir = Join-Path $BuildDir "app"
    $IssPath = Join-Path $BuildDir "setup.iss"
    
    # Create Inno Setup script
    $issContent = @"
#define MyAppName "$AppName"
#define MyAppVersion "$AppVersion"
#define MyAppPublisher "Your Organization"
#define MyAppURL "https://your-website.com"
#define MyAppExeName "$AppName.bat"

[Setup]
AppId={{YOUR-GUID-HERE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=$BuildDir
OutputBaseFilename={#MyAppName}-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "$AppDir\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall skipifsilent
"@
    
    Set-Content -Path $IssPath -Value $issContent
    
    # Run Inno Setup Compiler
    & $innoPath $IssPath
    
    Write-Success "Created: $BuildDir\$AppName-$AppVersion-setup.exe"
}

# =============================================================================
# Main
# =============================================================================
function Main {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Blue
    Write-Host " $AppName Build Script for Windows"
    Write-Host "==========================================" -ForegroundColor Blue
    Write-Host ""
    
    Test-Prerequisites
    
    # Create build directory
    if (Test-Path $BuildDir) {
        Remove-Item -Path $BuildDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path (Join-Path $BuildDir "app") -Force | Out-Null
    
    $jarPath = Build-JavaApp
    New-PythonBundle
    Copy-RuntimeFiles -JarPath $jarPath
    
    switch ($PackageType) {
        "msi" {
            Write-Warning "MSI creation requires WiX Toolset. Creating EXE instead."
            New-InnoSetupInstaller
        }
        "exe" {
            New-InnoSetupInstaller
        }
        default {
            New-ZipDistribution
        }
    }
    
    Write-Host ""
    Write-Success "Build complete! Output in: $BuildDir"
    Write-Host ""
}

Main
