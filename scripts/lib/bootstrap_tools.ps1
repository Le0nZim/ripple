# Portable JDK / Maven / Miniconda bootstrap for RIPPLE (Windows).
# Requires Windows PowerShell 5.1+. Called from quickstart.bat.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Preflight", "Install", "ValidateJava", "ValidateConda", "RepairConda", "Shortcut", "Urls")]
    [string]$Action,

    [string]$ProjectDir = "",
    [string]$ToolsDir = "",
    [string]$GpuMode = "cpu",
    [string]$ShortcutTarget = "",
    [string]$Os = "",
    [string]$Arch = "",
    [int]$RequiredJavaMajor = 17,
    [string]$MavenVersion = "3.9.9"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$ProjectDir = [System.IO.Path]::GetFullPath($ProjectDir)

if ([string]::IsNullOrWhiteSpace($ToolsDir)) {
    if (-not [string]::IsNullOrWhiteSpace($env:RIPPLE_TOOLS_DIR)) {
        $ToolsDir = $env:RIPPLE_TOOLS_DIR
    } else {
        $ToolsDir = Join-Path $ProjectDir "tools"
    }
}

$MinicondaHome = if ($env:RIPPLE_MINICONDA_HOME) { $env:RIPPLE_MINICONDA_HOME } else { Join-Path $env:USERPROFILE "miniconda3" }
$StatusFile = Join-Path $ToolsDir "ripple-preflight.env"
$ToolchainFile = Join-Path $ToolsDir "ripple-toolchain.cmd"
$StampFile = Join-Path $ProjectDir ".ripple-env.stamp"

function Get-RippleAdoptiumOs {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { $Value = "windows" }
    switch -Regex ($Value) {
        "^(Darwin|darwin|mac|macos)$" { "mac"; break }
        "^(Linux|linux)$" { "linux"; break }
        default { "windows" }
    }
}

function Get-RippleAdoptiumArch {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { $Value = "arm64" } else { $Value = "x64" }
    }
    switch -Regex ($Value) {
        "^(arm64|aarch64)$" { "aarch64"; break }
        default { "x64" }
    }
}

function Get-RippleJdkUrl {
    param([string]$OsValue = "", [string]$ArchValue = "")
    $osName = Get-RippleAdoptiumOs $OsValue
    $archName = Get-RippleAdoptiumArch $ArchValue
    "https://api.adoptium.net/v3/binary/latest/$RequiredJavaMajor/ga/$osName/$archName/jdk/hotspot/normal/eclipse?project=jdk"
}

function Get-RippleMavenUrl {
    param([string]$OsValue = "")
    $osName = Get-RippleAdoptiumOs $OsValue
    $name = if ($osName -eq "windows") {
        "apache-maven-$MavenVersion-bin.zip"
    } else {
        "apache-maven-$MavenVersion-bin.tar.gz"
    }
    "https://archive.apache.org/dist/maven/maven-3/$MavenVersion/binaries/$name"
}

function Get-RippleMinicondaUrl {
    param([string]$OsValue = "", [string]$ArchValue = "")
    $osName = Get-RippleAdoptiumOs $OsValue
    $archName = Get-RippleAdoptiumArch $ArchValue
    if ($osName -eq "mac") {
        if ($archName -eq "aarch64") {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        } else {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        }
    } elseif ($osName -eq "linux") {
        if ($archName -eq "aarch64") {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        } else {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        }
    } else {
        if ($archName -eq "aarch64") {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-arm64.exe"
        } else {
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        }
    }
}

function Get-JavaMajor {
    param([string]$VersionText)
    if ([string]::IsNullOrWhiteSpace($VersionText)) { return 0 }
    $m = [regex]::Match($VersionText, '"([0-9]+)(?:\.[0-9]+)*"')
    if ($m.Success) { return [int]$m.Groups[1].Value }
    $m = [regex]::Match($VersionText, 'version\s+([0-9]+)')
    if ($m.Success) { return [int]$m.Groups[1].Value }
    $m = [regex]::Match($VersionText, 'javac\s+([0-9]+)')
    if ($m.Success) { return [int]$m.Groups[1].Value }
    return 0
}

function Test-RippleJdkHome {
    param([string]$Home)
    if ([string]::IsNullOrWhiteSpace($Home)) { return $false }
    $javaExe = Join-Path $Home "bin\java.exe"
    $javacExe = Join-Path $Home "bin\javac.exe"
    if (-not ((Test-Path $javaExe) -and (Test-Path $javacExe))) { return $false }
    try {
        $out = & $javaExe -version 2>&1 | Out-String
        return ((Get-JavaMajor $out) -ge $RequiredJavaMajor)
    } catch {
        return $false
    }
}

function Get-RuntimeJavaHome {
    param([string]$JavaExe)
    if (-not $JavaExe) { return $null }
    try {
        $out = & $JavaExe -XshowSettings:properties -version 2>&1 | Out-String
        $m = [regex]::Match($out, 'java\.home\s*=\s*(.+)')
        if ($m.Success) { return $m.Groups[1].Value.Trim() }
    } catch {
        return $null
    }
    return $null
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Resolve-RippleJdk {
    $java = Get-CommandPath "java.exe"
    if (-not $java) { $java = Get-CommandPath "java" }
    $runtimeHome = Get-RuntimeJavaHome $java
    if (Test-RippleJdkHome $runtimeHome) { return $runtimeHome }
    $javac = Get-CommandPath "javac.exe"
    if (-not $javac) { $javac = Get-CommandPath "javac" }
    if ($javac) {
        $home = Split-Path (Split-Path $javac -Parent) -Parent
        if (Test-RippleJdkHome $home) { return $home }
    }
    foreach ($candidate in @(
            (Join-Path $ToolsDir "jdk"),
            $env:JAVA_HOME
        )) {
        if (Test-RippleJdkHome $candidate) { return $candidate }
    }
    return $null
}

function Test-RippleMavenVersion {
    param([string]$Version)
    if ([string]::IsNullOrWhiteSpace($Version)) { return $false }
    $m = [regex]::Match($Version, '^(\d+)\.(\d+)')
    if (-not $m.Success) { return $false }
    $maj = [int]$m.Groups[1].Value
    $min = [int]$m.Groups[2].Value
    return ($maj -gt 3) -or (($maj -eq 3) -and ($min -ge 8))
}

function Get-RippleMavenVersionText {
    param([string]$MvnExe)
    try {
        $out = & $MvnExe -version 2>&1 | Out-String
        $m = [regex]::Match($out, 'Apache Maven\s+([0-9.]+)')
        if ($m.Success) { return $m.Groups[1].Value }
    } catch {
    }
    return ""
}

function Resolve-RippleMaven {
    $mvn = Get-CommandPath "mvn.cmd"
    if (-not $mvn) { $mvn = Get-CommandPath "mvn" }
    if ($mvn) {
        $ver = Get-RippleMavenVersionText $mvn
        if (Test-RippleMavenVersion $ver) {
            try {
                $out = & $mvn -version 2>&1 | Out-String
                $m = [regex]::Match($out, 'Maven home:\s*(.+)')
                if ($m.Success) { return $m.Groups[1].Value.Trim() }
            } catch {
            }
            return (Split-Path (Split-Path $mvn -Parent) -Parent)
        }
    }
    $portable = Join-Path $ToolsDir "maven\bin\mvn.cmd"
    if (Test-Path $portable) {
        $ver = Get-RippleMavenVersionText $portable
        if ((-not $ver) -or (Test-RippleMavenVersion $ver)) {
            return (Join-Path $ToolsDir "maven")
        }
    }
    return $null
}

function Resolve-RippleConda {
    $conda = Get-CommandPath "conda.exe"
    if (-not $conda) { $conda = Get-CommandPath "conda" }
    if ($conda) { return $conda }
    $candidates = @(
        (Join-Path $MinicondaHome "Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "Miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "Anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:PROGRAMDATA "miniconda3\Scripts\conda.exe")
    )
    if ($env:CONDA_EXE) { $candidates = @($env:CONDA_EXE) + $candidates }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Write-RippleStatus {
    param(
        [int]$NeedJdk,
        [int]$NeedMaven,
        [int]$NeedConda,
        [string]$JdkLocation,
        [string]$MavenLocation,
        [string]$CondaLocation
    )
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    @(
        "NEED_JDK=$NeedJdk",
        "NEED_MAVEN=$NeedMaven",
        "NEED_CONDA=$NeedConda",
        "JDK_LOCATION=$JdkLocation",
        "MAVEN_LOCATION=$MavenLocation",
        "CONDA_LOCATION=$CondaLocation"
    ) | Set-Content -Path $StatusFile -Encoding ASCII
}

function Write-RippleToolchain {
    param([string]$JdkHome, [string]$MavenHome)
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $lines = @(
        "@echo off",
        "set `"RIPPLE_RESOLVED_JAVA_HOME=$JdkHome`"",
        "set `"RIPPLE_RESOLVED_MAVEN_HOME=$MavenHome`""
    )
    if ($JdkHome) {
        $lines += @(
            "set `"JAVA_HOME=$JdkHome`"",
            "set `"PATH=%JAVA_HOME%\bin;%PATH%`""
        )
    }
    if ($MavenHome -and (Test-Path (Join-Path $MavenHome "bin\mvn.cmd"))) {
        $lines += "set `"PATH=$MavenHome\bin;%PATH%`""
    }
    $lines | Set-Content -Path $ToolchainFile -Encoding ASCII
}

function Download-RippleFile {
    param([string]$Url, [string]$Destination)
    $dir = Split-Path $Destination -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
            Unblock-File -Path $Destination -ErrorAction SilentlyContinue
            return $true
        } catch {
            Write-Host "  Download failed (attempt $attempt/2): $($_.Exception.Message)"
        }
    }
    Write-Host "  ERROR: download failed. Check your network, firewall, or proxy."
    Write-Host "         URL: $Url"
    return $false
}

function Install-RippleJdk {
    Write-Host "  Downloading Eclipse Temurin JDK $RequiredJavaMajor (portable, no admin)..."
    $archive = Join-Path $ToolsDir "temurin-jdk.zip"
    if (-not (Download-RippleFile (Get-RippleJdkUrl) $archive)) { return $false }
    $extract = Join-Path $ToolsDir "jdk-extract"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Expand-Archive -Path $archive -DestinationPath $extract -Force
    $javac = Get-ChildItem -Path $extract -Recurse -Filter javac.exe | Select-Object -First 1
    if (-not $javac) {
        Write-Host "  ERROR: extracted JDK is missing javac"
        return $false
    }
    $home = $javac.Directory.Parent.FullName
    $dest = Join-Path $ToolsDir "jdk"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item -Path (Join-Path $home "*") -Destination $dest -Recurse -Force
    Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
    Remove-Item -Force $archive -ErrorAction SilentlyContinue
    if (-not (Test-RippleJdkHome $dest)) {
        Write-Host "  ERROR: portable JDK did not validate after extract"
        return $false
    }
    Write-Host "  Installed portable JDK to $dest"
    return $true
}

function Install-RippleMaven {
    Write-Host "  Downloading Apache Maven $MavenVersion (portable, no admin)..."
    $archive = Join-Path $ToolsDir "apache-maven-$MavenVersion-bin.zip"
    if (-not (Download-RippleFile (Get-RippleMavenUrl) $archive)) { return $false }
    $extract = Join-Path $ToolsDir "maven-extract"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Expand-Archive -Path $archive -DestinationPath $extract -Force
    $inner = Get-ChildItem -Path $extract -Directory | Where-Object { $_.Name -like "apache-maven-*" } | Select-Object -First 1
    if (-not $inner) {
        Write-Host "  ERROR: unexpected Maven archive layout"
        return $false
    }
    $dest = Join-Path $ToolsDir "maven"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Move-Item $inner.FullName $dest
    Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
    Remove-Item -Force $archive -ErrorAction SilentlyContinue
    if (-not (Test-Path (Join-Path $dest "bin\mvn.cmd"))) {
        Write-Host "  ERROR: portable Maven did not validate after extract"
        return $false
    }
    Write-Host "  Installed portable Maven to $dest"
    return $true
}

function Install-RippleMiniconda {
    Write-Host "  Downloading Miniconda (user-local, no admin)..."
    $installer = Join-Path $env:TEMP "Miniconda3-latest-Windows.exe"
    if (-not (Download-RippleFile (Get-RippleMinicondaUrl) $installer)) { return $false }
    $args = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$MinicondaHome"
    $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    Remove-Item -Force $installer -ErrorAction SilentlyContinue
    $condaExe = Join-Path $MinicondaHome "Scripts\conda.exe"
    if ($proc.ExitCode -ne 0 -or -not (Test-Path $condaExe)) {
        Write-Host "  ERROR: Miniconda install failed (exit $($proc.ExitCode))"
        return $false
    }
    Write-Host "  Installed Miniconda to $MinicondaHome"
    return $true
}

function Get-RippleReqFile {
    if ($GpuMode -eq "gpu") {
        return (Join-Path $ProjectDir "requirements\requirements-gpu.txt")
    }
    return (Join-Path $ProjectDir "requirements\requirements-cpu.txt")
}

function Get-RippleReqHash {
    $req = Get-RippleReqFile
    if (-not (Test-Path $req)) { return "" }
    return (Get-FileHash -Algorithm SHA256 -Path $req).Hash.ToLower()
}

function Get-RippleStampHash {
    if (-not (Test-Path $StampFile)) { return "" }
    $line = Select-String -Path $StampFile -Pattern "^requirements_hash=" | Select-Object -Last 1
    if ($line) { return ($line.Line -split "=", 2)[1] }
    return ""
}

function Write-RippleCondaStamp {
    $hash = Get-RippleReqHash
    if (-not $hash) { return }
    $utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @(
        "requirements_hash=$hash",
        "gpu_mode=$GpuMode",
        "generated_utc=$utc"
    ) | Set-Content -Path $StampFile -Encoding ASCII
}

function Get-RippleMissingModules {
    $modules = @("numpy", "scipy", "pandas", "tifffile", "cv2", "trackpy", "torch")
    if ($GpuMode -eq "gpu") { $modules += "torchvision" }
    $missing = @()
    foreach ($module in $modules) {
        & python -c "import $module" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { $missing += $module }
    }
    return $missing
}

switch ($Action) {
    "Urls" {
        Write-Output "jdk=$(Get-RippleJdkUrl $Os $Arch)"
        Write-Output "maven=$(Get-RippleMavenUrl $Os)"
        Write-Output "miniconda=$(Get-RippleMinicondaUrl $Os $Arch)"
    }
    "Preflight" {
        $jdk = Resolve-RippleJdk
        $maven = Resolve-RippleMaven
        $conda = Resolve-RippleConda
        $needJdk = if ($jdk) { 0 } else { 1 }
        $needMaven = if ($maven) { 0 } else { 1 }
        $needConda = if ($conda) { 0 } else { 1 }
        Write-RippleStatus $needJdk $needMaven $needConda $jdk $maven $conda
        Write-RippleToolchain $jdk $maven
        if ($jdk) { Write-Host "    [OK] JDK 17+: $jdk" } else { Write-Host "    [!] JDK 17+ not found (will download Eclipse Temurin into tools\jdk)" }
        if ($maven) { Write-Host "    [OK] Maven: $maven" } else { Write-Host "    [!] Maven 3.8+ not found (will download Apache Maven into tools\maven)" }
        if ($conda) { Write-Host "    [OK] Conda: $conda" } else { Write-Host "    [!] Conda not found (will install Miniconda to $MinicondaHome)" }
        if (($needJdk + $needMaven + $needConda) -gt 0) { exit 2 } else { exit 0 }
    }
    "Install" {
        $jdk = Resolve-RippleJdk
        $maven = Resolve-RippleMaven
        $conda = Resolve-RippleConda
        if (-not $jdk) {
            if (-not (Install-RippleJdk)) { exit 1 }
        }
        if (-not $maven) {
            if (-not (Install-RippleMaven)) { exit 1 }
        }
        if (-not $conda) {
            if (-not (Install-RippleMiniconda)) { exit 1 }
        }
        $jdk = Resolve-RippleJdk
        $maven = Resolve-RippleMaven
        $conda = Resolve-RippleConda
        Write-RippleStatus $(if ($jdk) { 0 } else { 1 }) $(if ($maven) { 0 } else { 1 }) $(if ($conda) { 0 } else { 1 }) $jdk $maven $conda
        Write-RippleToolchain $jdk $maven
        if (-not $jdk -or -not $maven -or -not $conda) { exit 1 }
        exit 0
    }
    "ValidateJava" {
        $jdk = Resolve-RippleJdk
        if (-not $jdk) {
            Write-Host "  [ERROR] JDK $RequiredJavaMajor+ with javac not found."
            exit 1
        }
        $env:JAVA_HOME = $jdk
        $env:PATH = "$(Join-Path $jdk 'bin');$env:PATH"
        Write-Host "  Java runtime:"
        & (Join-Path $jdk "bin\java.exe") -version 2>&1 | ForEach-Object { Write-Host "    $_" }
        Write-Host "  Java compiler:"
        & (Join-Path $jdk "bin\javac.exe") -version 2>&1 | ForEach-Object { Write-Host "    $_" }
        Write-RippleToolchain $jdk (Resolve-RippleMaven)
        Write-Host "  [OK] JDK $RequiredJavaMajor+ toolchain validated"
        exit 0
    }
    "ValidateConda" {
        $missing = @(Get-RippleMissingModules)
        $reqHash = Get-RippleReqHash
        $stampHash = Get-RippleStampHash
        if ($missing.Count -eq 0) {
            if ($reqHash -and ($reqHash -eq $stampHash)) {
                Write-Host "  [OK] Existing ripple-env environment validated (fast path)"
                exit 0
            }
            if (-not $stampHash) {
                Write-RippleCondaStamp
                Write-Host "  [OK] Existing ripple-env environment validated; stamp written"
                exit 0
            }
            Write-Host "  [!] Requirements changed since last install; refreshing dependencies"
            exit 2
        }
        Write-Host "  [!] Incomplete ripple-env environment; missing modules: $($missing -join ' ')"
        exit 2
    }
    "RepairConda" {
        Write-Host "  Upgrading pip/wheel/setuptools..."
        & python -m pip install --upgrade pip wheel setuptools -q
        $req = Get-RippleReqFile
        if ($GpuMode -eq "gpu" -and (Test-Path $req)) {
            Write-Host "  Installing GPU packages (this may take a few minutes)..."
        } else {
            Write-Host "  Installing CPU packages..."
            if (-not (Test-Path $req)) {
                $req = Join-Path $ProjectDir "requirements\requirements-cpu.txt"
            }
        }
        & python -m pip install -r $req -q
        if ($LASTEXITCODE -ne 0) { exit 1 }
        Write-RippleCondaStamp
        Write-Host "  [OK] Dependencies installed"
        exit 0
    }
    "Shortcut" {
        if ([string]::IsNullOrWhiteSpace($ShortcutTarget)) {
            $ShortcutTarget = Join-Path $ProjectDir "RIPPLE.bat"
        }
        $desktop = [Environment]::GetFolderPath("Desktop")
        $lnk = Join-Path $desktop "RIPPLE.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($lnk)
        $shortcut.TargetPath = $ShortcutTarget
        $shortcut.WorkingDirectory = $ProjectDir
        $shortcut.WindowStyle = 1
        $shortcut.Description = "RIPPLE Video Annotation Tool"
        $shortcut.Save()
        Write-Host "  [OK] Created desktop shortcut"
        exit 0
    }
}
