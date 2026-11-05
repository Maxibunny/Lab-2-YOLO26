# Bootstrap the OFFICIAL Gradle 8.9 wrapper without requiring a preinstalled Gradle.
# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\bootstrap_gradle_windows.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cache = Join-Path $env:USERPROFILE '.gradle-bootstrap'
$zipFile = Join-Path $cache 'gradle-8.9-bin.zip'
$gradle = Join-Path $cache 'gradle-8.9\bin\gradle.bat'
if (!(Test-Path $gradle)) {
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    Write-Host 'Downloading Gradle 8.9 from official services.gradle.org...'
    Invoke-WebRequest -Uri 'https://services.gradle.org/distributions/gradle-8.9-bin.zip' -OutFile $zipFile
    Expand-Archive -Path $zipFile -DestinationPath $cache -Force
}
Push-Location $root
try {
    & $gradle wrapper --gradle-version 8.9
    if ($LASTEXITCODE -ne 0) { throw "Gradle returned exit code $LASTEXITCODE" }
    Write-Host 'Official Gradle wrapper created. Open this Android folder in Android Studio.'
} finally { Pop-Location }
