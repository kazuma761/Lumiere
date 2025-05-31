$obsPath = "C:\Program Files\obs-studio"
$obsVCamPath = "C:\Program Files\obs-studio\data\obs-plugins\win-dshow\obs-virtualcam.dll"

if (-not (Test-Path $obsPath)) {
    Write-Host "OBS Studio not found. Downloading and installing..."
    $obsUrl = "https://cdn-fastly.obsproject.com/downloads/OBS-Studio-29.1.3-Full-Installer-x64.exe"
    $installerPath = "$env:TEMP\OBS-Studio-Installer.exe"
    
    # Download OBS
    Invoke-WebRequest -Uri $obsUrl -OutFile $installerPath
    
    # Install OBS silently
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
    
    # Clean up
    Remove-Item $installerPath
    
    Write-Host "OBS Studio installed successfully!"
} else {
    Write-Host "OBS Studio is already installed"
}

# Start OBS Virtual Camera if not running
if (Test-Path $obsVCamPath) {
    Write-Host "Starting OBS Virtual Camera..."
    Start-Process "C:\Program Files\obs-studio\bin\64bit\obs-virtualsource.exe" -ArgumentList "start"
} else {
    Write-Host "OBS Virtual Camera not found. Please run OBS Studio once to complete setup."
}
