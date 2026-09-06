[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$detectorPath = Join-Path $repoRoot 'yolo_service\detector_v7.py'
$venvPython = Join-Path $repoRoot '.venv311\Scripts\python.exe'
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
$mutex = [System.Threading.Mutex]::new($false, 'Local\OccupAI-Two-Camera-Launcher')
$ownsMutex = $false
$workers = @()

function Test-PortAvailable([int]$Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -eq $listener
}

function Start-CameraWorker([string]$Role, [string]$CameraId, [int]$WebcamIndex, [int]$Port) {
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $pythonExe
    $start.Arguments = '"' + $detectorPath + '"'
    $start.WorkingDirectory = $repoRoot
    $start.UseShellExecute = $false
    $start.EnvironmentVariables['CAMERA_ROLE'] = $Role
    $start.EnvironmentVariables['CAMERA_ID'] = $CameraId
    $start.EnvironmentVariables['WEBCAM_INDEX'] = [string]$WebcamIndex
    $start.EnvironmentVariables['STREAM_PORT'] = [string]$Port
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) {
        throw "Could not start the $Role camera worker."
    }
    Write-Host "Started $Role worker (PID $($process.Id), webcam $WebcamIndex, port $Port)."
    return [pscustomobject]@{ Role = $Role; Process = $process; Reported = $false }
}

try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw 'An OccupAI two-camera launcher is already running.'
    }
    foreach ($port in 8001, 8002) {
        if (-not (Test-PortAvailable $port)) {
            throw "Port $port is already in use. Stop the duplicate worker before launching."
        }
    }

    $workers += Start-CameraWorker -Role 'car' -CameraId 'cars' -WebcamIndex 0 -Port 8001
    $workers += Start-CameraWorker -Role 'motorcycle' -CameraId 'motorcycles' -WebcamIndex 1 -Port 8002
    Write-Host 'Both OccupAI camera workers are running. Press Ctrl+C to stop them.'

    while ($true) {
        foreach ($worker in $workers) {
            if ($worker.Process.HasExited -and -not $worker.Reported) {
                $worker.Reported = $true
                Write-Error "$($worker.Role) worker exited with code $($worker.Process.ExitCode)."
            }
        }
        if (($workers | Where-Object { -not $_.Process.HasExited }).Count -eq 0) { break }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($worker in $workers) {
        if (-not $worker.Process.HasExited) {
            $worker.Process.Kill()
            $worker.Process.WaitForExit()
        }
        $worker.Process.Dispose()
    }
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
