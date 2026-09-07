[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$detectorPath = Join-Path $repoRoot 'yolo_service\detector_v7.py'
$venvPython = Join-Path $repoRoot '.venv311\Scripts\python.exe'
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
$mutex = [System.Threading.Mutex]::new($false, 'Local\OccupAI-Two-Camera-Launcher')
$ownsMutex = $false

$workers = @(
    [pscustomobject]@{ Role = 'car'; CameraId = 'cars'; WebcamIndex = 0; Port = 8001; Process = $null; Restarts = 0; RetryAt = [datetime]::MinValue },
    [pscustomobject]@{ Role = 'motorcycle'; CameraId = 'motorcycles'; WebcamIndex = 1; Port = 8002; Process = $null; Restarts = 0; RetryAt = [datetime]::MinValue }
)

function Test-PortAvailable([int]$Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -eq $listener
}

function Start-CameraWorker($Worker) {
    if (-not (Test-PortAvailable $Worker.Port)) {
        throw "Port $($Worker.Port) is already in use. Stop the duplicate worker before launching."
    }

    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $pythonExe
    $start.Arguments = '"' + $detectorPath + '"'
    $start.WorkingDirectory = $repoRoot
    $start.UseShellExecute = $false
    $start.EnvironmentVariables['CAMERA_ROLE'] = $Worker.Role
    $start.EnvironmentVariables['CAMERA_ID'] = $Worker.CameraId
    $start.EnvironmentVariables['WEBCAM_INDEX'] = [string]$Worker.WebcamIndex
    $start.EnvironmentVariables['STREAM_PORT'] = [string]$Worker.Port
    $start.EnvironmentVariables['AUTO_CAMERA_RECALIBRATE'] = 'true'

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) {
        $process.Dispose()
        throw "Could not start the $($Worker.Role) camera worker."
    }
    $Worker.Process = $process
    Write-Host "Started $($Worker.Role) worker (PID $($process.Id), webcam $($Worker.WebcamIndex), port $($Worker.Port))."
}

try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw 'An OccupAI two-camera launcher is already running.'
    }

    foreach ($worker in $workers) {
        Start-CameraWorker $worker
    }
    Write-Host 'Both OccupAI workers are supervised. Press Ctrl+C to stop them.'

    while ($true) {
        foreach ($worker in $workers) {
            if ($null -ne $worker.Process -and $worker.Process.HasExited) {
                $exitCode = $worker.Process.ExitCode
                $worker.Process.Dispose()
                $worker.Process = $null
                $worker.Restarts++
                $delay = [Math]::Min(30, [Math]::Pow(2, [Math]::Min(4, $worker.Restarts - 1)))
                $worker.RetryAt = (Get-Date).AddSeconds($delay)
                Write-Warning "$($worker.Role) worker exited with code $exitCode. Restarting in $delay second(s)."
            }

            if ($null -eq $worker.Process -and (Get-Date) -ge $worker.RetryAt) {
                try {
                    Start-CameraWorker $worker
                    $worker.RetryAt = [datetime]::MinValue
                }
                catch {
                    $worker.RetryAt = (Get-Date).AddSeconds(5)
                    Write-Warning "$($worker.Role) worker restart failed: $($_.Exception.Message) Retrying in 5 seconds."
                }
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($worker in $workers) {
        if ($null -ne $worker.Process) {
            if (-not $worker.Process.HasExited) {
                $worker.Process.Kill()
                $worker.Process.WaitForExit()
            }
            $worker.Process.Dispose()
        }
    }
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
