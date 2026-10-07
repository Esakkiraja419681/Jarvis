# setup-windows-daemon.ps1 - Jarvis OS background voice daemon helper (Windows).
# Actions: install | register | unregister | start | stop | status
param([ValidateSet("install","register","unregister","start","stop","status")]
      [string]$Action = "install")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$StopFile = Join-Path $Root "data\STOP"
$Heartbeat = Join-Path $Root "data\heartbeat.json"
$TaskName = "JarvisOS"
$ModelDir = Join-Path $Root "models\vosk-model-small-en-us-0.15"
$ModelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
function Write-Msg($m) { Write-Host "[setup] $m" }
function Install-Dependencies {
    Write-Msg "installing Python dependencies (pyautogui, vosk, sounddevice, pyttsx3)..."
    python -m pip install --quiet pyautogui vosk sounddevice pyttsx3
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    if (-not (Test-Path (Join-Path $ModelDir "am"))) {
        Write-Msg "downloading Vosk model (~40 MB)..."
        $zip = Join-Path $Root "models\vosk-model-small-en-us-0.15.zip"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $zip) | Out-Null
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $ModelUrl -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath (Join-Path $Root "models") -Force
        Remove-Item $zip -Force
        Write-Msg "model extracted to models\vosk-model-small-en-us-0.15"
    } else { Write-Msg "model already present." }
    Write-Msg "install complete."
}
function Register-Task {
    $exe = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = "pythonw" }
    $action = New-ScheduledTaskAction -Execute $exe -Argument "`"$Root\jarvis.py`" --daemon --voice" -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    try {
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    } catch {
        throw "Register-ScheduledTask failed: $($_.Exception.Message) - run PowerShell as Administrator to register."
    }
    Write-Msg "registered '$TaskName' (at logon, interactive session, pythonw jarvis.py --daemon --voice)."
}
function Get-DaemonProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*jarvis.py*--daemon*' }
}
function Stop-Daemon {
    New-Item -ItemType File -Path $StopFile -Force | Out-Null
    Write-Msg "stop requested; waiting up to 15s for graceful exit..."
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $alive = @(Get-DaemonProcesses)
        if ($alive.Count -eq 0) { break }
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    foreach ($p in @(Get-DaemonProcesses)) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
    Write-Msg "stopped and unregistered."
}
switch ($Action) {
    "install"     { Install-Dependencies }
    "register"    { Register-Task }
    "unregister"  { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue; Write-Msg "unregistered." }
    "start"       {
        Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
        try {
            Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
            Write-Msg "scheduled task started."
        } catch {
            Write-Msg "task not registered; registering now..."
            Register-Task
            Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
            Write-Msg "scheduled task started."
        }
    }
    "stop"        { Stop-Daemon }
    "status"      {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
            Write-Msg "task: $($task.State)  last result: $($info.LastTaskResult)"
        } else { Write-Msg "task: not registered" }
        $pids = @(Get-DaemonProcesses | ForEach-Object { $_.ProcessId })
        Write-Msg "daemon pids: $(if ($pids) { $pids -join ', ' } else { '(none)' })"
        if (Test-Path $Heartbeat) {
            $hb = Get-Content $Heartbeat -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($hb -and $null -ne $hb.ts) {
                $ts = $hb.ts
                $when = if ($ts -is [string]) { ([datetime]$ts).ToUniversalTime() }
                        else { [DateTimeOffset]::FromUnixTimeMilliseconds([long]([double]$ts * 1000)).UtcDateTime }
                Write-Msg "heartbeat age: $([math]::Round(((Get-Date).ToUniversalTime() - $when).TotalSeconds, 1))s"
            }
        } else { Write-Msg "heartbeat: missing (daemon not running)" }
        $deps = python -c "import importlib.util,json; print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in ['pyautogui','vosk','sounddevice','pyttsx3']}))"
        Write-Msg "deps: $deps"
        Write-Msg "vosk model: $(if (Test-Path (Join-Path $ModelDir 'am')) { 'present' } else { 'MISSING (run -Action install)' })"
        if (Test-Path $StopFile) { Write-Msg "WARNING: data\STOP exists - stale stop request, run -Action start to clear" }
    }
}
