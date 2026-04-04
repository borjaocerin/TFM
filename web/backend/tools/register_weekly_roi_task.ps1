$ErrorActionPreference = 'Stop'

$workspace = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$pythonCmd = "python"
$scriptCmd = "`"$workspace\backend\tools\compute_upcoming_roi.py`""

$action = New-ScheduledTaskAction -Execute $pythonCmd -Argument $scriptCmd -WorkingDirectory $workspace
$trigger = New-ScheduledTaskTrigger -Daily -At 09:00AM
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "TFM_ROI_Update_Jornada" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Write-Output "Tarea creada/actualizada: TFM_ROI_Update_Jornada (diaria 09:00)."
