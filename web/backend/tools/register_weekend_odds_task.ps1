$ErrorActionPreference = 'Stop'

$workspace = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$pythonCmd = "python"
$scriptCmd = "`"$workspace\backend\tools\refresh_laliga_odds.py`" --limit 200"

$action = New-ScheduledTaskAction -Execute $pythonCmd -Argument $scriptCmd -WorkingDirectory $workspace
$triggerFriday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At 09:00AM
$triggerFriday.Repetition = (New-ScheduledTaskTrigger -Once -At 09:00AM -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition

$triggerSaturday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 09:00AM
$triggerSaturday.Repetition = (New-ScheduledTaskTrigger -Once -At 09:00AM -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition

$triggerSunday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 09:00AM
$triggerSunday.Repetition = (New-ScheduledTaskTrigger -Once -At 09:00AM -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Hours 15)).Repetition

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "TFM_Odds_Weekend_Snapshots" -Action $action -Trigger @($triggerFriday, $triggerSaturday, $triggerSunday) -Principal $principal -Settings $settings -Force | Out-Null
Write-Output "Tarea creada/actualizada: TFM_Odds_Weekend_Snapshots (vie-sab-dom cada 2h entre 09:00 y 24:00)."
