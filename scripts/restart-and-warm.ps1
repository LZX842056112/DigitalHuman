param(
    [int]$StartupWaitSeconds = 240,
    [int]$WarmupWaitSeconds = 900,
    [int]$BuildWaitSeconds = 900
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

function Get-Json([string]$Url) {
    $raw = & curl.exe --noproxy '*' -sS -m 8 $Url 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
    try { return ($raw -join "`n") | ConvertFrom-Json } catch { return $null }
}

Write-Output '1/4 stop old services'
Invoke-ProjectWSL 'run_python.sh' @('-m', 'scripts.services', 'stop')
Write-Output '2/4 start services; models warm up in the background'
Invoke-ProjectWSL 'run_python.sh' @('-m', 'scripts.services', 'start')

$deadline = (Get-Date).AddSeconds($StartupWaitSeconds)
$health = $null
while ((Get-Date) -lt $deadline) {
    $health = Get-Json 'http://127.0.0.1:7860/api/health'
    if ($health -and $health.status -eq 'ok') { break }
    Start-Sleep -Seconds 3
}
if (-not $health -or $health.status -ne 'ok') { throw "studio did not become healthy within $StartupWaitSeconds s" }
Write-Output ("    studio ok, ready_avatars=" + $health.ready_avatars)

Write-Output '3/4 wait for model warmup (no longer paid by the first answer)'
$workers = @(
    @{ name = 'voice'; url = 'http://127.0.0.1:9880/health' },
    @{ name = 'portrait'; url = 'http://127.0.0.1:9881/health' }
)
$deadline = (Get-Date).AddSeconds($WarmupWaitSeconds)
$pending = $workers
$startedAt = Get-Date
while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
    $still = @()
    foreach ($worker in $pending) {
        $body = Get-Json $worker.url
        if ($body -and $body.loaded -eq $true) {
            $elapsed = [int]((Get-Date) - $startedAt).TotalSeconds
            Write-Output ("    " + $worker.name + " warm after " + $elapsed + " s")
        }
        else { $still += $worker }
    }
    $pending = $still
    if ($pending.Count -gt 0) { Start-Sleep -Seconds 5 }
}
if ($pending.Count -gt 0) {
    Write-Output '    warmup still running; check .runtime/logs/voice.log and .runtime/logs/portrait.log'
}

Write-Output '4/4 rebuild avatars that are not ready'
$avatars = Get-Json 'http://127.0.0.1:7860/api/avatars'
$waiting = @($avatars | Where-Object { $_.status -ne 'ready' })
if ($waiting.Count -eq 0) { Write-Output '    every avatar is already ready' }
foreach ($avatar in $waiting) {
    $raw = & curl.exe --noproxy '*' -sS -m 15 -X POST "http://127.0.0.1:7860/api/avatars/$($avatar.id)/build" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) { Write-Output ("    " + $avatar.name + ": build request rejected"); continue }
    $job = ($raw -join "`n") | ConvertFrom-Json
    $deadline = (Get-Date).AddSeconds($BuildWaitSeconds)
    $last = ''
    while ((Get-Date) -lt $deadline) {
        $state = Get-Json ("http://127.0.0.1:7860/api/jobs/" + $job.job_id)
        if (-not $state) { Start-Sleep -Seconds 3; continue }
        $last = ([string]$state.status + '/' + [string]$state.stage)
        if ($state.status -eq 'complete' -or $state.status -eq 'failed' -or $state.status -eq 'waiting') { break }
        Start-Sleep -Seconds 5
    }
    Write-Output ("    " + $avatar.name + ": " + $last)
    if ($last -like 'failed*') { Write-Output ("      error: " + $state.error) }
}

$final = Get-Json 'http://127.0.0.1:7860/api/health'
Write-Output ("done: ready_avatars=" + $final.ready_avatars)
Write-Output 'open http://localhost:7860 and ask a question; the first answer no longer pays model load'
