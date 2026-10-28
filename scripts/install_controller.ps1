param(
    [string]$Python = 'C:\MySoftware\Anaconda3\envs\python311\python.exe',
    [int]$ModelDownloadPid = 0,
    [int]$PreviewPid = 0,
    [switch]$NoBuild
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$statePath=Join-Path $taskRoot '.runtime/install-controller.json'
function Set-InstallStage([string]$stage,[string]$status='running',[string]$message='') {
    @{stage=$stage;status=$status;message=$message;updated=(Get-Date -Format o);pid=$PID} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Output "$stage : $status"
}
$modelProcess=if($ModelDownloadPid -gt 0){Get-Process -Id $ModelDownloadPid -ErrorAction SilentlyContinue}else{$null}
$modelStarted=if($modelProcess){$modelProcess.StartTime.Ticks}else{0}
$previewProcess=if($PreviewPid -gt 0){Get-Process -Id $PreviewPid -ErrorAction SilentlyContinue}else{$null}
$previewStarted=if($previewProcess){$previewProcess.StartTime.Ticks}else{0}
try {
    Set-InstallStage 'pytorch_download'
    $wheel=Join-Path $taskRoot '.runtime/torch-2.3.0+cu121-cp310-cp310-linux_x86_64.whl'
    $downloaded=$false
    for($attempt=1;$attempt -le 3;$attempt++) {
        & $Python -u -m scripts.fetch_artifact 'https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl' $wheel '0a12aa9aa6bc442dff8823ac8b48d991fd0771562eaa38593f9c8196d65f7007' 780956465
        if($LASTEXITCODE -eq 0){$downloaded=$true;break}
        if($attempt -lt 3){Start-Sleep -Seconds 30}
    }
    if(-not $downloaded){throw 'PyTorch download did not pass verification. Verified partial data is retained.'}
    Set-InstallStage 'environment_install'
    & (Join-Path $PSScriptRoot 'setup.ps1') -EnvironmentOnly
    Set-InstallStage 'waiting_for_model_download'
    while($modelStarted -ne 0) {
        $current=Get-Process -Id $ModelDownloadPid -ErrorAction SilentlyContinue
        if(-not $current -or $current.StartTime.Ticks -ne $modelStarted){break}
        Start-Sleep -Seconds 15
    }
    Set-InstallStage 'models_and_runtime'
    & (Join-Path $PSScriptRoot 'setup.ps1')
    if($NoBuild){Set-InstallStage 'installed' 'complete';exit 0}
    # Close only this chat's temporary preview after runtime installation.
    $preview=Get-Process -Id $PreviewPid -ErrorAction SilentlyContinue
    if($preview -and $preview.StartTime.Ticks -eq $previewStarted) {
        $details=Get-CimInstance Win32_Process -Filter "ProcessId=$PreviewPid"
        if($details.CommandLine -like '*backend.api.app:app*--port*7860*') {
            Stop-Process -Id $PreviewPid
        }
    }
    Set-InstallStage 'start_local_services'
    & (Join-Path $PSScriptRoot 'start.ps1')
    Set-InstallStage 'build_real_avatar'
    & (Join-Path $PSScriptRoot 'build_avatar.ps1') -Stage all
    Set-InstallStage 'technical_build_finished' 'complete' 'Visual, voice identity and long session acceptance still require verification.'
} catch {
    Set-InstallStage 'failed' 'failed' $_.Exception.Message
    Write-Error $_
    exit 1
}
