param([switch]$EnvironmentOnly, [switch]$SkipModels)
. (Join-Path $PSScriptRoot 'common.ps1')
$runtime = Join-Path $ProjectRoot '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$volume = Get-PSDrive -Name ([IO.Path]::GetPathRoot($ProjectRoot).Substring(0,1))
if ($volume.Free -lt 10GB) { throw 'At least 10 GiB free space is required to continue installation.' }
$names = ((& wsl.exe --list --quiet 2>$null) -join "`n").Replace([string][char]0, '')
if (($names -split "`n" | ForEach-Object { $_.Trim() }) -notcontains $Distro) {
    $archive = Join-Path $runtime 'ubuntu22.04.rootfs.tar.gz'
    $expected = '1483cc5c1dce13064f774834cbffdff226559fd522a67a381a8ea77d63fb4109'
    $valid = (Test-Path -LiteralPath $archive) -and ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expected)
    if (-not $valid) {
        & curl.exe -fL --retry 3 --connect-timeout 20 --continue-at - --output $archive 'https://cloud-images.ubuntu.com/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz'
        if ($LASTEXITCODE -ne 0) { throw 'Ubuntu download failed. Re-run to resume.' }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Ubuntu SHA-256 mismatch. No image was registered.' }
    & wsl.exe --import $Distro (Join-Path $runtime 'ubuntu') $archive --version 2
    if ($LASTEXITCODE -ne 0) { throw 'WSL registration failed. Check WSL/Virtual Machine Platform and restart Windows if requested.' }
}
$arguments = @()
if ($EnvironmentOnly) { $arguments += '--environment-only' }
if ($SkipModels) { $arguments += '--skip-models' }
Invoke-ProjectWSL 'setup_linux.sh' $arguments
if (-not $EnvironmentOnly -and -not $SkipModels) { Invoke-ProjectWSL 'setup_head.sh' @() }
@{ distro=$Distro; installed=(Get-Date -Format o); environment_only=[bool]$EnvironmentOnly } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime 'setup-state.json') -Encoding UTF8
