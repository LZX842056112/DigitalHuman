param([string]$AvatarId = '')
. (Join-Path $PSScriptRoot 'common.ps1')
$arguments = @('-m','scripts.build_head')
if ($AvatarId) { $arguments += @('--avatar-id',$AvatarId) }
Invoke-ProjectWSL 'run_python.sh' $arguments
