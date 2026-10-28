param([switch]$StudioOnly)
. (Join-Path $PSScriptRoot 'common.ps1')
$arguments = @('-m','scripts.services','start')
if ($StudioOnly) { $arguments += '--studio-only' }
Invoke-ProjectWSL 'run_python.sh' $arguments
