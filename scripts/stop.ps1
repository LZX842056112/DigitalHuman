. (Join-Path $PSScriptRoot 'common.ps1')
Invoke-ProjectWSL 'run_python.sh' @('-m','scripts.services','stop')
