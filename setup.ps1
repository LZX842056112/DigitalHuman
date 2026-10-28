param([switch]$EnvironmentOnly,[switch]$SkipModels)
& (Join-Path $PSScriptRoot 'scripts/setup.ps1') @PSBoundParameters
