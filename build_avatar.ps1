param([string]$Stage='all',[string]$AvatarId='',[switch]$SkipAcceptance)
& (Join-Path $PSScriptRoot 'scripts/build_avatar.ps1') @PSBoundParameters
