param([string]$AvatarId = '', [string]$Photo = '', [string]$VoiceVideo = '', [string]$Name = '新数字人')
. (Join-Path $PSScriptRoot 'common.ps1')
$arguments = @('-m','scripts.build_avatar')
if ($AvatarId) { $arguments += @('--avatar-id',$AvatarId) }
if ($Photo) { $arguments += @('--photo',(ConvertTo-ProjectLinuxPath (Resolve-Path -LiteralPath $Photo).Path)) }
if ($VoiceVideo) { $arguments += @('--voice-video',(ConvertTo-ProjectLinuxPath (Resolve-Path -LiteralPath $VoiceVideo).Path)) }
$arguments += @('--name',$Name)
Invoke-ProjectWSL 'run_python.sh' $arguments
