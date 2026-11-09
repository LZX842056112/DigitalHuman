param(
    [Parameter(Mandatory = $true)][string]$AvatarId,
    [string]$Text = '',
    [string]$Emotion = 'neutral',
    [string]$Wav = '',
    [switch]$Diagnose
)
. (Join-Path $PSScriptRoot 'common.ps1')
$arguments = @('scripts/voice-similarity.py', '--avatar', $AvatarId)
if ($Text) { $arguments += @('--text', $Text) }
if ($Emotion) { $arguments += @('--emotion', $Emotion) }
if ($Wav) { $arguments += @('--wav', (ConvertTo-ProjectLinuxPath (Resolve-Path -LiteralPath $Wav).Path)) }
if ($Diagnose) { $arguments += @('--diagnose') }
Invoke-ProjectWSL 'run_python.sh' $arguments
