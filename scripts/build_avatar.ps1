param([string]$Stage = 'all', [string]$AvatarId = '', [switch]$SkipAcceptance)
. (Join-Path $PSScriptRoot 'common.ps1')
$arguments = @('-m','scripts.build_avatar','--stage',$Stage)
if ($AvatarId) { $arguments += @('--avatar-id',$AvatarId) }
Invoke-ProjectWSL 'run_python.sh' $arguments
if (-not $SkipAcceptance -and $Stage -in @('all','validate')) {
    if($AvatarId){
        Invoke-ProjectWSL 'run_python.sh' @('-m','scripts.acceptance','--avatar-id',$AvatarId,'--minutes','10','--rounds','10')
    }else{
        Invoke-ProjectWSL 'run_python.sh' @('-m','scripts.post_build')
    }
}
