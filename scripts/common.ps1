$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Distro = 'DigitalHuman'
function ConvertTo-ProjectLinuxPath {
    param([string]$WindowsPath)
    if ($WindowsPath -notmatch '^[A-Za-z]:\\') { throw 'An absolute Windows drive path is required.' }
    $drive = $WindowsPath.Substring(0,1).ToLowerInvariant()
    return '/mnt/' + $drive + '/' + $WindowsPath.Substring(3).Replace('\','/')
}
function Invoke-ProjectWSL {
    param([string]$Script, [string[]]$ScriptArguments = @())
    $linuxRoot = ConvertTo-ProjectLinuxPath $ProjectRoot
    & wsl.exe -d $Distro -u root --exec bash "$linuxRoot/scripts/$Script" $linuxRoot @ScriptArguments
    if ($LASTEXITCODE -ne 0) { throw "Project command failed ($LASTEXITCODE). See .runtime/logs." }
}
