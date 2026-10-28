$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath (Join-Path $root 'frontend')
& npm.cmd install --cache (Join-Path $root '.runtime/cache/npm') --registry=https://registry.npmmirror.com --fetch-retries=4 --fetch-timeout=30000
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
& npm.cmd run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend compilation failed.' }
