param([string]$Python = "python", [ValidateSet("cuda","cpu")][string]$Device = "cuda")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe $Arguments" }
}
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Invoke-Checked $Python @("-m", "venv", ".venv")
}
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
Invoke-Checked $taskPython @("-m","pip","install","pip==24.3.1","setuptools==75.6.0","wheel==0.45.1","swig==4.3.0")
$index = if ($Device -eq "cuda") { "https://download.pytorch.org/whl/cu121" } else { "https://download.pytorch.org/whl/cpu" }
Invoke-Checked $taskPython @("-m","pip","install","torch==2.5.1","--index-url",$index)
Invoke-Checked $taskPython @("-m","pip","install","--no-build-isolation","-r","requirements.txt")
Invoke-Checked $taskPython @("-m","pip","check")
Invoke-Checked $taskPython @("-m","pytest","-q")
Write-Host 'Setup complete. Run .venv\Scripts\python.exe -m racing.suite --preset smoke'
