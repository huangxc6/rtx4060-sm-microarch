<#
    run_all.ps1 —— 跑测量并把控制台编码设好（避免中文乱码）

    用法：
      powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
      powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 --round 2 3
      powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 --round 3 --tag mem
#>
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# 控制台输出中文，需要这两行（否则会看到 "΢�ܹ�"）
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "找不到 .venv，请先运行 scripts\bootstrap.ps1"
}

& $Py (Join-Path $Root "run_all.py") @Rest
exit $LASTEXITCODE
