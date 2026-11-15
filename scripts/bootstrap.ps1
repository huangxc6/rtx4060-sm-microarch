<#
    bootstrap.ps1 —— 一键准备工具链（无需管理员、无需 CUDA Toolkit）

    做四件事：
      1. 建 .venv/
      2. 从镜像下 nvidia-cuda-nvrtc 的 wheel，当 zip 解压进 site-packages
         （不走 pip install：本机 pip 在解析阶段会空转卡死）
      3. 从 NVIDIA redist 下 cuobjdump / nvdisasm 的独立 zip，解压到 build/tools/
      4. 跑 smoke.py 自检

    用法：
      powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
#>
[CmdletBinding()]
param(
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "工作目录: $Root`n" -ForegroundColor Cyan

function Get-File([string]$Url, [string]$Out) {
    if (Test-Path $Out) {
        Write-Host ("  已存在: {0}" -f (Split-Path $Out -Leaf)) -ForegroundColor DarkGray
        return
    }
    $name = Split-Path $Out -Leaf
    Write-Host ("  下载 {0} ..." -f $name) -NoNewline
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) pip/24.0"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Out -Headers @{ "User-Agent" = $UA } `
            -TimeoutSec 900 -UseBasicParsing
    } catch {
        Write-Host " 失败" -ForegroundColor Red
        throw "下载失败: $Url`n$($_.Exception.Message)"
    }
    $sw.Stop()
    Write-Host (" {0:N1} MB / {1:N0}s" -f ((Get-Item $Out).Length / 1MB), $sw.Elapsed.TotalSeconds) `
        -ForegroundColor Green
}

function Expand-Zip([string]$Zip, [string]$Dest) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Dest)
}

# ---------------------------------------------------------------- 1. venv
Write-Host "[1/4] 虚拟环境" -ForegroundColor Yellow
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $sys = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $sys) { throw "找不到 python，请先安装 Python 3.10+ 并加入 PATH" }
    python -m venv (Join-Path $Root ".venv")
}
& $Py -V
$SitePkgs = Join-Path $Root ".venv\Lib\site-packages"

# ------------------------------------------------------- 2. nvrtc（wheel）
Write-Host "`n[2/4] NVRTC" -ForegroundColor Yellow
$nvrtcDll = Join-Path $SitePkgs "nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll"
if (Test-Path $nvrtcDll) {
    Write-Host "  已存在: nvrtc64_120_0.dll" -ForegroundColor DarkGray
} else {
    # 固定版本 12.9.86（CUDA 12.9 的最后一个 cu12 版本，能编 sm_89 且被新驱动加载）
    $whl = Join-Path $Root "build\wheels\nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl"
    New-Item -ItemType Directory -Force -Path (Split-Path $whl) | Out-Null
    $pkgPath = "packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86"
    Get-File "$Mirror/$pkgPath/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl" $whl
    Write-Host "  解压到 site-packages ..." -NoNewline
    Expand-Zip $whl $SitePkgs
    Write-Host " 完成" -ForegroundColor Green
}
if (-not (Test-Path $nvrtcDll)) { throw "nvrtc 安装后仍找不到 $nvrtcDll" }

# ------------------------------------- 3. cuobjdump / nvdisasm（redist zip）
Write-Host "`n[3/4] 反汇编器（cuobjdump / nvdisasm）" -ForegroundColor Yellow
$redist = "https://developer.download.nvidia.com/compute/cuda/redist"
$tools = @(
    @{ n = "cuobjdump"; u = "$redist/cuda_cuobjdump/windows-x86_64/cuda_cuobjdump-windows-x86_64-12.9.26-archive.zip" },
    @{ n = "nvdisasm";  u = "$redist/cuda_nvdisasm/windows-x86_64/cuda_nvdisasm-windows-x86_64-12.9.19-archive.zip" }
)
$toolDir = Join-Path $Root "build\tools"
$zipDir = Join-Path $Root "build\wheels"
New-Item -ItemType Directory -Force -Path $toolDir, $zipDir | Out-Null
foreach ($t in $tools) {
    if (Get-ChildItem $toolDir -Recurse -Filter "$($t.n).exe" -ErrorAction SilentlyContinue) {
        Write-Host ("  已存在: {0}.exe" -f $t.n) -ForegroundColor DarkGray
        continue
    }
    $z = Join-Path $zipDir (Split-Path $t.u -Leaf)
    Get-File $t.u $z
    Write-Host ("  解压 {0} ..." -f $t.n) -NoNewline
    Expand-Zip $z (Join-Path $toolDir ([IO.Path]::GetFileNameWithoutExtension($z) -replace '-archive$', ''))
    Write-Host " 完成" -ForegroundColor Green
}

# ------------------------------------------------------------- 4. 自检
Write-Host "`n[4/4] 链路自检 (smoke.py)" -ForegroundColor Yellow
& $Py (Join-Path $Root "smoke.py")
if ($LASTEXITCODE -ne 0) { throw "smoke.py 失败（退出码 $LASTEXITCODE）" }

Write-Host @"

工具链就绪。下一步：

    .\.venv\Scripts\python.exe run_all.py        # 跑全部测量（约 14 分钟）
    .\.venv\Scripts\python.exe docs\make_html.py # 重新生成 HTML 报告

"@ -ForegroundColor Cyan
