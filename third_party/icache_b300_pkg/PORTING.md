# 移植说明（本目录是外部包）

来源：用户提供的 `icache_b300_pkg.tgz`，一套面向 **B300 / sm_103** 的
指令缓存 + L1D 微基准。原样需要 Linux + nvcc + make + 独占 sm_90+ 卡，
本机（Windows、无 CUDA Toolkit、RTX 4060 = sm_89、非独占）跑不了。

## 能复用什么

内核是**纯 device 代码**（inline PTX 算子 + `clock64` 计时 + 按值传的 `BenchArgs`），
所以用本工程的 nvrtc 链路重新编译即可：

```powershell
cd third_party\icache_b300_pkg
..\..\.venv\Scripts\python.exe gen_kernels.py --out-dir build --families fp,dual
..\..\.venv\Scripts\python.exe ..\..\run_all.py --round 12
```

关键点（都已在 `probes/round12_icache.py` 里实现）：

1. 生成的头文件末尾有 host 侧名册 `static const KernelInfo kKernels[]`，
   引用了 `KernelInfo`/`KIND_*`，nvrtc 只编 device 代码 → **从那里截断**。
2. 内核签名是 `__global__ void f(BenchArgs args)`，**结构体按值传**（168 字节）。
   `cuLaunchKernel` 的 `kernelParams` 是 `void**`，每个元素指向参数值，
   所以用 ctypes 镜像同样的布局、把指向它的指针放进数组即可。
3. `record()` / `c_arr` / `kBlockThreads` 在 `icache_bench.cu` 里定义，用 shim 补。

## 没搬的部分

`cf`（行大小/组相联）、`qd`（分区带宽）、`vis`（跨分区可见性）、`seq`（替换策略）、
`bw`（取指带宽矩阵）、`gpcmap`（需要 sm_90 的 thread block cluster）。
