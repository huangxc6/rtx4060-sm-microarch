"""计时与测量框架。

核心约定
--------
1. **时间单位是 SM 周期**，取自内核内的 ``%clock64``，不是 CUDA event。
   主机非管理员、无法锁频，绝对时间不可复现（见 docs/00-methodology.md）。
2. **一次测量 = R 次背靠背入队 + 最后一次统一同步**。GPU 在整个测量窗口保持
   满载不掉频，且没有逐次同步的调度抖动。每次重复写不同的 cycles 输出区
   （指针偏移），R 次结果互不覆盖。
3. **差值法**：同一内核取两个迭代数 N1/N2，``(C2-C1)/(N2-N1)``，
   消掉内核启动、clock64 读取、参数装载等全部固定开销。

本模块只依赖标准库（无 numpy）：数据量小，``array`` 足够，少一个依赖少一处坑。
"""
from __future__ import annotations

import time
from array import array

_PRIME_SRC = r"""
// 预热用：足够长的 FP32 依赖链 + 大量 CTA，只为把 GPU 顶到 boost 态。
// 迭代数编译期固定，因此内核参数仍然只有指针，符合本工程的参数约定。
extern "C" __global__ void prime(unsigned long long* sink, const void* in, void* out) {
    float a = threadIdx.x * 1e-6f;
    float b = 1.000001f, c = 0.5f;
    #pragma unroll 1
    for (int i = 0; i < 20000; ++i) {
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == 1.0e30f) sink[threadIdx.x] = a;
}
"""

CLOCK_SRC = r"""
// 用 globaltimer(ns) 与 clock64(cycle) 在内核内同步采样，得到瞬时 SM 频率。
extern "C" __global__ void clock_probe(unsigned long long* cyc, const void* in, void* out) {
    unsigned long long t0, c0, t1, c1, acc;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    asm volatile("mov.u64 %0, %%clock64;"    : "=l"(c0));
    asm volatile("bar.sync 0;");
    acc = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < 50000; ++i) {
        asm volatile("add.u64 %0, %0, 1;" : "+l"(acc));
        asm volatile("add.u64 %0, %0, 1;" : "+l"(acc));
        asm volatile("add.u64 %0, %0, 1;" : "+l"(acc));
        asm volatile("add.u64 %0, %0, 1;" : "+l"(acc));
    }
    asm volatile("bar.sync 0;");
    asm volatile("mov.u64 %0, %%clock64;"    : "=l"(c1));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0) {
        cyc[blockIdx.x * 4 + 0] = c1 - c0;
        cyc[blockIdx.x * 4 + 1] = t1 - t0;
        cyc[blockIdx.x * 4 + 2] = acc;
    }
}
"""


def u64(raw: bytes) -> array:
    a = array("Q")
    a.frombytes(raw)
    return a


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Bench:
    def __init__(self, gpu, log=print):
        self.gpu = gpu
        self.log = log
        self._prime = None
        self._sink = None
        self._clk = None
        self.clock_ghz: float | None = None

    # -- 预热 ------------------------------------------------------------
    def prime(self, duration_ms: float = 250.0, blocks: int = 1024, threads: int = 512):
        """持续满载，把 GPU 拉到 boost 并维持。"""
        from .jit import Module

        if self._prime is None:
            self._prime = Module(self.gpu, _PRIME_SRC, "prime.cu").func("prime")
            self._sink = self.gpu.malloc(threads * 4)
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < duration_ms:
            self._prime.launch(blocks, threads, (self._sink, 0, 0))
        self.gpu.sync()

    # -- 周期测量 --------------------------------------------------------
    def measure(self, kernel, grid: int, block: int, ptrs=(), reps: int = 7,
                smem: int = 0) -> list[list[int]]:
        """返回 (reps x grid) 的周期表。

        被测内核签名必须是 ``(unsigned long long* cyc, ...)``，第 0 个参数是
        cycles 输出区，``cyc[blockIdx.x]`` 写该 CTA 的耗时周期。
        """
        gpu = self.gpu
        stride = grid * 8
        buf = gpu.malloc(reps * stride)
        try:
            for r in range(reps):
                kernel.launch(grid, block, (buf + r * stride,) + tuple(ptrs), smem=smem)
            gpu.sync()
            vals = u64(gpu.dtoh(buf, reps * stride))
        finally:
            gpu.free(buf)
        return [list(vals[r * grid:(r + 1) * grid]) for r in range(reps)]

    def measure1(self, kernel, block: int, ptrs=(), reps: int = 7) -> list[int]:
        """单 CTA（单 SM）测量。"""
        return [row[0] for row in self.measure(kernel, 1, block, ptrs, reps)]

    @staticmethod
    def best(rows: list[list[int]]) -> list[int]:
        """逐 CTA 取 R 次中的最小值：噪声只会让周期变多，最小值最接近真实值。"""
        return [min(col) for col in zip(*rows)]

    def delta(self, k_lo, k_hi, block: int, pts, n_lo: int, n_hi: int,
              reps: int = 7) -> float:
        """差值法：返回 (C_hi - C_lo) / (n_hi - n_lo) 周期/操作。"""
        c_lo = self.best([self.measure1(k_lo, block, pts, reps)])[0]
        c_hi = self.best([self.measure1(k_hi, block, pts, reps)])[0]
        return (c_hi - c_lo) / float(n_hi - n_lo)

    # -- 墙钟（仅用于毫秒级的大内核，如带宽测量）-------------------------
    def wall_s(self, kernel, grid, block, ptrs=(), reps: int = 5,
               smem: int = 0) -> float:
        """返回 best-of-reps 的墙钟秒数。

        只适用于耗时 ≫ 启动开销（≳1 ms）的内核。带宽就是这种场景：
        周期计数器是每 SM 独立的，跨全部 SM 的聚合带宽没法用 clock64 表示，
        只能用墙钟；而毫秒级内核的 ~50 µs 启动抖动只占百分之几。
        """
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            kernel.launch(grid, block, ptrs, smem=smem)
            self.gpu.sync()
            best = min(best, time.perf_counter() - t0)
        return best

    # -- 时钟 ------------------------------------------------------------
    def measure_clock_ghz(self, reps: int = 5) -> float:
        from .jit import Module

        if self._clk is None:
            self._clk = Module(self.gpu, CLOCK_SRC, "clock.cu").func("clock_probe")
        buf = self.gpu.malloc(reps * 4 * 8)
        try:
            for r in range(reps):
                self._clk.launch(1, 32, (buf + r * 32, 0, 0))
            self.gpu.sync()
            vals = u64(self.gpu.dtoh(buf, reps * 4 * 8))
        finally:
            self.gpu.free(buf)
        ratios = [vals[i * 4] / vals[i * 4 + 1]
                  for i in range(reps) if vals[i * 4 + 1] > 0]
        ghz = median(ratios)
        self.clock_ghz = ghz
        return ghz
