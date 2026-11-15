"""Round 10：桌面干扰的量化，以及"多 SM 取最小"的降噪方法。

背景
----
Round 3 的 L2 区间（256 KiB–8 MiB）在不同运行之间差到 35%，怀疑是桌面合成抢占 L2。
这台机器无法独占 GPU（i5-13490F 无核显，4060 同时驱动两块屏），所以只能靠方法降噪。

本轮的思路
----------
1. **量化噪声**：固定测同一个尺寸（1 MiB），重复 40 次，看分布（min/median/max）。
   这直接给出"桌面到底让我们偏了多少"。
2. **多 SM 取最小**：24 个 CTA（每 SM 一个）各自在**同一块**工作集上跑一条独立链，
   取**跨 SM 的最小值**——被干扰最少的那个 SM 的最干净读数。
   总 footprint 不变（所有 CTA 共读同一区域），所以缓存行为与单 SM 版本可比。
"""
from __future__ import annotations

import struct
import time

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

CHASE_K = [16, 18, 20, 21, 22, 23]      # 64 KiB, 256 KiB, 1 MiB, 2 MiB, 4 MiB, 8 MiB
ITERS_FACTOR = 4                        # 跳数 = 4 × line 数
NOISE_K = 20                            # 噪声量化用 1 MiB
NOISE_REPS = 40


def _mix_params(k: int):
    nodes = 1 << (k - 3)
    mask = nodes - 1
    a = max(3, int(nodes * 0.6180339887)) | 1
    shift = max(1, (k - 3) // 2)
    return nodes, mask, a, shift


def _iters_for(k: int) -> tuple[int, int]:
    lines = max(16, (1 << k) // 128)
    hi = max(1024, ITERS_FACTOR * lines)
    return hi // 4, hi


C = 0x9E3779B1


def _init_kernel(k: int) -> str:
    nodes, mask, a, shift = _mix_params(k)
    return f"""
extern "C" __global__ void ch_init_{k}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned long long* p = (unsigned long long*)in;
    unsigned int nthr = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < {nodes}u; i += nthr) {{
        unsigned int t0 = (i * {C}u) & {mask}u;
        unsigned int g0 = t0 ^ (t0 >> {shift});
        unsigned int t1 = (((i + {a}u) & {mask}u) * {C}u) & {mask}u;
        unsigned int g1 = t1 ^ (t1 >> {shift});
        p[g0] = base + 8ull * (unsigned long long)g1;
    }}
}}
"""


def _chase_kernel(k: int, iters: int, multi: bool) -> str:
    """multi=True 时每个 CTA 从不同节点起跑（用于跨 SM 取最小）。"""
    nodes, mask, a, shift = _mix_params(k)
    name = f"ch_{k}_{iters}_{'m' if multi else 's'}"
    if multi:
        start = (f"unsigned int sn = (blockIdx.x * {max(1, nodes // 32)}u) & {mask}u;\n"
                 f"    unsigned int st = (sn * {C}u) & {mask}u;\n"
                 f"    unsigned int sg = st ^ (st >> {shift});\n"
                 f"    unsigned long long addr = base + 8ull * (unsigned long long)sg;")
        cyc_store = "cyc[blockIdx.x] = t1 - t0;"
    else:
        start = "unsigned long long addr = base;"
        cyc_store = "cyc[blockIdx.x] = t1 - t0;"
    return f"""
extern "C" __global__ void {name}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    {start}
    unsigned long long t0, t1;
    CLK(t0);
    #pragma unroll 1
    for (int i = 0; i < {iters}; ++i) {{
        asm volatile("ld.global.ca.b64 %0, [%0];" : "+l"(addr));
    }}
    CLK(t1);
    if (threadIdx.x == 0) {cyc_store}
    asm volatile("st.global.b64 [%0], %1;" :: "l"(out), "l"(addr));
}}
"""


def _util() -> int:
    """读一次 GPU 利用率（采样桌面负载），失败就返回 -1。"""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    out: dict = {"clock_ghz": round(ghz, 4), "sm_count": gpu.sm_count}

    src = PREAMBLE + "".join(_init_kernel(k) for k in CHASE_K)
    src += "".join(_chase_kernel(k, it, False) for k in CHASE_K for it in _iters_for(k))
    src += "".join(_chase_kernel(k, it, True) for k in CHASE_K for it in _iters_for(k))
    mod = Module(gpu, src, "round10.cu")

    buf = gpu.malloc(1 << 23)
    sink = gpu.malloc(8)
    gpu.memset(buf, 0, 1 << 23)
    try:
        # ---------- 1. 噪声量化 ----------
        log("=" * 92)
        log(f"[Round 10.1] 噪声量化：1 MiB 工作集，单 warp，重复 {NOISE_REPS} 次")
        log("=" * 92)
        lo_it, hi_it = _iters_for(NOISE_K)
        nodes, mask, a, shift = _mix_params(NOISE_K)
        mod.func(f"ch_init_{NOISE_K}").launch(256, 256, (0, buf, 0))
        gpu.sync()
        k_lo = mod.func(f"ch_{NOISE_K}_{lo_it}_s")
        k_hi = mod.func(f"ch_{NOISE_K}_{hi_it}_s")
        vals = []
        utils = []
        t_start = time.time()
        for _ in range(NOISE_REPS):
            c_lo = min(b.measure1(k_lo, 32, (buf, sink), 1))
            c_hi = min(b.measure1(k_hi, 32, (buf, sink), 1))
            vals.append((c_hi - c_lo) / float(hi_it - lo_it))
            utils.append(_util())
        elapsed = time.time() - t_start
        sv = sorted(vals)
        out["noise"] = {
            "n": len(vals),
            "min": round(sv[0], 1),
            "p25": round(sv[len(sv) // 4], 1),
            "median": round(sv[len(sv) // 2], 1),
            "p75": round(sv[3 * len(sv) // 4], 1),
            "max": round(sv[-1], 1),
            "spread_max_over_min": round(sv[-1] / sv[0], 2),
            "gpu_util_samples": utils,
            "gpu_util_median": sorted(utils)[len(utils) // 2],
            "elapsed_s": round(elapsed, 1),
        }
        log(f"  min={sv[0]:.0f}  p25={sv[len(sv)//4]:.0f}  median={sv[len(sv)//2]:.0f}  "
            f"p75={sv[3*len(sv)//4]:.0f}  max={sv[-1]:.0f} 周期")
        log(f"  max/min = {sv[-1]/sv[0]:.2f}×   （{elapsed:.0f} 秒内 {len(vals)} 次独立测量）")
        log(f"  期间的 GPU 利用率采样: 中位数 {sorted(utils)[len(utils)//2]}%")
        log(f"  直方图（10 个桶）:")
        lo, hi = sv[0], sv[-1]
        if hi > lo:
            buckets = [0] * 10
            for v in sv:
                buckets[min(9, int((v - lo) / (hi - lo + 1e-9) * 10))] += 1
            for i, n in enumerate(buckets):
                edge = lo + (hi - lo) * i / 10
                log(f"    {edge:>7.0f} | {'#' * n} {n}")

        # ---------- 2. 多 SM 取最小 ----------
        log("")
        log("=" * 92)
        log("[Round 10.2] 多 SM 取最小：24 个 CTA 各跑一条独立链，取跨 SM 最小值")
        log("=" * 92)
        log(f"  {'工作集':>10} {'单 SM':>9} {'多SM min':>9} {'多SM 中位':>10} {'多SM max':>9} {'改善':>7}")
        multi: dict = {}
        for k in CHASE_K:
            nodes, mask, a, shift = _mix_params(k)
            lo_it, hi_it = _iters_for(k)
            grid = min(24, max(1, nodes // 256))
            mod.func(f"ch_init_{k}").launch(min(512, max(1, nodes // 256)), 256, (0, buf, 0))
            gpu.sync()
            # 单 SM
            c_lo = min(b.measure1(mod.func(f"ch_{k}_{lo_it}_s"), 32, (buf, sink), reps))
            c_hi = min(b.measure1(mod.func(f"ch_{k}_{hi_it}_s"), 32, (buf, sink), reps))
            single = (c_hi - c_lo) / float(hi_it - lo_it)
            # 多 SM：取每个 CTA 的 delta，再跨 CTA 统计
            r_lo = b.measure(mod.func(f"ch_{k}_{lo_it}_m"), grid, 32, (buf, sink), reps)
            r_hi = b.measure(mod.func(f"ch_{k}_{hi_it}_m"), grid, 32, (buf, sink), reps)
            per_cta = []
            for c in range(grid):
                lo_c = min(row[c] for row in r_lo)
                hi_c = min(row[c] for row in r_hi)
                per_cta.append((hi_c - lo_c) / float(hi_it - lo_it))
            per_cta.sort()
            med = per_cta[len(per_cta) // 2]
            multi[str(k)] = {
                "footprint_bytes": 1 << k,
                "single_cycles": round(single, 1),
                "multi_min": round(per_cta[0], 1),
                "multi_median": round(med, 1),
                "multi_max": round(per_cta[-1], 1),
                "n_cta": grid,
            }
            log(f"  {1 << k:>10,} {single:>9.1f} {per_cta[0]:>9.1f} {med:>10.1f} "
                f"{per_cta[-1]:>9.1f} {single / per_cta[0]:>6.2f}×")
        out["multi_sm"] = multi
    finally:
        gpu.free(buf)
        gpu.free(sink)
    return out
