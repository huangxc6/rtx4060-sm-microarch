"""Round 4：warp 调度器的行为。

两个问题
--------
4.1 **6 warp 为什么只有 3 IPC？**（Round 2 里所有算子一致出现这个低谷）
    做法：让**每个 warp 自己记录起止周期**，输出 per-warp 耗时。
    * 若某几个 warp 明显更快 → warp 被不均匀地分到调度器上（例如 2/2/1/1），
      单占调度器的 warp 能跑满 1 IPC，共享的只能 0.5 IPC；这是**双峰**分布。
    * 若所有 warp 耗时基本相同且都变慢 → 每个调度器上都是 2 个 warp（2/2/2/0），
      也就是有一个调度器根本没被用上；这是**单峰**分布。

4.2 **有没有双发射？**（同一周期一个调度器发两条指令）
    做法：混合流里放 8 条 FFMA + 1 条 LDS，测**总指令发射率**（IPC）。
    * 单发射：总发射被 4 调度器 × 1 条/周期卡死 → IPC ≈ 4。
    * 双发射：FFMA 能到 4 条/周期、LDS 还能额外挤进 1 条/周期 → IPC 可到 5。
    计数一律用 SASS 指令条数的差值，不数源码。
"""
from __future__ import annotations

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

NFFMA = 256                      # 每个线程的 FFMA 条数（每 warp 独立计时）
WARPS = [1, 2, 3, 4, 5, 6, 8, 12, 16]

MIX_OUT_LO, MIX_OUT_HI = 8, 64   # 混合流的"组"数；每组 = 8 FFMA + 1 LDS + 1 LOP
MIX_WARPS = 4


def _warp_timing_kernel() -> str:
    """每个 warp 跑 256 条独立 FFMA，各自记录本 warp 的耗时。"""
    group = "\n        ".join(
        f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));'
        for k in range(8)
    )
    body = "\n        ".join([group] * (NFFMA // 8))
    return f"""
extern "C" __global__ void warp_timing(unsigned long long* cyc, const void* in, void* out) {{
    float b = ((const float*)((const char*)in + 0))[0];
    float c = ((const float*)((const char*)in + 4))[0];
    float a0 = b + threadIdx.x, a1 = b + threadIdx.x + 1, a2 = b + threadIdx.x + 2,
          a3 = b + threadIdx.x + 3, a4 = b + threadIdx.x + 4, a5 = b + threadIdx.x + 5,
          a6 = b + threadIdx.x + 6, a7 = b + threadIdx.x + 7;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    unsigned int wid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(wid));
    if ((threadIdx.x & 31u) == 0) {{
        unsigned long long base = (unsigned long long)out + (unsigned long long)(threadIdx.x >> 5) * 32ull;
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base), "l"((unsigned long long)wid));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 8), "l"(t1 - t0));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 16), "l"((unsigned long long)threadIdx.x));
    }}
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + 16384 + threadIdx.x * 4), "f"(s));
}}
"""


def _mix_issue_kernel(groups: int) -> str:
    """每组 = 8 条 FFMA + 1 条 LDS（+1 条 LOP 用来消费载入值，防止被删）。"""
    one = []
    for k in range(8):
        one.append(f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));')
    one.append('asm volatile("ld.shared.b32 %0, [%1];" : "=r"(t) : "r"(addr));')
    one.append('asm volatile("xor.b32 %0, %0, %1;" : "+r"(v) : "r"(t));')
    group = "\n        ".join(one)
    body = "\n        ".join([group] * groups)
    return f"""
extern "C" __global__ void mix_issue_{groups}(unsigned long long* cyc, const void* in,
                                              void* out) {{
    __shared__ unsigned int sm[4096];
    sm[threadIdx.x] = threadIdx.x;
    asm volatile("bar.sync 0;");
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned int addr = (unsigned int)sbase + ((threadIdx.x & 1023u) * 4u);
    float b = ((const float*)((const char*)in + 0))[0];
    float c = ((const float*)((const char*)in + 4))[0];
    float a0 = b + threadIdx.x, a1 = b + threadIdx.x + 1, a2 = b + threadIdx.x + 2,
          a3 = b + threadIdx.x + 3, a4 = b + threadIdx.x + 4, a5 = b + threadIdx.x + 5,
          a6 = b + threadIdx.x + 6, a7 = b + threadIdx.x + 7;
    unsigned int v = 0, t = 0;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(s));
    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + 8192 + threadIdx.x * 4), "r"(v));
}}
"""


def _sass_count(mod, name: str) -> int:
    return sum(1 for ln in mod.sass(name).splitlines()
               if ln.strip().startswith("/*") and "*/" in ln)


def run(log=print, reps: int = 7) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    # ================= 4.1 每 warp 计时 =================
    log("=" * 96)
    log("[Round 4.1] 每 warp 独立计时：6 warp 为什么只有 3 IPC？")
    log("=" * 96)
    src = PREAMBLE + _warp_timing_kernel()
    mod = Module(gpu, src, "warp_timing.cu")
    k = mod.func("warp_timing")
    inbuf = gpu.malloc(64)
    import struct
    gpu.htod(inbuf, struct.pack("<ff", 1.000001, 0.5) + bytes(56))
    obuf = gpu.malloc(32768)
    gpu.memset(obuf, 0, 32768)
    timing: dict = {}
    try:
        log(f"  {'warp':>5} {'每 warp 耗时（周期）':<40} {'IPC':>6}  判读")
        for w in WARPS:
            recs = []
            for _ in range(reps):
                k.launch(1, w * 32, (0, inbuf, obuf))
                gpu.sync()
                raw = gpu.dtoh(obuf, w * 32)
                vals = [struct.unpack_from("<Q", raw, i * 32 + 8)[0] for i in range(w)]
                recs.append(vals)
            best = min(recs, key=lambda v: max(v))
            cyc_max = max(best)
            ipc = w * NFFMA / cyc_max if cyc_max else float("nan")
            lo, hi = min(best), max(best)
            spread = hi / lo if lo else float("nan")
            timing[str(w)] = {
                "per_warp_cycles": [int(x) for x in best],
                "ipc": round(ipc, 3),
                "max_cycles": int(cyc_max),
                "spread": round(spread, 3),
            }
            shown = " ".join(str(int(x)) for x in best)
            verdict = "均匀" if spread < 1.15 else f"双峰(最快/最慢={spread:.2f})"
            log(f"  {w:>5} {shown:<40} {ipc:>6.2f}  {verdict}")
    finally:
        gpu.free(inbuf)
        gpu.free(obuf)
    out["warp_timing"] = timing

    # 用 per-warp 数据把 6 warp 的分布单独说清楚
    if "6" in timing:
        cy = sorted(timing["6"]["per_warp_cycles"])
        log("")
        log(f"  6 warp 的耗时排序: {cy}")
        log(f"    → 若分成两簇（≈一半快一半慢）说明 2/2/1/1；若一簇且都慢说明 2/2/2/0")
        log(f"    → 实测 spread = {timing['6']['spread']}")

    # ================= 4.2 发射宽度 =================
    log("")
    log("=" * 96)
    log(f"[Round 4.2] 发射宽度：8 FFMA + 1 LDS + 1 LOP 一组，{MIX_WARPS} warp")
    log("=" * 96)
    src = PREAMBLE + "".join(_mix_issue_kernel(g) for g in (MIX_OUT_LO, MIX_OUT_HI))
    mod2 = Module(gpu, src, "mix_issue.cu")
    k_lo = mod2.func(f"mix_issue_{MIX_OUT_LO}")
    k_hi = mod2.func(f"mix_issue_{MIX_OUT_HI}")
    d_sass = _sass_count(mod2, f"mix_issue_{MIX_OUT_HI}") - _sass_count(mod2, f"mix_issue_{MIX_OUT_LO}")
    inbuf2 = gpu.malloc(64)
    gpu.htod(inbuf2, struct.pack("<ff", 1.000001, 0.5) + bytes(56))
    obuf2 = gpu.malloc(16384)
    gpu.memset(obuf2, 0, 16384)
    try:
        c_lo = min(b.measure1(k_lo, MIX_WARPS * 32, (inbuf2, obuf2), reps))
        c_hi = min(b.measure1(k_hi, MIX_WARPS * 32, (inbuf2, obuf2), reps))
    finally:
        gpu.free(inbuf2)
        gpu.free(obuf2)
    d_cyc = c_hi - c_lo
    # 总发射的 warp 指令数 = 每线程新增指令数 × warp 数
    ipc = d_sass * MIX_WARPS / d_cyc if d_cyc else float("nan")
    log(f"  dSASS（每线程新增指令数）= {d_sass}")
    log(f"  dCyc = {d_cyc}")
    log(f"  总发射率 = {ipc:.2f} 条 warp 指令/周期/SM")
    log(f"  判读: 4 调度器 × 单发射 → 上限 4.0；若出现 >4.5 则说明存在双发射")
    out["issue"] = {"d_sass": d_sass, "d_cycles": int(d_cyc),
                    "ipc": round(ipc, 3), "warps": MIX_WARPS}

    return out
