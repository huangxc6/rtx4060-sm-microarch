"""Round 9：INT8/INT4 张量 + TLB。

9.1 INT8 / INT4 MMA
    补全张量精度谱：`mma.sync` 的 s8（m16n8k32）与 s4（m16n8k64）。
    与 Round 5 的 f16/tf32/fp8 对照，看"每条指令覆盖的 K 翻倍"是否仍严格成立。

9.2 TLB
    难点：无法直接读 TLB 命中率，也无法指定页大小。
    办法：**只访问每条 2 MiB 页里的一个 cache line**，构造一个"页数可选、但
    字节工作集极小（P × 128 B）"的指针追逐链。
      * 字节工作集小 ⟹ 数据常驻 L2（延迟本来该是 ~230 周期）；
      * 但每次访问都换一页 ⟹ 页表压力被单独放大。
    P 从 4 扫到 256：若 TLB 容量不够，延迟会明显上台阶。
"""
from __future__ import annotations

import struct

from harness import bench as bench_mod
from harness import env
from harness.jit import Module
from harness.cuda import CompileError

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

# --------------------------------------------------------------------------
# 9.1 INT8 / INT4 张量
# --------------------------------------------------------------------------
MMA_OPS = {
    "int8": {
        "ptx": "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32",
        "flop": 8192,      # 2*16*8*32
        "note": "INT8 s8 (m16n8k32)",
    },
    "int4": {
        "ptx": "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32",
        "flop": 16384,     # 2*16*8*64
        "note": "INT4 s4 (m16n8k64)",
    },
}
WARPS = [1, 2, 4, 8]
N_LO, N_HI = 8, 64


def _mma_kernel(op: str, n: int) -> str:
    ptx = MMA_OPS[op]["ptx"]
    one = "\n        ".join([
        f'asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"',
        '    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));',
        f'asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"',
        '    : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));',
    ])
    body = "\n        ".join([one] * n)
    return f"""
extern "C" __global__ void tp_{op}_{n}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned int a0, a1, a2, a3, b0, b1;
    int d0, d1, d2, d3, e0, e1, e2, e3;
    a0 = a1 = a2 = a3 = ((const unsigned int*)((const char*)in + 0))[0];
    b0 = b1 = ((const unsigned int*)((const char*)in + 4))[0];
    d0 = d1 = d2 = d3 = 0; e0 = e1 = e2 = e3 = 0;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.s32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "r"(d0 + d1 + d2 + d3 + e0 + e1 + e2 + e3));
}}
"""


def _mma_count(mod, name: str) -> int:
    n = 0
    for ln in mod.sass(name).splitlines():
        s = ln.strip()
        if s.startswith("/*") and "*/" in s:
            txt = s.split("*/", 1)[1].split(";")[0].strip()
            if "MMA" in txt:
                n += 1
    return n


# --------------------------------------------------------------------------
# 9.2 TLB
# --------------------------------------------------------------------------
PAGE = 1 << 21                    # 2 MiB
P_LIST = [4, 8, 16, 32, 64, 128, 256]
TLB_ITERS_LO, TLB_ITERS_HI = 512, 2048


def _tlb_init_kernel(p: int) -> str:
    """链节点 i 放在 base + i*2MiB，值为节点 (i+A) mod P 的地址。

    只写每页的第一个 8 字节，所以**字节工作集只有 P × 128 B**（常驻 L2），
    但每次追逐步都换一页。
    """
    a = max(1, int(p * 0.6180339887)) | 1
    return f"""
extern "C" __global__ void tlb_init_{p}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {p}u) {{
        unsigned int nxt = (i + {a}u) % {p}u;
        ((unsigned long long*)in)[i * {PAGE // 8}u] = base + (unsigned long long)nxt * {PAGE}u;
    }}
}}
"""


def _tlb_chase_kernel(p: int, iters: int) -> str:
    return f"""
extern "C" __global__ void tlb_{p}_{iters}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long addr = (unsigned long long)in;
    unsigned long long t0, t1;
    CLK(t0);
    #pragma unroll 1
    for (int i = 0; i < {iters}; ++i) {{
        asm volatile("ld.global.ca.b64 %0, [%0];" : "+l"(addr));
    }}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.b64 [%0], %1;" :: "l"(out), "l"(addr));
}}
"""


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    sm = gpu.sm_count
    out: dict = {"clock_ghz": round(ghz, 4)}

    inbuf = gpu.malloc(64)
    gpu.htod(inbuf, struct.pack("<IIII", 0x01010101, 0x01010101, 0x01010101, 0x01010101)
             + bytes(48))
    obuf = gpu.malloc(4096)
    gpu.memset(obuf, 0, 4096)

    # ================= 9.1 INT8/INT4 =================
    log("=" * 96)
    log("[Round 9.1] INT8 / INT4 张量")
    log("=" * 96)
    tens: dict = {}
    for op, spec in MMA_OPS.items():
        try:
            mod = Module(gpu, PREAMBLE + _mma_kernel(op, N_LO) + _mma_kernel(op, N_HI),
                         f"mma_{op}.cu")
        except CompileError as exc:
            lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
            hits = [ln for ln in lines if "error" in ln.lower() and "ptxas" in ln.lower()]
            msg = hits[0] if hits else (lines[-1] if lines else "")
            tens[op] = {"error": msg}
            log(f"  {op}: 编译失败 {msg[:100]}")
            continue
        d_instr = _mma_count(mod, f"tp_{op}_{N_HI}") - _mma_count(mod, f"tp_{op}_{N_LO}")
        sweep = {}
        for w in WARPS:
            k_lo = mod.func(f"tp_{op}_{N_LO}")
            k_hi = mod.func(f"tp_{op}_{N_HI}")
            c_lo = bench_mod.median(b.measure1(k_lo, w * 32, (inbuf, obuf), reps))
            c_hi = bench_mod.median(b.measure1(k_hi, w * 32, (inbuf, obuf), reps))
            d = c_hi - c_lo
            if d <= 0:
                continue
            ir = d_instr * w / d
            sweep[w] = {"mma_per_cycle": round(ir, 4),
                        "flop_per_cycle": round(ir * spec["flop"], 1)}
        sass_mid = ""
        for ln in mod.sass(f"tp_{op}_{N_HI}").splitlines():
            s = ln.strip()
            if s.startswith("/*") and "*/" in s:
                txt = s.split("*/", 1)[1].split(";")[0].strip()
                if "MMA" in txt:
                    sass_mid = txt
                    break
        peak = max(sweep.values(), key=lambda v: v["flop_per_cycle"]) if sweep else None
        tens[op] = {"note": spec["note"], "flop_per_mma": spec["flop"],
                    "sass": sass_mid, "d_instr": d_instr,
                    "sweep": {str(k): v for k, v in sweep.items()},
                    "peak_flop_per_cycle": peak["flop_per_cycle"] if peak else None,
                    "peak_mma_per_cycle": peak["mma_per_cycle"] if peak else None}
        log(f"  {op:<5} {sass_mid}")
        log(f"        dInstr={d_instr}  峰值 {peak['mma_per_cycle']:.4f} MMA/cyc/SM, "
            f"{peak['flop_per_cycle']:.0f} FLOP/cyc/SM, "
            f"{peak['flop_per_cycle'] * ghz * 1e9 * sm / 1e12:.2f} TFLOPS")
        steps = " ".join(f"{w}w:{v['mma_per_cycle']:.4f}" for w, v in sweep.items())
        log(f"        [{steps}]")
    out["tensor"] = tens

    # ================= 9.2 TLB =================
    log("")
    log("=" * 96)
    log(f"[Round 9.2] TLB：只碰每条 2 MiB 页的一个 cache line")
    log("=" * 96)
    src = PREAMBLE + "".join(_tlb_init_kernel(p) for p in P_LIST) \
        + "".join(_tlb_chase_kernel(p, it) for p in P_LIST
                  for it in (TLB_ITERS_LO, TLB_ITERS_HI))
    mod2 = Module(gpu, src, "tlb.cu")
    buf = gpu.malloc(PAGE * (max(P_LIST) + 1))
    sink = gpu.malloc(8)
    gpu.memset(buf, 0, PAGE * (max(P_LIST) + 1))
    tlb: dict = {}
    try:
        log(f"  {'页数':>6} {'字节工作集':>12} {'跳数':>8} {'延迟(周期)':>12}")
        for p in P_LIST:
            mod2.func(f"tlb_init_{p}").launch(1, 256, (0, buf, 0))
            gpu.sync()
            k_lo = mod2.func(f"tlb_{p}_{TLB_ITERS_LO}")
            k_hi = mod2.func(f"tlb_{p}_{TLB_ITERS_HI}")
            c_lo = min(b.measure1(k_lo, 32, (buf, sink), reps))
            c_hi = min(b.measure1(k_hi, 32, (buf, sink), reps))
            lat = (c_hi - c_lo) / float(TLB_ITERS_HI - TLB_ITERS_LO)
            tlb[str(p)] = {"pages": p, "bytes_touched": p * 128,
                           "latency_cycles": round(lat, 2)}
            log(f"  {p:>6} {p * 128:>12,} {TLB_ITERS_HI:>8} {lat:>12.2f}")
    finally:
        gpu.free(buf)
        gpu.free(sink)
    out["tlb"] = tlb

    gpu.free(obuf)
    gpu.free(inbuf)
    return out
