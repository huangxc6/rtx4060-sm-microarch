"""Round 5：第 4 代 Tensor Core。

测什么
------
* `mma.sync` 三种精度的吞吐：f16 (m16n8k16)、tf32 (m16n8k8)、fp8 (m16n8k32)
* f16 MMA 的依赖链延迟
* `wgmma`（Hopper 才有）在 sm_89 上是否被拒绝——一个干净的负向验证

为什么用寄存器操作数
--------------------
MMA 的 A/B 都可以直接从寄存器取，于是**完全不碰显存**，
测到的就是 Tensor Core 本身的吞吐/延迟，不受 LSU 与缓存干扰。

计数方式
--------
一条 MMA 是 **warp 级指令**：一个 warp 执行一条 m16n8k16 = 2·16·8·16 = **4096 FLOP**
（整条 warp 合计，不是每线程）。所以::

    FLOP/cycle/SM = warps * mma_count * 4096 / cycles

与 Round 2 一样按 SASS 指令条数归一，避免被编译器化简骗。
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

# ptx 指令模板、每条 MMA 的 FLOP、说明
MMA_OPS: dict[str, dict] = {
    "f16": {
        "ptx": "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        "flop": 4096,
        "note": "FP16 (m16n8k16)",
    },
    "tf32": {
        "ptx": "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        "flop": 2048,
        "note": "TF32 (m16n8k8)",
    },
    "fp8": {
        "ptx": "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
        "flop": 8192,
        "note": "FP8 e4m3 (m16n8k32)",
    },
}

WARPS = [1, 2, 4, 8]
N_LO, N_HI = 8, 64          # 每线程（每 warp）的 MMA 条数


def _mma_asm(ptx: str, dset: str, aset: str = "a") -> str:
    d = ", ".join(f"%{{{4 + i}}}" for i in range(4))
    return (f'asm volatile("{ptx} {{{d}}}, '
            f'{{%4,%5,%6,%7}}, {{%8,%9}}, {{{d}}};"')


def _decl() -> tuple[str, str]:
    decl = ("unsigned int a0, a1, a2, a3, b0, b1;\n"
            "    float d0, d1, d2, d3, e0, e1, e2, e3;")
    init = ("a0 = a1 = a2 = a3 = ((const unsigned int*)((const char*)in + 0))[0];\n"
            "    b0 = b1 = ((const unsigned int*)((const char*)in + 4))[0];\n"
            "    d0 = d1 = d2 = d3 = 0.f; e0 = e1 = e2 = e3 = 0.f;")
    return decl, init


def _tp_kernel(op: str, n: int) -> str:
    ptx = MMA_OPS[op]["ptx"]
    one = "\n        ".join([
        f'asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"',
        '    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));',
        f'asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"',
        '    : "+f"(e0), "+f"(e1), "+f"(e2), "+f"(e3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));',
    ])
    body = "\n        ".join([one] * n)
    decl, init = _decl()
    return f"""
extern "C" __global__ void tp_{op}_{n}(unsigned long long* cyc, const void* in, void* out) {{
    {decl}
    {init}
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"(d0 + d1 + d2 + d3 + e0 + e1 + e2 + e3));
}}
"""


def _lat_kernel(op: str, n: int) -> str:
    """单累加器依赖链：D 同时是下一条的 C，于是每条都等前一条。"""
    ptx = MMA_OPS[op]["ptx"]
    one = "\n        ".join([
        f'asm volatile("{ptx} {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};"',
        '    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));',
    ])
    body = "\n        ".join([one] * n)
    decl, init = _decl()
    return f"""
extern "C" __global__ void lat_{op}_{n}(unsigned long long* cyc, const void* in, void* out) {{
    {decl}
    {init}
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"(d0 + d1 + d2 + d3));
}}
"""


WGMMA_SRC = r"""
extern "C" __global__ void wgmma_test(unsigned long long* cyc, const void* in, void* out) {
    if (threadIdx.x == 0xFFFFFFFFu) {
        asm volatile("wgmma.fence.sync.aligned;");
    }
}
"""


def _sass_count(mod, name: str) -> int:
    return sum(1 for ln in mod.sass(name).splitlines()
               if ln.strip().startswith("/*") and "*/" in ln)


def _mma_count(mod, name: str) -> int:
    """只数 MMA 指令（HMMA/QMMA/...）条数。

    这是本探针的关键修正：函数里除了 MMA 还有 MOV/LDG 等杂项，且**杂项条数也随
    迭代数增长**（实测 f16 从 n=8 到 n=64：MMA 增 112 条，杂项增 56 条）。
    若按"所有指令"归一，FLOP 会被高估 1.5 倍（768 vs 真实的 512）。
    """
    n = 0
    for ln in mod.sass(name).splitlines():
        s = ln.strip()
        if s.startswith("/*") and "*/" in s:
            txt = s.split("*/", 1)[1].split(";")[0].strip()
            if "MMA" in txt:
                n += 1
    return n


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    sm = gpu.sm_count
    out: dict = {"clock_ghz": round(ghz, 4), "sm_count": sm}

    inbuf = gpu.malloc(64)
    # 0x3C003C00 = 两个 half 都是 1.0；tf32/fp8 下也是良性值
    gpu.htod(inbuf, struct.pack("<II", 0x3C003C00, 0x3C003C00) + bytes(56))
    obuf = gpu.malloc(4096)
    gpu.memset(obuf, 0, 4096)

    log("=" * 96)
    log("[Round 5] 第 4 代 Tensor Core")
    log("=" * 96)

    results: dict = {}
    try:
        for op, spec in MMA_OPS.items():
            flop = spec["flop"]
            try:
                src = PREAMBLE + _tp_kernel(op, N_LO) + _tp_kernel(op, N_HI) \
                    + _lat_kernel(op, 32)
                mod = Module(gpu, src, f"mma_{op}.cu")
            except CompileError as exc:
                msg = str(exc).split("\n")[0]
                log(f"  {op:<5} 编译失败（该精度在 sm_89 上不支持？）: {msg}")
                results[op] = {"error": msg}
                continue

            d_sass = _mma_count(mod, f"tp_{op}_{N_HI}") - _mma_count(mod, f"tp_{op}_{N_LO}")
            sweep = {}
            for w in WARPS:
                k_lo = mod.func(f"tp_{op}_{N_LO}")
                k_hi = mod.func(f"tp_{op}_{N_HI}")
                c_lo = bench_mod.median(b.measure1(k_lo, w * 32, (inbuf, obuf), reps))
                c_hi = bench_mod.median(b.measure1(k_hi, w * 32, (inbuf, obuf), reps))
                d = c_hi - c_lo
                if d <= 0:
                    continue
                # 每线程新增 d_sass 条 MMA；每条 MMA 是 warp 级、4096 FLOP（按 flop 参数折算）
                flops = d_sass * w / d * flop
                sweep[w] = round(flops, 1)
            peak_w = max(sweep, key=lambda w: sweep[w]) if sweep else 0
            peak = sweep.get(peak_w, float("nan"))

            # 延迟
            kl = mod.func(f"lat_{op}_32")
            c_lo = bench_mod.median(b.measure1(mod.func(f"lat_{op}_32"), 32,
                                                (inbuf, obuf), reps))
            lat = c_lo / 32.0
            sass_mid = ""
            for ln in mod.sass(f"tp_{op}_{N_HI}").splitlines():
                s = ln.strip()
                if s.startswith("/*") and "*/" in s:
                    txt = s.split("*/", 1)[1].split(";")[0].strip()
                    if "MMA" in txt or "HMMA" in txt:
                        sass_mid = txt
                        break

            tflops = peak * ghz * 1e9 * sm / 1e12
            results[op] = {
                "note": spec["note"],
                "flop_per_mma": flop,
                "sass_instr_per_round": d_sass,
                "flop_per_cycle_per_sm": peak,
                "sweep": {str(k): v for k, v in sweep.items()},
                "tflops_at_measured_clock": round(tflops, 2),
                "mma_latency_cycles": round(lat, 2),
                "sass": sass_mid,
            }
            steps = " ".join(f"{w}w:{sweep[w]:.0f}" for w in sweep)
            log(f"  {op:<5} {peak:>8.0f} FLOP/cyc/SM   {tflops:>7.2f} TFLOPS   "
                f"延迟 {lat:>5.1f} 周期   SASS={sass_mid}")
            log(f"        [{steps}]")
    finally:
        gpu.free(inbuf)
        gpu.free(obuf)

    # ---- wgmma 负向验证 ------------------------------------------------
    log("")
    try:
        Module(gpu, WGMMA_SRC, "wgmma.cu")
        out["wgmma_supported"] = True
        log("  wgmma: **编译通过**（意外——sm_89 本不该支持 Hopper 的 wgmma）")
    except CompileError as exc:
        msg = str(exc)
        # 取 nvrtc 真正的那行诊断（含 "error:"），而不是异常头
        # 优先取 ptxas 的诊断行（异常头那行只是 nvrtc 的返回码名，没有信息量）
        lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
        hits = [ln for ln in lines if "ptxas" in ln and "error" in ln.lower()]
        if not hits:
            hits = [ln for ln in lines if "not supported" in ln]
        detail = hits[0] if hits else (msg.splitlines()[0].strip() if msg else "")
        out["wgmma_supported"] = False
        out["wgmma_error"] = detail
        log(f"  wgmma: 编译被拒绝（符合预期）: {detail}")

    out["mma"] = results
    return out
