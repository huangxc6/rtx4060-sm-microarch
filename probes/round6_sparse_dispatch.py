"""Round 6：稀疏 MMA + warp 调度分配规则。

6.1 稀疏 MMA（`mma.sp`）
    2:4 结构化稀疏下，A 只有一半非零。规格书上 Ada 的 2× 张量数字来自这里。
    要回答的是：稀疏指令的**指令级发射率**是否与 dense 相同？
    若相同，则同样的时间能算 2 倍的矩阵面积 ⟹ 2× 成立。

6.2 warp 分配规则（收尾 Round 4 的最后一个问号）
    Round 4 的 per-warp 耗时出现双峰，但有两种可能没分开：
      (a) 两个 warp 共享一个调度器（各 0.5 IPC）；
      (b) 后面的 warp **启动得更晚**（warp 发射有节流），它们自己那段照样是满速。
    区别办法：记**绝对起始时刻** t0。若能对齐 t0，则是 (a)；
    若后面几个 warp 的 t0 晚了上百周期，则是 (b)。
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
# 6.1 稀疏 MMA
# --------------------------------------------------------------------------
SPARSE_PTX = "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
DENSE_PTX = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
SPARSE_EQUIV_FLOP = 8192      # 等价于 dense m16n8k32 的面积
SPARSE_USEFUL_FLOP = 4096     # 实际非零元素贡献的乘法数
N_LO, N_HI = 8, 64
WARPS = [1, 2, 4, 8]


def _sparse_kernel(n: int) -> str:
    # 花括号只在 f-string 里需要写两次；这里把 PTX 文本单独拼好再插入，
    # 避免"一段 f-string、一段普通字符串"造成的双层花括号（踩过）。
    # 关键：稀疏 m16n8k32 的 **B 是 4 个 .b32**（32x8 个 half → 每线程 8 个 half），
    # 不是 dense m16n8k16 的 2 个。少给两个会得到
    # ptxas: "Argument vector size mismatch for instruction 'Sparse mma'"。
    asm_text = (f"{SPARSE_PTX} "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                "{%0,%1,%2,%3}, {%12}, 0x0;")
    one = "\n        ".join([
        f'asm volatile("{asm_text}"',
        '    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),',
        '      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));',
        f'asm volatile("{asm_text}"',
        '    : "+f"(e0), "+f"(e1), "+f"(e2), "+f"(e3)',
        '    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),',
        '      "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(meta));',
    ])
    body = "\n        ".join([one] * n)
    return f"""
extern "C" __global__ void sp_{n}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned int a0, a1, a2, a3, b0, b1, b2, b3, meta;
    float d0, d1, d2, d3, e0, e1, e2, e3;
    a0 = a1 = a2 = a3 = ((const unsigned int*)((const char*)in + 0))[0];
    b0 = b1 = b2 = b3 = ((const unsigned int*)((const char*)in + 4))[0];
    meta = ((const unsigned int*)((const char*)in + 8))[0];
    d0 = d1 = d2 = d3 = 0.f; e0 = e1 = e2 = e3 = 0.f;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"(d0 + d1 + d2 + d3 + e0 + e1 + e2 + e3));
}}
"""
    return f"""
extern "C" __global__ void sp_{n}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned int a0, a1, a2, a3, b0, b1, meta;
    float d0, d1, d2, d3, e0, e1, e2, e3;
    a0 = a1 = a2 = a3 = ((const unsigned int*)((const char*)in + 0))[0];
    b0 = b1 = ((const unsigned int*)((const char*)in + 4))[0];
    meta = ((const unsigned int*)((const char*)in + 8))[0];
    d0 = d1 = d2 = d3 = 0.f; e0 = e1 = e2 = e3 = 0.f;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"(d0 + d1 + d2 + d3 + e0 + e1 + e2 + e3));
}}
"""


# --------------------------------------------------------------------------
# 6.2 warp 分配：记绝对起始时刻
# --------------------------------------------------------------------------
NFFMA = 256


def _dispatch_kernel() -> str:
    group = "\n        ".join(
        f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));'
        for k in range(8)
    )
    body = "\n        ".join([group] * (NFFMA // 8))
    return f"""
extern "C" __global__ void dispatch(unsigned long long* cyc, const void* in, void* out) {{
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
        unsigned long long base = (unsigned long long)out
            + (unsigned long long)(threadIdx.x >> 5) * 32ull;
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base), "l"((unsigned long long)wid));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 8), "l"(t0));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 16), "l"(t1 - t0));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 24), "l"((unsigned long long)threadIdx.x));
    }}
    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + 16384 + threadIdx.x * 4), "f"(s));
}}
"""


def _count(mod, name: str, needle: str) -> int:
    n = 0
    for ln in mod.sass(name).splitlines():
        s = ln.strip()
        if s.startswith("/*") and "*/" in s:
            txt = s.split("*/", 1)[1].split(";")[0].strip()
            if needle in txt:
                n += 1
    return n


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    sm = gpu.sm_count
    out: dict = {"clock_ghz": round(ghz, 4)}

    inbuf = gpu.malloc(64)
    gpu.htod(inbuf, struct.pack("<IIIf", 0x3C003C00, 0x3C003C00, 0x3C003C00, 0.0)
             + bytes(48))
    obuf = gpu.malloc(32768)
    gpu.memset(obuf, 0, 32768)

    # ================= 6.1 稀疏 MMA =================
    log("=" * 96)
    log("[Round 6.1] 稀疏 MMA（2:4 结构化稀疏）")
    log("=" * 96)
    out["sparse"] = {}
    try:
        mod = Module(gpu, PREAMBLE + _sparse_kernel(N_LO) + _sparse_kernel(N_HI),
                     "mma_sp.cu")
        d_instr = _count(mod, f"sp_{N_HI}", "MMA") - _count(mod, f"sp_{N_LO}", "MMA")
        sweep = {}
        for w in WARPS:
            k_lo = mod.func(f"sp_{N_LO}")
            k_hi = mod.func(f"sp_{N_HI}")
            c_lo = bench_mod.median(b.measure1(k_lo, w * 32, (inbuf, obuf), reps))
            c_hi = bench_mod.median(b.measure1(k_hi, w * 32, (inbuf, obuf), reps))
            d = c_hi - c_lo
            if d <= 0:
                continue
            instr_per_cyc = d_instr * w / d
            sweep[w] = {
                "mma_per_cycle_per_sm": round(instr_per_cyc, 4),
                "equiv_flop_per_cycle": round(instr_per_cyc * SPARSE_EQUIV_FLOP, 1),
                "useful_flop_per_cycle": round(instr_per_cyc * SPARSE_USEFUL_FLOP, 1),
            }
        sass_mid = ""
        for ln in mod.sass(f"sp_{N_HI}").splitlines():
            s = ln.strip()
            if s.startswith("/*") and "*/" in s:
                txt = s.split("*/", 1)[1].split(";")[0].strip()
                if "MMA" in txt:
                    sass_mid = txt
                    break
        out["sparse"] = {"sweep": {str(k): v for k, v in sweep.items()},
                         "sass_instr_delta": d_instr, "sass": sass_mid}
        log(f"  SASS: {sass_mid}")
        log(f"  dInstr（MMA 条数差）= {d_instr}")
        for w, v in sweep.items():
            log(f"    {w}w: {v['mma_per_cycle_per_sm']:.4f} MMA/cyc/SM  "
                f"≈{v['equiv_flop_per_cycle']:.0f} 等效 FLOP/cyc  "
                f"（有用 {v['useful_flop_per_cycle']:.0f}）")
        peak = max(sweep.values(), key=lambda v: v["equiv_flop_per_cycle"]) \
            if sweep else None
        if peak:
            out["sparse"]["peak_equiv_flop_per_cycle"] = peak["equiv_flop_per_cycle"]
            out["sparse"]["peak_mma_per_cycle"] = peak["mma_per_cycle_per_sm"]
            log(f"  峰值: {peak['mma_per_cycle_per_sm']:.4f} MMA/cyc/SM, "
                f"等效 {peak['equiv_flop_per_cycle']:.0f} FLOP/cyc/SM, "
                f"{peak['equiv_flop_per_cycle'] * ghz * 1e9 * sm / 1e12:.2f} TFLOPS")
    except CompileError as exc:
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        hits = [ln for ln in lines if "error" in ln.lower() and ":" in ln]
        msg = hits[0] if hits else (lines[0] if lines else "")
        out["sparse"] = {"error": msg}
        log(f"  稀疏 MMA 编译失败: {msg}")

    # ================= 6.2 warp 分配 =================
    log("")
    log("=" * 96)
    log("[Round 6.2] warp 分配：区分「共享调度器」与「错峰启动」")
    log("=" * 96)
    mod2 = Module(gpu, PREAMBLE + _dispatch_kernel(), "dispatch.cu")
    k = mod2.func("dispatch")
    disp: dict = {}
    try:
        log(f"  {'warp':>5}  {'起始偏移':<28} {'每 warp 耗时':<28} 判读")
        for w in [1, 2, 3, 4, 5, 6, 8, 12, 16]:
            best = None
            for _ in range(7):
                k.launch(1, w * 32, (0, inbuf, obuf))
                gpu.sync()
                raw = gpu.dtoh(obuf, w * 32)
                rec = [struct.unpack_from("<QQQQ", raw, i * 32) for i in range(w)]
                # 选总跨度最小的一次
                span = max(r[1] + r[2] for r in rec) - min(r[1] for r in rec)
                if best is None or span < best[0]:
                    best = (span, rec)
            rec = best[1]
            t0s = [r[1] for r in rec]
            base = min(t0s)
            starts = [t - base for t in t0s]
            durs = [r[2] for r in rec]
            ipc = w * NFFMA / max(durs)
            max_start = max(starts)
            disp[str(w)] = {
                "start_offsets": [int(x) for x in starts],
                "durations": [int(x) for x in durs],
                "max_start_offset": int(max_start),
                "ipc": round(ipc, 3),
            }
            s_txt = " ".join(str(int(x)) for x in starts)
            d_txt = " ".join(str(int(x)) for x in durs)
            verdict = "同时启动" if max_start < 40 else f"错峰启动(最大 {int(max_start)} 周期)"
            log(f"  {w:>5}  {s_txt:<28} {d_txt:<28} {verdict}")
    finally:
        gpu.free(inbuf)
        gpu.free(obuf)
    out["dispatch"] = disp

    return out
