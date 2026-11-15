"""Round 7：寄存器 bank / 操作数端口 + 多 CTA 聚合扩展性。

7.1 寄存器 bank 冲突
    难点：CUDA C++ 无法指定物理寄存器号，ptxas 自己分配。
    做法：不去"控制"分配，而是**从 SASS 里读出真实的寄存器号**，
    再把"操作数的 bank 分布"与"实测吞吐"对上。
    生成若干种源级操作数模式（全不同 / 复用 / 同一寄存器 / 间隔取用），
    每种都打印 SASS 里的实际寄存器三元组，并测 4 warp（峰值点）下的吞吐。

    判据：FFMA 是 3 源操作数指令。若寄存器堆的读端口/bank 不足，
    同 bank 组合应当掉到 64 ops/cycle（每 2 周期一条）；若仍是 ~128，
    说明这块 SM 的寄存器堆对 3 读 1 写不构成瓶颈（操作数收集器有足够端口）。

7.2 多 CTA 聚合扩展性
    本工程所有吞吐都是"每 SM"的，全局数字一律用 ×24 换算，但那个假设从未直接验证。
    做法：固定每个 CTA 的工作量，扫 grid 从 1 到 48，用墙钟测时间。
    若 24 SM，时间应在 G≤24 时保持不变、在 G=25 起翻倍。
    同时给出全卡 FP32 实测峰值（TFLOPS），与 24×128×f 对照。
"""
from __future__ import annotations

import struct

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

NACC = 8
ROUNDS = 32          # 每线程 8*32 = 256 条 FFMA
WARPS = 4            # 峰值点（Round 2 已确认 4 warp 饱和）

# 操作数模式：都是 8 条独立的 FFMA 链，只改源寄存器的"来源"
PATTERNS: dict[str, dict] = {
    "distinct": {
        "decl": "float d0,d1,d2,d3,d4,d5,d6,d7;\n"
                "    float b0,b1,b2,b3,b4,b5,b6,b7;\n"
                "    float c0,c1,c2,c3,c4,c5,c6,c7;",
        "init": "d{k} = bv + {k}; b{k} = bv + {k} * 0.5f; c{k} = cv - {k};",
        "expr": 'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d{k}) : "f"(b{k}), "f"(c{k}));',
        "sink": "d0+d1+d2+d3+d4+d5+d6+d7",
        "note": "三个源全部不同（d/b/c 各自独立）",
    },
    "reuse_src": {
        "decl": "float d0,d1,d2,d3,d4,d5,d6,d7;\n"
                "    float b0,b1,b2,b3,b4,b5,b6,b7;",
        "init": "d{k} = bv + {k}; b{k} = bv + {k} * 0.5f;",
        "expr": 'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d{k}) : "f"(b{k}), "f"(b{k}));',
        "sink": "d0+d1+d2+d3+d4+d5+d6+d7",
        "note": "两个源是同一个寄存器（b 被读两次）",
    },
    "same_all": {
        "decl": "float d0,d1,d2,d3,d4,d5,d6,d7;",
        "init": "d{k} = bv + {k};",
        "expr": 'asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(d{k}));',
        "sink": "d0+d1+d2+d3+d4+d5+d6+d7",
        "note": "三个源都是它自己（读同一个寄存器 3 次）",
    },
    "spaced2": {
        "decl": "float d0,d1,d2,d3,d4,d5,d6,d7;\n"
                "    float b0,b1,b2,b3,b4,b5,b6,b7;",
        "init": "d{k} = bv + {k}; b{k} = bv + {k} * 0.5f;",
        # b 的下标跳着取，拉开声明顺序上的间隔
        "expr": 'asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(d{k}) : "f"(b{s}));',
        "sink": "d0+d1+d2+d3+d4+d5+d6+d7",
        "note": "源按声明顺序跨 3 取，制造「不同寄存器但可能同 bank」",
    },
}


def _kernel(name: str, spec: dict) -> str:
    # b/c 必须来自运行时（显存），否则 ptxas 会把它们折成立即数，
    # 第三源就不是寄存器了（实测得到 `FFMA R7, R0, R7, 0.5`）。
    init = ("float bv = ((const float*)((const char*)in + 0))[0];\n"
            "    float cv = ((const float*)((const char*)in + 4))[0];\n    "
            + "\n    ".join(spec["init"].format(k=k) for k in range(NACC)))
    group = "\n        ".join(spec["expr"].format(k=k, s=(k * 3) % NACC) for k in range(NACC))
    body = "\n        ".join([group] * ROUNDS)
    return f"""
extern "C" __global__ void {name}(unsigned long long* cyc, const void* in, void* out) {{
    {spec["decl"]}
    {init}
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"({spec["sink"]}));
}}
"""


def _ffma_operands(mod, name: str) -> tuple[str, list[int]]:
    """取**函数中段**的一条 FFMA：开头的 FFMA 往往来自初始化代码（如 bv + k*0.5f），
    不是被测的链体（初版就取错成了 init 里的那条）。"""
    ffmas = []
    for ln in mod.sass(name).splitlines():
        s = ln.strip()
        if not s.startswith("/*") or "*/" not in s:
            continue
        txt = s.split("*/", 1)[1].split(";")[0].strip()
        if txt.startswith("FFMA"):
            ffmas.append(txt)
    if not ffmas:
        return "", []
    txt = ffmas[len(ffmas) // 2]
    regs = []
    for part in txt.split(None, 1)[1].split(","):
        p = part.strip()
        if p.startswith("R"):
            try:
                regs.append(int(p[1:].split(".")[0]))
            except ValueError:
                regs.append(-1)
        else:
            regs.append(-1)
    return txt, regs


# ---- 7.2 聚合扩展性 ------------------------------------------------------
AGG_FFMA_PER_THREAD = 200_000
AGG_BLOCK = 1024
AGG_GRID = [1, 2, 4, 8, 12, 24, 25, 48]


def _agg_kernel() -> str:
    """每线程 AGG_FFMA_PER_THREAD 条 FFMA。

    每次循环发 **64 条** FFMA（8 个累加器 × 8 轮），而不是 8 条——
    循环控制（IADD/ISETP/BRA）是 3 条指令，8 条体量时它要占 27% 的发射槽，
    会把聚合峰值测低 20% 以上。
    """
    eight = "\n        ".join(
        f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));'
        for k in range(8)
    )
    body = "\n        ".join([eight] * 8)      # 64 条
    iters = AGG_FFMA_PER_THREAD // 64
    return f"""
extern "C" __global__ void agg(unsigned long long* cyc, const void* in, void* out) {{
    float b = ((const float*)((const char*)in + 0))[0];
    float c = ((const float*)((const char*)in + 4))[0];
    float a0 = b + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3,
          a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    #pragma unroll 1
    for (int i = 0; i < {iters}; ++i) {{
    {body}
    }}
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "f"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));
}}
"""


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    sm = gpu.sm_count
    out: dict = {"clock_ghz": round(ghz, 4), "sm_count": sm}

    inbuf = gpu.malloc(64)
    gpu.htod(inbuf, struct.pack("<ff", 1.000001, 0.5) + bytes(56))
    obuf = gpu.malloc(8192)
    gpu.memset(obuf, 0, 8192)

    # ================= 7.1 寄存器 bank =================
    log("=" * 100)
    log("[Round 7.1] 寄存器 bank / 操作数端口：源寄存器模式 vs 吞吐")
    log("=" * 100)
    src = PREAMBLE + "".join(_kernel(f"p_{k}", spec) for k, spec in PATTERNS.items())
    mod = Module(gpu, src, "regbank.cu")
    bank: dict = {}
    log(f"  {'模式':<11} {'ops/cyc':>8} {'SASS（首条 FFMA）':<44} 源寄存器")
    for name, spec in PATTERNS.items():
        k = mod.func(f"p_{name}")
        cyc = bench_mod.median(b.measure1(k, WARPS * 32, (inbuf, obuf), reps))
        ops = NACC * ROUNDS * WARPS * 32
        ops_cyc = ops / cyc
        sass_txt, regs = _ffma_operands(mod, f"p_{name}")
        distinct = len({r for r in regs[1:] if r >= 0}) if len(regs) > 1 else 0
        bank[name] = {
            "note": spec["note"],
            "sass": sass_txt,
            "regs": regs,
            "distinct_src": distinct,
            "ops_per_cycle": round(ops_cyc, 1),
            "num_regs": k.num_regs,
        }
        log(f"  {name:<11} {ops_cyc:>8.1f} {sass_txt:<44} {regs[1:]}  (不同源 {distinct})")
    out["bank"] = bank

    # ================= 7.2 多 CTA 聚合 =================
    log("")
    log("=" * 100)
    log(f"[Round 7.2] 多 CTA 聚合扩展性（每 CTA {AGG_BLOCK} 线程 × "
        f"{AGG_FFMA_PER_THREAD:,} 条 FFMA）")
    log("=" * 100)
    mod2 = Module(gpu, PREAMBLE + _agg_kernel(), "agg.cu")
    ka = mod2.func("agg")
    abuf = gpu.malloc(AGG_BLOCK * 4)
    gpu.memset(abuf, 0, AGG_BLOCK * 4)
    agg: dict = {}
    try:
        ops_per_cta = AGG_BLOCK * AGG_FFMA_PER_THREAD
        t1 = None
        log(f"  {'grid':>5} {'时间 ms':>9} {'总 ops':>14} {'聚合 TFLOPS':>12} {'相对 1 CTA':>10}")
        for g in AGG_GRID:
            pts = (abuf, inbuf, abuf)
            ka.launch(g, AGG_BLOCK, pts)
            gpu.sync()
            t = b.wall_s(ka, g, AGG_BLOCK, pts, reps=3)
            total_ops = ops_per_cta * g
            tflops = total_ops * 2 / t / 1e12
            if t1 is None:
                t1 = t
            agg[str(g)] = {"ms": round(t * 1e3, 3),
                           "total_ops": total_ops,
                           "tflops": round(tflops, 2),
                           "speedup": round(t1 / t, 2)}
            log(f"  {g:>5} {t * 1e3:>9.3f} {total_ops:>14,} {tflops:>12.2f} "
                f"{t1 / t:>10.2f}×")
        peak = max(v["tflops"] for v in agg.values())
        ideal = 24 * 128 * 2 * ghz * 1e9 / 1e12
        out["agg"] = {"sweep": agg, "peak_tflops": peak,
                      "ideal_24sm_tflops": round(ideal, 2)}
        log("")
        log(f"  实测全卡峰值 {peak:.2f} TFLOPS；"
            f"24 SM × 128 lane × 2 × {ghz:.3f} GHz = {ideal:.2f} TFLOPS")
        log(f"  达成率 {peak / ideal * 100:.1f}%")
    finally:
        gpu.free(abuf)
        gpu.free(obuf)
        gpu.free(inbuf)

    return out
