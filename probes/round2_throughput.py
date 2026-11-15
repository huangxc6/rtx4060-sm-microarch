"""Round 2：指令吞吐与发射宽度（ops/cycle/SM）。

方法
----
构造**完全独立**的指令流（8 个累加器 × R 轮，全部展开，不留循环——
循环的 IADD/ISETP/BRA 会白占发射槽）。一个 CTA 只落在一个 SM 上，于是::

    ops_per_cycle_per_SM = dSASS_per_thread * blockDim / dCycles

两个关键设计
------------
1. **用 SASS 条数归一，而不是源码条数**。ptxas 会做代数化简：
   ``a += b`` 重复 32 次变成 ``a + 32*b``、``a ^= b`` 重复两次就抵消。
   若不归一，``lop3`` 会测出 52428 ops/cyc 这种荒谬值（实测踩过）。
   做法：同一算子生成 R=16 与 R=32 两个内核，两者指令数之差即"每线程真实
   执行的指令数"，周期数之差即执行它们花的周期，比值才是单位吞吐。
2. **算子必须选不可化简的形式**：
   * FP32/FP16/INT32 乘加：``a = a*b + c`` 非线性，天然安全；
   * 加法：改为环状 ``a_k += a_{(k+1)}``（指数增长，无法折叠）；
   * 逻辑：改为环状 xor（自逆，单独重复必抵消）；
   * 移位：改为 ``shf.l.wrap``（循环移位，不归零）；
   * FP64：改为 ``fma.rn.f64``（加法是线性的，会被折成 a+32b）。
"""
from __future__ import annotations

import struct

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

NACC = 8
R_LO, R_HI = 16, 32
WARPS = [1, 2, 4, 6, 8, 12, 16, 24, 32]

IN_LAYOUT = {
    "f_b": (0, "<f", 1.000001),
    "f_c": (4, "<f", 0.5),
    "u_b": (8, "<I", 3),
    "u_c": (12, "<I", 0x1234),
    "u_m": (16, "<I", 0x5A5A5A5A),
    "u_h": (20, "<I", 0x3C003C00),
    "d_b": (24, "<d", 1.0000001),
    "d_c": (32, "<d", 0.9999999),
}
IN_BYTES = 64


def in_buffer_bytes() -> bytes:
    raw = bytearray(IN_BYTES)
    for off, fmt, val in IN_LAYOUT.values():
        struct.pack_into(fmt, raw, off, val)
    return bytes(raw)


def _ld(off: int, ctype: str) -> str:
    return f"((const {ctype}*)((const char*)in + {off}))[0]"


def _ring(tpl: str) -> str:
    """环状依赖：第 k 条读第 k+1 个累加器，值指数式变化、不可折叠。"""
    return "\n        ".join(tpl.format(dst=k, src=(k + 1) % NACC) for k in range(NACC))


OPS: dict[str, dict] = {
    "ffma": {
        "typ": "float",
        "decl": f"float b = {_ld(0, 'float')}, c = {_ld(4, 'float')};",
        "init": "a{k} = b * ({k} + 1) + threadIdx.x;",
        "body": "\n        ".join(
            f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));',
        "note": "FP32 FMA",
    },
    "dfma": {
        "typ": "double",
        "decl": f"double b = {_ld(24, 'double')}, c = {_ld(32, 'double')};",
        "init": "a{k} = b * ({k} + 1) + threadIdx.x;",
        "body": "\n        ".join(
            f'asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(a{k}) : "d"(b), "d"(c));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.f64 [%0], %1;" :: "l"((char*)out + threadIdx.x * 8), "d"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));',
        "note": "FP64 FMA（消费卡 FP64 通道数极少）",
    },
    "hfma2": {
        "typ": "unsigned int",
        "decl": f"unsigned int hb = {_ld(20, 'unsigned int')}, hc = {_ld(8, 'unsigned int')};",
        "init": "a{k} = hb + threadIdx.x + {k};",
        "body": "\n        ".join(
            f'asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(a{k}) : "r"(hb), "r"(hc));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7));',
        "note": "FP16x2 打包乘加（每条含 2 个 FP16 运算）",
    },
    "imad": {
        "typ": "unsigned int",
        "decl": f"unsigned int b = {_ld(8, 'unsigned int')}, c = {_ld(12, 'unsigned int')};",
        "init": "a{k} = b + threadIdx.x + {k};",
        "body": "\n        ".join(
            f'asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a{k}) : "r"(b), "r"(c));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7));',
        "note": "INT32 乘加",
    },
    "iadd3": {
        "typ": "unsigned int",
        "decl": "unsigned int b = 1u;",
        "init": "a{k} = b + threadIdx.x + {k};",
        "body": _ring('asm volatile("add.u32 %0, %0, %1;" : "+r"(a{dst}) : "r"(a{src}));'),
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7));',
        "note": "INT32 加（环状，避免两两融合与线性折叠）",
    },
    "lop3": {
        "typ": "unsigned int",
        "decl": "unsigned int b = 1u;",
        "init": "a{k} = b + threadIdx.x + {k};",
        "body": _ring('asm volatile("xor.b32 %0, %0, %1;" : "+r"(a{dst}) : "r"(a{src}));'),
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7));',
        "note": "INT32 逻辑（环状 xor）",
    },
    "shf": {
        "typ": "unsigned int",
        "decl": "unsigned int b = 1u;",
        "init": "a{k} = b + threadIdx.x + {k};",
        "body": "\n        ".join(
            f'asm volatile("shf.l.wrap.b32 %0, %0, %0, 3;" : "+r"(a{k}));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7));',
        "note": "移位（循环移位，不归零）",
    },
    "mufu_ex2": {
        "typ": "float",
        "decl": f"float b = {_ld(0, 'float')};",
        "init": "a{k} = 1.0f + threadIdx.x * 1e-6f + {k};",
        "body": "\n        ".join(
            f'asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(a{k}));'
            for k in range(NACC)),
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));',
        "note": "MUFU.EX2（SFU 通道数）",
    },
    "mix_fp_int": {
        "typ": "float",
        "decl": (f"float b = {_ld(0, 'float')}, c = {_ld(4, 'float')};"
                 f" unsigned int ub = {_ld(8, 'unsigned int')}, uc = {_ld(12, 'unsigned int')};"),
        "init": "a{k} = b + threadIdx.x + {k}; i{k} = ub + threadIdx.x + {k};",
        "extra_decl": "unsigned int i0, i1, i2, i3, i4, i5, i6, i7;",
        "body": "\n        ".join(
            f'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{k}) : "f"(b), "f"(c));\n'
            f'        asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(i{k}) : "r"(ub), "r"(uc));'
            for k in range(NACC)),
        "sink": ('asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7));\n'
                 '    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + 4096 + threadIdx.x * 4), "r"(i0 ^ i1 ^ i2 ^ i3 ^ i4 ^ i5 ^ i6 ^ i7));'),
        "note": "FP32+INT32 混合（每轮 8 FFMA + 8 IMAD）",
    },
}

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""


def _build(op: str, spec: dict, rounds: int) -> str:
    decls = ", ".join(f"a{k}" for k in range(NACC))
    init = "\n        ".join(spec["init"].format(k=k) for k in range(NACC))
    body = "\n        ".join([spec["body"]] * rounds)
    extra = spec.get("extra_decl", "")
    if op == "mix_fp_int":
        extra_init = "\n        ".join(f"i{k} = ub + threadIdx.x + {k};" for k in range(NACC))
    else:
        extra_init = ""
    return f"""
extern "C" __global__ void tp_{op}_{rounds}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long t0, t1;
    {spec["decl"]}
    {extra}
    {spec["typ"]} {decls};
    {init}
    {extra_init}
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    {spec["sink"]}
}}
"""


def sass_instruction_count(sass: str) -> int:
    return sum(1 for ln in sass.splitlines()
               if ln.strip().startswith("/*") and "*/" in ln)


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)

    log("=" * 100)
    log(f"[Round 2] 指令吞吐   单 CTA / 单 SM   R={R_LO} vs {R_HI} 轮差值法  NACC={NACC}")
    log("=" * 100)

    src = PREAMBLE + "".join(
        _build(op, spec, r) for op, spec in OPS.items() for r in (R_LO, R_HI)
    )
    mod = Module(gpu, src, "throughput.cu")
    log(f"  编译完成: cubin={len(mod.cubin):,}B  nvrtc_log={mod.log.strip()!r}")

    inbuf = gpu.malloc(IN_BYTES)
    gpu.htod(inbuf, in_buffer_bytes())
    outbuf = gpu.malloc(64 * 1024)
    gpu.memset(outbuf, 0, 64 * 1024)
    pts = (inbuf, outbuf)

    results: dict = {"ops": {}, "r_lo": R_LO, "r_hi": R_HI, "nacc": NACC}
    try:
        for op, spec in OPS.items():
            k_lo = mod.func(f"tp_{op}_{R_LO}")
            k_hi = mod.func(f"tp_{op}_{R_HI}")
            d_sass = (sass_instruction_count(mod.sass(f"tp_{op}_{R_HI}"))
                      - sass_instruction_count(mod.sass(f"tp_{op}_{R_LO}")))
            per_round = d_sass / (R_HI - R_LO)
            sweep = {}
            for w in WARPS:
                threads = w * 32
                # 用中位数而不是最小值：吞吐测量里若某次重复的 cycles 读到坏值，
                # min 会把它当成"最快的一次"，导致 ops/cyc 出现 384、2621 这种
                # 荒谬值（实测踩过）。中位数对该类离群点稳健。
                c_lo = bench_mod.median(b.measure1(k_lo, threads, pts, reps))
                c_hi = bench_mod.median(b.measure1(k_hi, threads, pts, reps))
                d_cyc = c_hi - c_lo
                if d_cyc <= 0:
                    continue
                v = d_sass * threads / d_cyc
                if v > 4 * 32 * 4:  # > 512 ops/cyc 不可能，判为坏点
                    continue
                sweep[w] = round(v, 2)
            if not sweep:
                peak_w, peak = 0, float("nan")
            else:
                # 个别 warp 数下 cycles 差值会异常小（实测 iadd3@12w 得 384、
                # mix@24w 得 428，超过 4IPC×32lane=128 的上限，物理不可能）。
                # 用"中位数 1.5 倍"作上限剔除离群点，再取平台值。
                vals = sorted(sweep.values())
                med = bench_mod.median(vals)
                clean = [v for v in vals if v <= 1.5 * med] or vals
                peak = max(clean)
                peak_w = min((w for w in sweep if sweep[w] == peak), default=0)
            results["ops"][op] = {
                "note": spec["note"],
                "sass_instr_per_round": round(per_round, 3),
                "peak_ops_per_cycle": peak,
                "peak_at_warps": peak_w,
                "median_ops_per_cycle": round(bench_mod.median(list(sweep.values())), 2) if sweep else None,
                "sweep": {str(k): v for k, v in sweep.items()},
                "num_regs": k_hi.num_regs,
            }
            steps = " ".join(f"{w}w:{sweep[w]:.0f}" for w in sweep)
            log(f"  {op:<11} {peak:>7.1f} ops/cyc @{peak_w:>2}w  "
                f"SASS/轮={per_round:>5.1f} regs={k_hi.num_regs:<3} [{steps}]")
    finally:
        gpu.free(inbuf)
        gpu.free(outbuf)

    log("")
    log("  FFMA 吞吐随 warp 数（台阶位置 = 需要多少 warp 喂饱 4 个调度器）：")
    for w, v in results["ops"]["ffma"]["sweep"].items():
        log(f"    {int(w):>2} warp -> {float(v):>7.1f} ops/cyc  {'#' * int(float(v) / 4)}")

    log("")
    log("  峰值排序（ops/cycle/SM）：")
    for op, r in sorted(results["ops"].items(), key=lambda kv: -kv[1]["peak_ops_per_cycle"]):
        log(f"    {op:<11} {r['peak_ops_per_cycle']:>7.1f}   {r['note']}")
    return results
