"""Round 1：指令延迟（cycles/instruction）。

方法
----
被测指令组成**完全串行**的依赖链（第 i 条的操作数就是第 i-1 条的结果），
用 ``asm volatile`` 固定指令、阻止编译器重排。取两个迭代数 N1/N2 做差值，
消掉 clock64 读取与内核启动的固定开销。

三个必须防的坑（均已踩过并验证）
-------------------------------
1. **常量折叠**：链输入若是编译期常量，ptxas 会把整条链算成一个常数
   （``a += 1`` 重复 320 次被折成一条 ``IADD3 tid, 0x140``，"延迟"测出 0）。
   对策：所有操作数从 ``in`` 显存读，编译期不可知。
2. **指令替换/展开**：``fma.rn.f16x2`` 第三操作数为 0 会被换成 ``HADD2``；
   ``rcp/lg2/rsqrt.approx.f32``（非 ftz）会因 denormal 范围检查被展开成
   4~7 条指令。对策：第三操作数取运行时值；MUFU 全部用 ``.ftz`` 变体。
3. **指令融合**：``add.u32 a,a,b`` 重复 320 次会被两两并进一条 ``IADD3``
   （IADD3 是三操作数加法，一条顶两次加）。对策：改用
   **按实际 SASS 指令数归一** 的指标 ``cyc/SASS``，它对这个情形仍然正确。

每条结果都附带真实 SASS 条数校验：``dSASS`` 应≈(N2-N1)*per_iter。
"""
from __future__ import annotations

import struct

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

N_LO, N_HI = 64, 320

# in 缓冲区的布局：全部是"运行时未知"的值，供被测链当操作数
IN_LAYOUT = {
    "f_1000001": (0, "<f", 1.000001),
    "f_05": (4, "<f", 0.5),
    "u_3": (8, "<I", 3),
    "u_mask": (12, "<I", 0x5A5A5A5A),
    "u_1": (16, "<I", 1),
    "f_2": (20, "<f", 2.0),
    "f_1": (24, "<f", 1.0),
    "f_m1": (28, "<f", -1.0),
    "u_3c003c00": (32, "<I", 0x3C003C00),
    "u_1c003c00": (36, "<I", 0x3C001C00),
}


def in_buffer_bytes() -> bytes:
    raw = bytearray(64)
    for off, fmt, val in IN_LAYOUT.values():
        struct.pack_into(fmt, raw, off, val)
    return bytes(raw)


def _ld(off: int, ctype: str) -> str:
    return f"((const {ctype}*)((const char*)in + {off}))[0]"


PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

OPS: dict[str, dict] = {
    "ffma": {
        "decl": f"float b = {_ld(0, 'float')}, c = {_ld(4, 'float')}; float a = b + threadIdx.x;",
        "line": 'asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "FP32 FMA",
    },
    "fadd": {
        "decl": f"float b = {_ld(4, 'float')}; float a = b + threadIdx.x;",
        "line": 'asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(a) : "f"(b));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "FP32 加法",
    },
    "fmul": {
        "decl": f"float b = {_ld(0, 'float')}; float a = b + threadIdx.x;",
        "line": 'asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(a) : "f"(b));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "FP32 乘法",
    },
    "dadd": {
        "decl": f"double b = {_ld(24, 'float')}; double a = b * threadIdx.x;",
        "line": 'asm volatile("add.rn.f64 %0, %0, %1;" : "+d"(a) : "d"(b));',
        "sink": 'asm volatile("st.global.f64 [%0], %1;" :: "l"((char*)out + threadIdx.x * 8), "d"(a));',
        "note": "FP64 加法（消费卡 FP64 单元极少）",
    },
    "imad": {
        "decl": f"unsigned int b = {_ld(8, 'unsigned int')}, c = {_ld(12, 'unsigned int')};"
                f" unsigned int a = b + threadIdx.x;",
        "line": 'asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a) : "r"(b), "r"(c));',
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a));',
        "note": "INT32 乘加",
    },
    "iadd3": {
        "decl": f"unsigned int b = {_ld(16, 'unsigned int')}; unsigned int a = b + threadIdx.x;",
        "line": 'asm volatile("add.u32 %0, %0, %1;" : "+r"(a) : "r"(b));',
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a));',
        "note": "INT32 加：会被两两融合进 IADD3，看 cyc/SASS",
    },
    "shf": {
        "decl": f"unsigned int b = {_ld(16, 'unsigned int')}; unsigned int a = b + threadIdx.x;",
        "line": 'asm volatile("shl.b32 %0, %0, %1;" : "+r"(a) : "r"(b));',
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(a));',
        "note": "移位（SHF）",
    },
    "f2i": {
        "decl": f"float a = {_ld(0, 'float')} + threadIdx.x; int ia = 0;",
        "line": 'asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(ia) : "f"(a));\n'
                '        asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(a) : "r"(ia));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "类型转换往返（2 条/迭代）",
        "per_iter": 2,
    },
    "mufu_rcp": {
        "decl": f"float a = {_ld(20, 'float')} + threadIdx.x;",
        "line": 'asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(a));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "MUFU.RCP：rcp∘rcp=恒等，必被折叠，留作对照（无效）",
        "expect_fold": True,
    },
    "mufu_rsqrt": {
        "decl": f"float a = {_ld(24, 'float')} + threadIdx.x;",
        "line": 'asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(a));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "MUFU.RSQ (ftz)",
    },
    "mufu_ex2": {
        "decl": f"float a = {_ld(28, 'float')} - threadIdx.x * 1e-6f;",
        "line": 'asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(a));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "MUFU.EX2 (ftz)；值会饱和到 +inf",
    },
    "mufu_lg2": {
        "decl": f"float a = {_ld(20, 'float')} + threadIdx.x * 1e-3f;",
        "line": 'asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(a));',
        "sink": 'asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "f"(a));',
        "note": "MUFU.LG2 (ftz)",
    },
    "hfma2": {
        "decl": f"unsigned int hb = {_ld(32, 'unsigned int')}, hc = {_ld(8, 'unsigned int')};"
                f" unsigned int h = hb + threadIdx.x;",
        "line": 'asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h) : "r"(hb), "r"(hc));',
        "sink": 'asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(h));',
        "note": "FP16x2 打包乘加",
    },
}


def _kernel(name: str, n: int, spec: dict) -> str:
    body = "\n        ".join(spec["line"] for _ in range(n))
    return f"""
extern "C" __global__ void {name}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long t0, t1;
    {spec["decl"]}
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


def run(log=print, reps: int = 9) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)

    log("=" * 96)
    log(f"[Round 1] 指令延迟   N_lo={N_LO} N_hi={N_HI} reps={reps}   单 CTA / 1 warp")
    log("=" * 96)

    src = PREAMBLE + "".join(
        _kernel(f"lat_{op}_{n}", n, spec)
        for op, spec in OPS.items()
        for n in (N_LO, N_HI)
    )
    mod = Module(gpu, src, "latency.cu")
    log(f"  编译完成: cubin={len(mod.cubin):,}B  nvrtc_log={mod.log.strip()!r}")

    inbuf = gpu.malloc(64)
    gpu.htod(inbuf, in_buffer_bytes())
    sink = gpu.malloc(32 * 8)
    gpu.memset(sink, 0, 32 * 8)
    pts = (inbuf, sink)

    results: dict = {}
    suspicious: list[str] = []
    log("")
    log(f"  {'指令':<12} {'cyc/src':>8} {'cyc/SASS':>9} {'dSASS':>6}  {'说明'}")
    log("  " + "-" * 96)
    try:
        for op, spec in OPS.items():
            k_lo = mod.func(f"lat_{op}_{N_LO}")
            k_hi = mod.func(f"lat_{op}_{N_HI}")
            c_lo = min(b.measure1(k_lo, 32, pts, reps))
            c_hi = min(b.measure1(k_hi, 32, pts, reps))
            per_iter = spec.get("per_iter", 1)
            lat = (c_hi - c_lo) / float(N_HI - N_LO) / per_iter
            cnt_lo = sass_instruction_count(mod.sass(f"lat_{op}_{N_LO}"))
            cnt_hi = sass_instruction_count(mod.sass(f"lat_{op}_{N_HI}"))
            d_cnt = cnt_hi - cnt_lo
            expect_d = (N_HI - N_LO) * per_iter
            # 按真实执行的 SASS 条数归一：即使 ptxas 把两次 add 融进一条
            # IADD3，这个值仍然是对的（它测的是"一条 IADD3 多少周期"）。
            per_sass = (c_hi - c_lo) / d_cnt if d_cnt else float("nan")
            valid = d_cnt >= expect_d * 0.95
            if not valid and not spec.get("expect_fold"):
                suspicious.append(op)
            results[op] = {
                "cycles_per_src_instr": round(lat, 4),
                "cycles_per_sass_instr": round(per_sass, 4) if d_cnt else None,
                "sass_instr_lo": cnt_lo,
                "sass_instr_hi": cnt_hi,
                "sass_delta": d_cnt,
                "sass_delta_expected": expect_d,
                "valid": valid,
                "note": spec["note"],
                "num_regs": k_hi.num_regs,
            }
            if valid:
                flag = ""
            elif spec.get("expect_fold"):
                flag = "  <<预期折叠（见 note）"
            else:
                flag = "  <<指令数不符，以 cyc/SASS 为准"
            log(f"  {op:<12} {lat:>8.3f} {per_sass:>9.3f} {d_cnt:>6}  {spec['note']}{flag}")
    finally:
        gpu.free(inbuf)
        gpu.free(sink)

    log("")
    log("  --- SASS 证据（链中段的实际指令）---")
    for op in OPS:
        sass = mod.sass(f"lat_{op}_{N_HI}")
        core = []
        for ln in sass.splitlines():
            s = ln.strip()
            if not s.startswith("/*") or "*/" not in s:
                continue
            txt = s.split("*/", 1)[1].split(";")[0].strip()
            if txt and "SR_CLOCK" not in txt:
                core.append(txt)
        mid = core[len(core) // 2] if core else "?"
        results[op]["sass_mid"] = mid
        log(f"    {op:<12} {mid}")

    if suspicious:
        log("")
        log(f"  !! 指令数与源码不符（已被 ptxas 融合/展开）: {suspicious}")
    return results
