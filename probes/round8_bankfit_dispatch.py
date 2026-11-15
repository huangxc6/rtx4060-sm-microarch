"""Round 8：warp 发射节流速率 + 用统计法拟合寄存器 bank 结构。

8.1 warp 发射节流
    Round 6 看到 16 warp 时有 warp 的起始偏移达 807 周期，但没测速率。
    做法：blockDim 从 1 warp 扫到 32 warp，每个 warp 记录**绝对**起始 `%clock64`，
    对 warp 序号做线性拟合 → 每 N 周期上一个 warp。

8.2 寄存器 bank 结构的统计拟合
    Round 7 测到"三源全不同"的 FFMA 只有 78 ops/cycle（可复用时 125），
    但分不清是"读端口不足"还是"bank 冲突"。这里不需要控制寄存器分配：

    * 从 SASS 里取出计时区内**全部** FFMA 的 (A,B,C) 寄存器三元组；
    * 对每个候选 bank 数 B ∈ {2,4,8,16}，算「三个源落在 3 个不同 bank」的比例 f_B；
    * 若冲突代价是 +1 周期，则预测吞吐 = 128 / (1 + (1 - f_B))；
    * 与实测值比对，看哪个 B 能对上。

    "读端口不足"模型的预测与寄存器分布无关，恒为 128/1.5 = 85.3——
    这也是一个可区分的预测。
"""
from __future__ import annotations

import struct

from harness import bench as bench_mod
from harness import env
from harness.jit import Module
from probes.round7_bank_agg import PATTERNS, PREAMBLE, _kernel

# --------------------------------------------------------------------------
# 8.1 warp 发射节流
# --------------------------------------------------------------------------
WARP_COUNTS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]


def _dispatch_kernel() -> str:
    """每个 warp 只做很少的活，尽量把它自己的起始时刻暴露出来。"""
    return f"""
extern "C" __global__ void wdisp(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long t0;
    CLK(t0);
    // 一点点工作，确保 warp 真的被调度过
    float a = ((const float*)((const char*)in + 0))[0] + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < 32; ++i) {{
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(a), "f"(a));
    }}
    unsigned long long t1;
    CLK(t1);
    unsigned int wid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(wid));
    if ((threadIdx.x & 31u) == 0) {{
        unsigned long long base = (unsigned long long)out
            + (unsigned long long)(threadIdx.x >> 5) * 32ull;
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base), "l"((unsigned long long)wid));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 8), "l"(t0));
        asm volatile("st.global.b64 [%0], %1;" :: "l"(base + 16), "l"(t1 - t0));
    }}
    asm volatile("st.global.f32 [%0], %1;" :: "l"((char*)out + 16384 + threadIdx.x * 4), "f"(a));
}}
"""


# --------------------------------------------------------------------------
# 8.2 bank 拟合
# --------------------------------------------------------------------------
def _ffma_triples(mod, name: str) -> list[tuple[int, int, int]]:
    """取函数里全部 FFMA 的 (A,B,C) 寄存器号，丢掉首尾各 16 条（初始化/收尾）。"""
    lines = []
    for ln in mod.sass(name).splitlines():
        s = ln.strip()
        if not s.startswith("/*") or "*/" not in s:
            continue
        txt = s.split("*/", 1)[1].split(";")[0].strip()
        if txt.startswith("FFMA"):
            lines.append(txt)
    out = []
    for txt in lines[16:-16] if len(lines) > 40 else lines:
        regs = []
        for part in txt.split(None, 1)[1].split(","):
            p = part.strip().replace(".reuse", "")
            if p.startswith("R"):
                try:
                    regs.append(int(p[1:].split(".")[0]))
                except ValueError:
                    regs.append(-1)
            else:
                regs.append(-1)
        if len(regs) >= 4 and all(r >= 0 for r in regs[1:4]):
            out.append((regs[1], regs[2], regs[3]))
    return out


def _fit_banks(triples: list[tuple[int, int, int]], measured: float) -> dict:
    res = {"n_instr": len(triples), "measured_ops_per_cycle": measured, "models": {}}
    if not triples:
        return res
    for banks in (2, 4, 8, 16, 32):
        distinct = sum(1 for a, b, c in triples
                       if len({a % banks, b % banks, c % banks}) == 3)
        frac = distinct / len(triples)
        pred = 128.0 / (1.0 + (1.0 - frac))
        res["models"][str(banks)] = {
            "frac_all_distinct": round(frac, 4),
            "predicted_ops_per_cycle": round(pred, 2),
            "error_pct": round(abs(pred - measured) / measured * 100, 2),
        }
    # 端口模型：与寄存器分布无关
    res["models"]["port_3reads"] = {
        "frac_all_distinct": None,
        "predicted_ops_per_cycle": round(128.0 / 1.5, 2),
        "error_pct": round(abs(128.0 / 1.5 - measured) / measured * 100, 2),
    }
    return res


def run(log=print, reps: int = 7) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    out: dict = {"clock_ghz": round(ghz, 4)}

    inbuf = gpu.malloc(64)
    gpu.htod(inbuf, struct.pack("<ff", 1.000001, 0.5) + bytes(56))
    obuf = gpu.malloc(32768)
    gpu.memset(obuf, 0, 32768)

    # ================= 8.1 warp 发射节流 =================
    log("=" * 96)
    log("[Round 8.1] warp 发射节流：第 n 个 warp 何时开始？")
    log("=" * 96)
    mod = Module(gpu, PREAMBLE + _dispatch_kernel(), "wdisp.cu")
    k = mod.func("wdisp")
    disp: dict = {}
    log(f"  {'warp':>5} {'最大起始偏移':>12} {'拟合斜率':>10}  各 warp 起始偏移")
    for w in WARP_COUNTS:
        best = None
        for _ in range(5):
            k.launch(1, w * 32, (0, inbuf, obuf))
            gpu.sync()
            raw = gpu.dtoh(obuf, w * 32)
            rec = [struct.unpack_from("<QQQ", raw, i * 32) for i in range(w)]
            t0s = [r[1] for r in rec]
            span = max(t0s) - min(t0s)
            if best is None or span < best[0]:
                best = (span, t0s)
        t0s = best[1]
        base = min(t0s)
        offs = [t - base for t in t0s]
        # 线性拟合 offset ≈ slope * warp_index
        n = len(offs)
        idx = list(range(n))
        mx = sum(idx) / n
        my = sum(offs) / n
        denom = sum((i - mx) ** 2 for i in idx)
        slope = sum((i - mx) * (o - my) for i, o in zip(idx, offs)) / denom if denom else 0.0
        disp[str(w)] = {"offsets": [int(x) for x in offs],
                        "max_offset": int(max(offs)),
                        "slope_cycles_per_warp": round(slope, 2)}
        shown = " ".join(str(int(x)) for x in offs[:12])
        log(f"  {w:>5} {max(offs):>12} {slope:>10.2f}  {shown}"
            + (" …" if len(offs) > 12 else ""))
    out["dispatch"] = disp

    # ================= 8.2 bank 拟合 =================
    log("")
    log("=" * 96)
    log("[Round 8.2] 用 SASS 里的全部 FFMA 三元组拟合 bank 结构")
    log("=" * 96)
    src = PREAMBLE + _kernel("p_distinct", PATTERNS["distinct"])
    mod2 = Module(gpu, src, "bankfit.cu")
    k2 = mod2.func("p_distinct")
    cyc = bench_mod.median(b.measure1(k2, 4 * 32, (inbuf, obuf), reps))
    ops = 8 * 32 * 4 * 32
    measured = ops / cyc
    triples = _ffma_triples(mod2, "p_distinct")
    fit = _fit_banks(triples, measured)
    out["bank_fit"] = fit

    log(f"  实测: {measured:.1f} ops/cycle（{len(triples)} 条 FFMA 参与统计）")
    log("")
    log(f"  {'模型':<14} {'三源全不同的比例':>14} {'预测 ops/cyc':>12} {'误差':>8}")
    for name, m in fit["models"].items():
        frac = "-" if m["frac_all_distinct"] is None else f'{m["frac_all_distinct"]:.3f}'
        log(f"  {name:<14} {frac:>14} {m['predicted_ops_per_cycle']:>12.1f} "
            f"{m['error_pct']:>7.1f}%")
    log("")
    log("  判读：若某个 bank 数能对上实测，则 78 是 bank 冲突；")
    log("        若只有 port_3reads（与寄存器分布无关）接近，则只是读端口不足。")

    gpu.free(obuf)
    gpu.free(inbuf)
    return out
