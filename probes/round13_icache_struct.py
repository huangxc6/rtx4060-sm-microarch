"""Round 13：指令缓存——稳态差值法（重复放大以战胜桌面干扰）。

核心测量式
----------
两个序列只差一步，各自在内核里重复 R 次，取总时间差 / R：

    delta = (T(seq_A + [x]) x R  -  T(seq_A) x R) / R

这样固定开销与大部分干扰被抵消，而"碰 x 这一行的代价"被放大 R 倍。

三种情形
--------
* 稳态命中：x 就是刚碰过的那一行 ⟹ delta 很小；
* 稳态失效：x 被前面的填充块挤出 L1I ⟹ delta 明显更大；
* 预取：碰 0（失效）之后紧接 x；若 x 与 0 相邻且被预取，它的 delta 接近命中值。

引擎（踩过的坑都写在 build_src 的注释里）：不用 `brx.idx`（sm_89+nvrtc 上必崩），
用谓词跳转梯；整体计时而非块内计时；累加器必须写出否则 fma 被删。
"""
from __future__ import annotations

import ctypes
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import bench as bench_mod  # noqa: E402
from harness import cuda, env          # noqa: E402
from harness.jit import Module         # noqa: E402

MARK_BASE = 0x5A0000
NPROBE = 8
NFILL = 48
PROBE_BODY = 4         # 128 B/块 = 一条 line（间距实测 (body+4)*16）
FILL_BODY = 86         # 1440 B/块
REPEAT = 48            # 放大倍数


def build_src(bodies: list[int], stride: int) -> str:
    nblk = len(bodies)
    ladder = "".join(f"\tsetp.eq.u32 p{q}, k, {q};\n\t@p{q} bra.uni LB{q};\n" for q in range(nblk))
    preds = "".join(f"\t.reg .pred p{q};\n" for q in range(nblk))
    blocks = []
    for q, body in enumerate(bodies):
        blocks.append(f"LB{q}:")
        blocks.append(f"\tadd.u32 m, m, {MARK_BASE + q};")
        blocks.extend("\tfma.rn.f32 a, a, b, e;" for _ in range(body))
        blocks.append("\tbra.uni RET;")
    asm = (
        "{\n"
        "\t.reg .u32 k, i, n, m;\n"
        "\t.reg .pred pd;\n"
        "\t.reg .f32 a, b, e;\n"
        "\t.reg .u64 pp, t0, t1;\n"
        + preds +
        "\tmov.u32 m, 0;\n\tmov.u32 i, 0;\n"
        "\tmov.f32 a, 1.0;\n\tmov.f32 b, 2.0;\n\tmov.f32 e, 0.5;\n"
        "\tmov.u32 n, %1;\n\tmov.u64 pp, %2;\n"
        "\tmov.u64 t0, %clock64;\n"
        "DISP:\n"
        "\tld.global.u32 k, [pp];\n"
        + ladder +
        "\tbra.uni DONE;\n"
        + "\n".join(blocks) + "\n"
        "RET:\n"
        "\tadd.u32 i, i, 1;\n\tadd.u64 pp, pp, 4;\n"
        "\tsetp.lt.u32 pd, i, n;\n\t@pd bra.uni DISP;\n"
        "DONE:\n"
        "\tmov.u64 t1, %clock64;\n\tsub.u64 t1, t1, t0;\n\tcvt.u32.u64 k, t1;\n"
        "\tst.global.u32 [%3], k;\n"
        "\tst.u32 [%4], m;\n"
        "\tst.f32 [%5], a;\n"
        "}\n"
    )
    esc = asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return (
        "extern \"C\" __global__ void ic_steps(unsigned n, const unsigned* sq,\n"
        "                                     unsigned* out, unsigned* mark,\n"
        "                                     float* sink, unsigned stride) {\n"
        "    unsigned dummy;\n"
        f'    asm volatile("{esc}"\n'
        '        : "=r"(dummy)\n'
        '        : "r"(n), "l"(sq), "l"(out), "l"(mark), "l"(sink), "r"(stride), "f"(1.0f));\n'
        "}\n"
    )


def block_offsets(sass: str, nblk: int) -> dict[int, int]:
    offs: dict[int, int] = {}
    for ln in sass.splitlines():
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+\S+\s+R\d+, R\d+, (0x[0-9a-f]+)", ln)
        if m:
            v = int(m.group(2), 16)
            if MARK_BASE <= v < MARK_BASE + nblk:
                offs.setdefault(v - MARK_BASE, int(m.group(1), 16))
    return offs


class Engine:
    def __init__(self, gpu, bodies: list[int]):
        self.gpu = gpu
        self.nblk = len(bodies)
        self.stride = self.nblk + 64
        self.mod = Module(gpu, build_src(bodies, self.stride), "ic13.cu")
        self.offs = block_offsets(self.mod.sass("ic_steps"), self.nblk)
        self.fn = self.mod.func("ic_steps").fn
        self.d_mark = gpu.malloc(4)
        self.d_sink = gpu.malloc(4)
        self.d_sq = gpu.malloc(self.stride * 4)
        self.d_sq_cap = self.stride
        self.d_out = gpu.malloc(4)

    def total(self, seq: list[int], reps: int = 5) -> int:
        n = len(seq)
        if n > self.d_sq_cap:                      # 按需扩容（放大后序列会很长）
            self.gpu.free(self.d_sq)
            self.d_sq_cap = n + 64
            self.d_sq = self.gpu.malloc(self.d_sq_cap * 4)
        flat = seq + [0] * (self.d_sq_cap - n)
        self.gpu.htod(self.d_sq, struct.pack(f"<{self.d_sq_cap}I", *flat))
        best = 1 << 62
        for _ in range(reps):
            self.gpu.memset(self.d_out, 0xFF, 4)
            args = [ctypes.c_uint(n), ctypes.c_uint64(self.d_sq), ctypes.c_uint64(self.d_out),
                    ctypes.c_uint64(self.d_mark), ctypes.c_uint64(self.d_sink),
                    ctypes.c_uint(self.stride)]
            arr = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args])
            cuda.chk(cuda._F["cuLaunchKernel"](self.fn, 1, 1, 1, 32, 1, 1, 0, 0, arr, None), "launch")
            self.gpu.sync()
            best = min(best, struct.unpack("<I", self.gpu.dtoh(self.d_out, 4))[0])
        return best

    def per_step(self, pattern: list[int], tail: list[int], reps: int = 5) -> float:
        """(T(pattern+tail) - T(pattern)) / len(tail)，输入输出都已放大 REPEAT 倍。"""
        a = self.total(pattern * REPEAT, reps)
        b = self.total((pattern + tail) * REPEAT, reps)
        return (b - a) / (len(tail) * REPEAT)


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    bodies = [PROBE_BODY] * NPROBE + [FILL_BODY] * NFILL
    eng = Engine(gpu, bodies)
    ks = sorted(eng.offs)
    gap = abs(eng.offs[ks[1]] - eng.offs[ks[0]])

    log("=" * 96)
    log("[Round 13] 指令缓存：稳态差值法（重复 48 次放大，战胜桌面干扰）")
    log("=" * 96)
    log(f"  引擎: {NPROBE} 个行级块（间距 {gap} B）+ {NFILL} 个填充块（{FILL_BODY*16} B/块），"
        f"足迹 {abs(eng.offs[ks[-1]]-eng.offs[ks[0]])/1024:.1f} KiB，放大 {REPEAT}×")

    # 基线：命中（紧邻自重复） vs 失效（每次都被 48 个填充块冲洗）
    hit = eng.per_step([1], [1], reps)
    miss = eng.per_step(list(range(NPROBE, NPROBE + NFILL)), [0], reps)
    log(f"\n  基线 每步代价: 命中 {hit:.1f} 周期 / 失效 {miss:.1f} 周期")
    out["baseline"] = {"hit": round(hit, 2), "miss": round(miss, 2)}
    thr = (hit + miss) / 2
    log(f"  判定阈值 {thr:.1f} 周期")

    # ---------------- A. 容量 ----------------
    log("\n  A. 容量：n_ev 个填充块施压后，碰探针块 0 的代价")
    log(f"     {'n_ev':>5} {'压力':>10} {'每步周期':>9} {'判定':>6}")
    cap = {}
    for n_ev in (0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48):
        d = eng.per_step([NPROBE + i for i in range(n_ev)], [0], reps)
        pressure = n_ev * FILL_BODY * 16
        cap[n_ev] = {"pressure_bytes": pressure, "cycles": round(d, 2),
                     "verdict": "HIT" if d < thr else "MISS"}
        log(f"     {n_ev:>5} {pressure/1024:>9.1f}K {d:>9.1f} {cap[n_ev]['verdict']:>6}")
    out["capacity"] = cap

    # ---------------- B. 预取 ----------------
    log("\n  B. 预取：先碰块 0（必失效），紧接碰目标块")
    log(f"     {'目标':>10} {'每步周期':>9} {'判定':>8}")
    evict = [NPROBE + i for i in range(NFILL)]
    pf = []
    for label, idx in (("+1 相邻", 1), ("+2", 2), ("+3", 3), ("+7 远", 7)):
        d = eng.per_step(evict + [0], [idx], reps)
        pf.append({"target": label, "index": idx, "cycles": round(d, 2),
                   "prefetched": bool(d < thr)})
        log(f"     {label:>10} {d:>9.1f} {('已预取' if d < thr else '未预取'):>8}")
    out["prefetch"] = pf

    # ---------------- C. 替换 ----------------
    log("\n  C. 替换：1 行循环 vs 8 行循环 的每步代价")
    one = eng.per_step([0], [0], reps)
    eight = eng.per_step([0], list(range(1, 8)), reps)
    log(f"     单行循环 {one:.1f} 周期/步；8 行循环 {eight:.1f} 周期/步")
    out["replacement"] = {"one_line": round(one, 2), "eight_lines": round(eight, 2)}
    return out
