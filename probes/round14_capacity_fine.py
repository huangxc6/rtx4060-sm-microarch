"""Round 13b：L1I 容量细扫——二分派发 + 稳态每步代价。

三次修正的历程（都留在这里，避免以后重踩）
------------------------------------------
1. **线性派发梯的代价 ∝ 块号**。用它做"足迹扫描"时，每步耗时随块号线性上升，
   测出来是一条平滑的 1.84× 上升曲线——那不是缓存失效，是派发开销。
   → 改用**二分派发树**，深度 = log2(nblk)，与块号无关。
2. **差值法在超容量区间失效**。`T(压力)` 与 `T(压力+探针)` 是两种不同长度的稳态循环，
   每个块的命中率都变了，差值反映的不是"碰探针那一行"。
   实测表现为 29 KiB 就 MISS、34.8 KiB 又 HIT 的乱跳。
   → 改成直接测**循环本身的稳态每步代价**：足迹 < 容量 ⟹ 每步命中；超过 ⟹ 每步重取。
3. **块内计时会漏掉该行自己的取指**（块 0 压到 75 KiB 仍"命中"）。
   → 整体计时（循环前后各一次 clock64），再除以步数。
"""
from __future__ import annotations

import ctypes
import json
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
NLINE = 8
NBIG = 60
NTRIM = 12
LINE_BODY = 4          # (4+4)*16 = 128 B
BIG_BODY = 86          # (86+4)*16 = 1440 B
REPEAT = 24
IDX_BIG = NLINE
IDX_TRIM = NLINE + NBIG


def build_src(bodies: list[int]) -> str:
    """二分派发树：深度 log2(nblk)，与目标块号无关。"""
    nblk = len(bodies)
    tree: list[str] = []
    counter = [0]

    def emit(lo: int, hi: int):
        if hi - lo == 1:
            tree.append(f"\tbra.uni LB{lo};")
            return
        mid = (lo + hi) // 2
        a = f"T{counter[0]}"
        b = f"T{counter[0] + 1}"
        counter[0] += 2
        tree.append(f"\tsetp.lt.u32 pd, k, {mid};")
        tree.append(f"\t@pd bra.uni {a};")
        tree.append(f"T{counter[0]}_r:")
        tree.append(f"\tbra.uni {b};")
        tree.append(f"{a}:")
        emit(lo, mid)
        tree.append(f"{b}:")
        emit(mid, hi)

    emit(0, nblk)

    blocks: list[str] = []
    for q, body in enumerate(bodies):
        blocks.append(f"LB{q}:")
        blocks.append(f"\tadd.u32 m, m, {MARK_BASE + q};")
        blocks.extend("\tfma.rn.f32 a, a, b, e;" for _ in range(body))
        blocks.append("\tbra.uni RET;")

    asm_lines = [
        "{",
        "\t.reg .u32 k, i, n, m;",
        "\t.reg .pred pd;",
        "\t.reg .f32 a, b, e;",
        "\t.reg .u64 pp, t0, t1;",
        "\tmov.u32 m, 0;",
        "\tmov.u32 i, 0;",
        "\tmov.f32 a, 1.0;",
        "\tmov.f32 b, 2.0;",
        "\tmov.f32 e, 0.5;",
        "\tmov.u32 n, %1;",
        "\tmov.u64 pp, %2;",
        "\tmov.u64 t0, %clock64;",
        "DISP:",
        "\tld.global.u32 k, [pp];",
        *tree,
        *blocks,
        "RET:",
        "\tadd.u32 i, i, 1;",
        "\tadd.u64 pp, pp, 4;",
        "\tsetp.lt.u32 pd, i, n;",
        "\t@pd bra.uni DISP;",
        "DONE:",
        "\tmov.u64 t1, %clock64;",
        "\tsub.u64 t1, t1, t0;",
        "\tcvt.u32.u64 k, t1;",
        "\tst.global.u32 [%3], k;",
        "\tst.u32 [%4], m;",
        "\tst.f32 [%5], a;",
        "}",
    ]
    asm = "\n".join(asm_lines) + "\n"
    esc = asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return (
        'extern "C" __global__ void ic_steps(unsigned n, const unsigned* sq,\n'
        '                                     unsigned* out, unsigned* mark,\n'
        '                                     float* sink, unsigned stride) {\n'
        '    unsigned dummy;\n'
        f'    asm volatile("{esc}"\n'
        '        : "=r"(dummy)\n'
        '        : "r"(n), "l"(sq), "l"(out), "l"(mark), "l"(sink), "r"(stride), "f"(1.0f));\n'
        '}\n'
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
        self.mod = Module(gpu, build_src(bodies), "ic13e.cu")
        self.offs = block_offsets(self.mod.sass("ic_steps"), len(bodies))
        self.fn = self.mod.func("ic_steps").fn
        self.d_mark = gpu.malloc(4)
        self.d_sink = gpu.malloc(4)
        self.d_cap = 64
        self.d_sq = gpu.malloc(self.d_cap * 4)
        self.d_out = gpu.malloc(4)

    def total(self, seq: list[int]) -> int:
        n = len(seq)
        if n > self.d_cap:
            self.gpu.free(self.d_sq)
            self.d_cap = n + 256
            self.d_sq = self.gpu.malloc(self.d_cap * 4)
        flat = seq + [0] * (self.d_cap - n)
        self.gpu.htod(self.d_sq, struct.pack(f"<{self.d_cap}I", *flat))
        self.gpu.memset(self.d_out, 0xFF, 4)
        args = [ctypes.c_uint(n), ctypes.c_uint64(self.d_sq), ctypes.c_uint64(self.d_out),
                ctypes.c_uint64(self.d_mark), ctypes.c_uint64(self.d_sink), ctypes.c_uint(0)]
        arr = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args])
        cuda.chk(cuda._F["cuLaunchKernel"](self.fn, 1, 1, 1, 32, 1, 1, 0, 0, arr, None), "launch")
        self.gpu.sync()
        return struct.unpack("<I", self.gpu.dtoh(self.d_out, 4))[0]


def run(log=print, reps: int = 7) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    bodies = [LINE_BODY] * NLINE + [BIG_BODY] * NBIG + [LINE_BODY] * NTRIM
    eng = Engine(gpu, bodies)
    ks = sorted(eng.offs)
    gap_big = abs(eng.offs[ks[NLINE + 1]] - eng.offs[ks[NLINE]])
    gap_trim = abs(eng.offs[ks[NLINE + NBIG + 1]] - eng.offs[ks[NLINE + NBIG]])

    log("=" * 96)
    log("[Round 13b] L1I 容量：稳态每步代价 vs 足迹（二分派发）")
    log("=" * 96)
    log(f"  行级块 128 B、大填充块 {gap_big} B、微调块 {gap_trim} B；放大 {REPEAT}×")

    curve = []
    for n_big in range(4, NBIG + 1):
        seq = [IDX_BIG + i for i in range(n_big)] * REPEAT
        best = 1 << 62
        for _ in range(reps):
            best = min(best, eng.total(seq))
        per = best / (n_big * REPEAT)
        curve.append({"n_big": n_big, "bytes": n_big * gap_big, "per_step": round(per, 2)})
    out["curve"] = curve

    base = min(c["per_step"] for c in curve)
    log(f"\n  粗扫（{len(curve)} 点，{gap_big} B/点）")
    log(f"  {'足迹':>9} {'每步周期':>9} {'相对':>7}")
    for c in curve:
        if c["n_big"] % 2 == 0 or c["bytes"] > 55 * 1024:
            bar = "#" * int(min(40, (c["per_step"] - base) * 1.5))
            log(f"  {c['bytes']/1024:>8.2f}K {c['per_step']:>9.1f} "
                f"{c['per_step']/base:>6.2f}x {bar}")

    # 说明：曾尝试用 112 B 的"微调块"把临界区再细化到 128 B/点，但结果是锯齿——
    # 因为"每步平均"把 ~600 周期的大块步和 ~20 周期的微调步平均在一起（合成伪影）。
    # 所以本探针只保留**统一尺寸**的块，粒度 = 块大小 1424 B。

    knee = next((c for c in curve if c["per_step"] > base * 1.03), None)
    if knee and curve.index(knee) > 0:
        prev = curve[curve.index(knee) - 1]
        log(f"\n  拐点: {prev['bytes']/1024:.2f} KiB（{prev['per_step']:.1f}） → "
            f"{knee['bytes']/1024:.2f} KiB（{knee['per_step']:.1f}）")
        out["knee"] = {"last_flat": prev, "first_rise": knee, "baseline": base}
    return out


if __name__ == "__main__":
    env.prepare()
    r = run()
    print(json.dumps(r.get("knee", {}), ensure_ascii=False))