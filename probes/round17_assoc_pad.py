"""Round 17：相联度与 set 跨度（用「可达但不执行」的填充解耦间距与执行代价）。

前一轮（Round 15）为什么失败
--------------------------
为了让块落在 S 字节处，块体必须填到 S/16−4 条 fma，于是**执行代价 ∝ 间距**
（208 → 2322 周期/步），30–50 周期的取指信号被埋掉；再加上二分派发路径随 N 变化，
结果非单调、无法解读。

本轮的修法
----------
块体固定（4 条 fma = 64 B 真代码），间距靠**从不执行但在 CFG 里可达**的填充撑开：

    LB{q}:
        add.u32 m, m, MARK+q        ; 标记（用于从 SASS 反查块地址）
        <body: 4 × fma>
        bra.uni RET
        setp.eq.u32 pd2, flag, 0x7FFFFFFF   ; 运行时 flag=1 ⟹ 恒假
        @pd2 bra.uni SKIP{q}                ; 永不跳转
        <PAD: 若干 fma，写 a2>
    SKIP{q}:

于是：执行代价恒定、块地址间距可控、**每块只触碰第一条 line（128 B）**，
足迹 = N × 128 B —— 与间距无关。这样"同集合碰撞"就能和"总容量不足"彻底分开：

* 若 S 是 set 跨度（或它的整数倍），N 个块落在同一组：
  N ≤ ways 时全命中，**N = ways+1 开始抖动** ⟹ 直接读出 ways；
* 若 S 不是 set 跨度，N 个块分散在不同组，足迹又只有 N×128 B，
  所以**永远不会抖**（128 B × 16 = 2 KiB 远小于 64 KiB）。
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
NBLK = 16
BODY = 4          # 固定的真实代码量（64 B）
REPEAT = 16


def build_src(pad: int) -> str:
    nblk = NBLK
    tree: list[str] = []
    counter = [0]

    def emit(lo: int, hi: int):
        if hi - lo == 1:
            tree.append(f"\tbra.uni LB{lo};")
            return
        mid = (lo + hi) // 2
        a, b = f"T{counter[0]}", f"T{counter[0] + 1}"
        counter[0] += 2
        tree.append(f"\tsetp.lt.u32 pd, k, {mid};")
        tree.append(f"\t@pd bra.uni {a};")
        tree.append(f"\tbra.uni {b};")
        tree.append(f"{a}:")
        emit(lo, mid)
        tree.append(f"{b}:")
        emit(mid, hi)

    emit(0, nblk)

    blocks: list[str] = []
    for q in range(nblk):
        blocks.append(f"LB{q}:")
        blocks.append(f"\tadd.u32 m, m, {MARK_BASE + q};")
        blocks.extend("\tfma.rn.f32 a, a, b, e;" for _ in range(BODY))
        blocks.append("\tbra.uni RET;")
        if pad > 0:
            blocks.append("\tsetp.eq.u32 pd2, flag, 0x7FFFFFFF;")
            blocks.append(f"\t@pd2 bra.uni SKIP{q};")
            blocks.extend("\tfma.rn.f32 a2, a2, b, e;" for _ in range(pad))
            blocks.append(f"SKIP{q}:")

    asm = "\n".join([
        "{",
        "\t.reg .u32 k, i, n, m, flag;",
        "\t.reg .pred pd, pd2;",
        "\t.reg .f32 a, a2, b, e;",
        "\t.reg .u64 pp, t0, t1;",
        "\tmov.u32 m, 0;", "\tmov.u32 i, 0;",
        "\tmov.f32 a, 1.0;", "\tmov.f32 a2, 3.0;",
        "\tmov.f32 b, 2.0;", "\tmov.f32 e, 0.5;",
        "\tmov.u32 n, %1;", "\tmov.u32 flag, %6;", "\tmov.u64 pp, %2;",
        "\tmov.u64 t0, %clock64;",
        "DISP:", "\tld.global.u32 k, [pp];",
        *tree, *blocks,
        "RET:", "\tadd.u32 i, i, 1;", "\tadd.u64 pp, pp, 4;",
        "\tsetp.lt.u32 pd, i, n;", "\t@pd bra.uni DISP;",
        "DONE:", "\tmov.u64 t1, %clock64;", "\tsub.u64 t1, t1, t0;",
        "\tcvt.u32.u64 k, t1;",
        "\tst.global.u32 [%3], k;",
        "\tst.u32 [%4], m;", "\tst.f32 [%5], a;", "\tst.f32 [%5+4], a2;",
        "}",
    ]) + "\n"
    esc = asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return (
        'extern "C" __global__ void ic_assoc(unsigned n, const unsigned* sq,\n'
        '                                     unsigned* out, unsigned* mark,\n'
        '                                     float* sink, unsigned stride, unsigned flag) {\n'
        '    unsigned dummy;\n'
        f'    asm volatile("{esc}"\n'
        '        : "=r"(dummy)\n'
        '        : "r"(n), "l"(sq), "l"(out), "l"(mark), "l"(sink), "r"(stride),\n'
        '          "r"(flag), "f"(1.0f));\n'
        '}\n'
    )


def block_offsets(sass: str) -> dict[int, int]:
    offs: dict[int, int] = {}
    for ln in sass.splitlines():
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+\S+\s+R\d+, R\d+, (0x[0-9a-f]+)", ln)
        if m:
            v = int(m.group(2), 16)
            if MARK_BASE <= v < MARK_BASE + NBLK:
                offs.setdefault(v - MARK_BASE, int(m.group(1), 16))
    return offs


class Engine:
    def __init__(self, gpu, pad: int):
        self.gpu = gpu
        self.mod = Module(gpu, build_src(pad), f"ic17_{pad}.cu")
        self.offs = block_offsets(self.mod.sass("ic_assoc"))
        ks = sorted(self.offs)
        self.gap = abs(self.offs[ks[1]] - self.offs[ks[0]]) if len(ks) > 1 else 0
        self.fn = self.mod.func("ic_assoc").fn
        self.d_mark = gpu.malloc(4)
        self.d_sink = gpu.malloc(8)
        self.cap = NBLK * REPEAT + 8
        self.d_sq = gpu.malloc(self.cap * 4)
        self.d_out = gpu.malloc(4)

    def per_step(self, n_touch: int, reps: int) -> float:
        seq = list(range(n_touch)) * REPEAT
        n = len(seq)
        flat = seq + [0] * (self.cap - n)
        self.gpu.htod(self.d_sq, struct.pack(f"<{self.cap}I", *flat))
        best = 1 << 62
        for _ in range(reps):
            self.gpu.memset(self.d_out, 0xFF, 4)
            args = [ctypes.c_uint(n), ctypes.c_uint64(self.d_sq), ctypes.c_uint64(self.d_out),
                    ctypes.c_uint64(self.d_mark), ctypes.c_uint64(self.d_sink),
                    ctypes.c_uint(0), ctypes.c_uint(1)]
            arr = (ctypes.c_void_p * 7)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args])
            cuda.chk(cuda._F["cuLaunchKernel"](self.fn, 1, 1, 1, 32, 1, 1, 0, 0, arr, None), "launch")
            self.gpu.sync()
            best = min(best, struct.unpack("<I", self.gpu.dtoh(self.d_out, 4))[0])
        return best / n


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    log("=" * 96)
    log("[Round 17] 相联度 / set 跨度（块体固定 4 条 fma，间距靠不执行的填充撑开）")
    log("=" * 96)
    log(f"     足迹恒为 N × 128 B（每块只碰第一条 line），与间距无关")
    log(f"\n     {'间距 S':>9} {'N=1':>7} " + " ".join(f"{n:>6}" for n in (2, 4, 8, 9, 12, 16)))

    results = {}
    for S in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        # 每块开销约 body+4 条，其余用填充补齐
        pad = max(0, S // 16 - BODY - 6)
        eng = Engine(gpu, pad)
        row = {}
        for n in (1, 2, 4, 8, 9, 12, 16):
            row[n] = eng.per_step(n, reps)
        base = row[1]
        log(f"     {eng.gap:>8}B " + f"{base:7.1f} " +
            " ".join(f"{row[n] + 0:6.1f}" for n in (2, 4, 8, 9, 12, 16)))
        results[str(eng.gap)] = {str(k): round(v, 1) for k, v in row.items()}

        deltas = {n: row[n] - base for n in row if n > 1}
        worst = max(deltas, key=lambda n: deltas[n])
        log(f"       → 最大增量 {deltas[worst]:+.1f}（N={worst}）"
            + ("   <<< 疑似同集合冲突" if deltas[worst] > 15 else ""))
    out["sweep"] = results
    return out


if __name__ == "__main__":
    env.prepare()
    r = run()
    print(r.get("sweep", {}).keys())
