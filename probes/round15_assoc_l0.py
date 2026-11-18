"""Round 15：相联度 / set 跨度 / 容量确认 / 干扰量化。

为什么之前的"60 KiB"不是容量
----------------------------
缓存的物理构造是 `2^k 组 × ways 路 × 行大小`，所以容量必然是
`2^k × ways × 128 B`（64 KiB = 8×64×128 或 4×128×128）。
我之前报的 "58.4–59.8 KiB" 是**平均每步代价开始上升的位置**，它低于真容量，原因有二：
1. 循环里各块并非同时驻留——替换与预取让一部分块在恰好装满前就已被挤出；
2. 每块 1424 B = 11 行，块内顺序执行、后 10 行被预取，所以超容量后也只多付 1 行的取指。

真正能给出**整数答案**的实验是冲突实验
------------------------------------
取 16 个块，让它们的**入口间距 = S 字节**，循环访问其中 N 个：
* 若 S 恰好是"一组跨过的字节数"（set span），这 N 个块**落在同一组**，
  N = ways 时还正好装下，**N = ways+1 立刻开始抖动** ⟹ 直接读出路数；
* 若 S 不是组跨度，N 个块分散到不同组，要等 N×S 超过总容量才抖动。

于是 S 扫描就能同时定出 **set 跨度**（哪些地址位参与索引）与 **ways**。

干扰
----
用户在这张卡上同时渲染图像。本轮的量化做法：同一配置重复 40 次，
报 min / median / max 与 GPU 利用率；比值就是干扰幅度。
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
REPEAT = 16


def build_src(body: int) -> str:
    """NBLK 个等长块，间距 = (body+4)*16 B；二分派发。"""
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
        blocks.extend("\tfma.rn.f32 a, a, b, e;" for _ in range(body))
        blocks.append("\tbra.uni RET;")

    asm = "\n".join([
        "{",
        "\t.reg .u32 k, i, n, m;",
        "\t.reg .pred pd;",
        "\t.reg .f32 a, b, e;",
        "\t.reg .u64 pp, t0, t1;",
        "\tmov.u32 m, 0;", "\tmov.u32 i, 0;",
        "\tmov.f32 a, 1.0;", "\tmov.f32 b, 2.0;", "\tmov.f32 e, 0.5;",
        "\tmov.u32 n, %1;", "\tmov.u64 pp, %2;",
        "\tmov.u64 t0, %clock64;",
        "DISP:", "\tld.global.u32 k, [pp];",
        *tree, *blocks,
        "RET:", "\tadd.u32 i, i, 1;", "\tadd.u64 pp, pp, 4;",
        "\tsetp.lt.u32 pd, i, n;", "\t@pd bra.uni DISP;",
        "DONE:", "\tmov.u64 t1, %clock64;", "\tsub.u64 t1, t1, t0;",
        "\tcvt.u32.u64 k, t1;", "\tst.global.u32 [%3], k;",
        "\tst.u32 [%4], m;", "\tst.f32 [%5], a;",
        "}",
    ]) + "\n"
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
    def __init__(self, gpu, body: int):
        self.gpu = gpu
        self.mod = Module(gpu, build_src(body), f"ic15_{body}.cu")
        self.offs = block_offsets(self.mod.sass("ic_steps"))
        ks = sorted(self.offs)
        self.gap = abs(self.offs[ks[1]] - self.offs[ks[0]]) if len(ks) > 1 else 0
        self.fn = self.mod.func("ic_steps").fn
        self.d_mark = gpu.malloc(4)
        self.d_sink = gpu.malloc(4)
        self.d_sq = gpu.malloc((NBLK * REPEAT + 8) * 4)
        self.d_out = gpu.malloc(4)

    def per_step(self, n_touch: int, reps: int) -> float:
        seq = list(range(n_touch)) * REPEAT
        n = len(seq)
        flat = seq + [0] * ((NBLK * REPEAT + 8) - n)
        self.gpu.htod(self.d_sq, struct.pack(f"<{len(flat)}I", *flat))
        best = 1 << 62
        for _ in range(reps):
            self.gpu.memset(self.d_out, 0xFF, 4)
            args = [ctypes.c_uint(n), ctypes.c_uint64(self.d_sq), ctypes.c_uint64(self.d_out),
                    ctypes.c_uint64(self.d_mark), ctypes.c_uint64(self.d_sink), ctypes.c_uint(0)]
            arr = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args])
            cuda.chk(cuda._F["cuLaunchKernel"](self.fn, 1, 1, 1, 32, 1, 1, 0, 0, arr, None), "launch")
            self.gpu.sync()
            best = min(best, struct.unpack("<I", self.gpu.dtoh(self.d_out, 4))[0])
        return best / n


def gpu_util(log):
    """采一次 nvidia-smi 利用率（失败就算了）。"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        log(f"    采样时 GPU: {out}  (util %, SM MHz)")
    except Exception:
        pass


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    log("=" * 96)
    log("[Round 15] 相联度 / set 跨度 / 容量 / 干扰")
    log("=" * 96)

    # ---------- A. 干扰量化 ----------
    eng128 = Engine(gpu, 4)          # 间距 128 B
    log(f"\n  A. 干扰量化：同一配置（4 块 × 间距 {eng128.gap} B）重复 40 次")
    samples = [eng128.per_step(4, 1) for _ in range(40)]
    samples.sort()
    mn, md, mx = samples[0], samples[len(samples) // 2], samples[-1]
    log(f"     min={mn:.1f}  median={md:.1f}  max={mx:.1f} 周期/步   max/min={mx/mn:.2f}×")
    gpu_util(log)
    out["noise"] = {"min": round(mn, 2), "median": round(md, 2), "max": round(mx, 2),
                    "spread": round(mx / mn, 2)}

    # ---------- B. 冲突实验：间距 S 扫描 × 触碰块数 N ----------
    log("\n  B. 冲突实验：16 个等距块，间距 S 扫描；N = 触碰块数")
    log("     同一组里的块数超过 ways 才会抖动，所以「哪个 S 先抖」定组跨度，「哪个 N 先抖」定 ways")
    strides = [128, 256, 512, 1024, 2048, 4096, 8192]
    grid: dict[int, list[float]] = {}
    log(f"     {'S':>7} 每列 = 相对 N=1 的增量（周期/步）")
    log(f"     {'S':>7} " + " ".join(f"{n:<5}" for n in range(1, NBLK + 1)))
    for S in strides:
        body = S // 16 - 4
        if body < 1:
            continue
        eng = Engine(gpu, body)
        row = []
        for n in range(1, NBLK + 1):
            row.append(eng.per_step(n, reps))
        grid[S] = row
        d0 = row[0]
        log(f"     {S:>6}B " + " ".join(f"{v - d0:+6.1f}" for v in row))
    out["conflict"] = {str(k): [round(v, 1) for v in row] for k, row in grid.items()}

    # 分析：以"N=1 的代价"为基线，找每个 S 下第一个明显上升的 N
    log("\n     每个间距下，第一个抖动点（比 N=1 高 15% 以上）：")
    onsets = {}
    for S, row in grid.items():
        base = row[0]
        onset = next((i + 1 for i, v in enumerate(row) if v > base * 1.15), None)
        onsets[S] = onset
        log(f"       S={S:>6}B  基线 {base:6.1f}  首个抖动点 "
            + (f"N={onset}" if onset else "未出现（8 块内）"))
    out["onset"] = {str(k): v for k, v in onsets.items()}
    return out
