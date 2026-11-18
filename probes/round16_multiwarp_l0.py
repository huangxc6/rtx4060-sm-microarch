"""Round 16：L0 / L1.5 的分区归属（多 warp 各跑独立代码区）。

要回答的问题
------------
1. 每个调度器分区是否有**独立**的指令缓存（L0）？多大？
2. 多个分区之间是**共享**还是独立？如果有共享的中间层（L1.5），它的容量是多少？

判据设计
--------
* **同分区**：warp 0/4/8/12（Round 4/6 已确认 4k 与 4k+4 落在同一调度器），
  每个 warp 流式访问**自己那一段代码**（8 块 × 间距 S）。
  若分区有独立 L0，那么 4 个 warp 合计足迹超过 L0 时，每步代价上升；
  而 1 个 warp 单独跑同样足迹不上升 ⟹ 上升点就是 L0 容量。
* **跨分区**：warp 0/1/2/3（各自一个分区）。同样足迹下若**不**上升，
  说明限制是"每分区"的（L0）；若同样上升，说明是共享层（L1/L1.5）。
* 再把单 warp 的容量上限（Round 14 约 64 KiB）作为共享层的参考。

引擎
----
二分派发（深度恒定）+ 每 warp 自己的步进指针；每个 warp 用 `clock64` 自己计时，
所以并发时的相互争用会被计入各自的读数。
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
NBLK = 32              # 每 warp 8 块，最多 4 个 warp
PER_WARP = 8
REPEAT = 24


def build_src(body: int) -> str:
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

    stride_words = PER_WARP * REPEAT + 4
    asm = "\n".join([
        "{",
        "\t.reg .u32 k, i, n, m, w, off;",
        "\t.reg .pred pd;",
        "\t.reg .f32 a, b, e;",
        "\t.reg .u64 pp, qq, t0, t1, base;",
        "\tmov.u32 w, %tid.x;", "\tshr.u32 w, w, 5;",
        f"\tsetp.ge.u32 pd, w, 16;", "\t@pd bra.uni DONE;",
        f"\tmul.lo.u32 off, w, {stride_words};",
        "\tcvt.u64.u32 base, off;", "\tshl.b64 base, base, 2;",
        "\tadd.u64 pp, base, %2;",
        "\tadd.u64 qq, base, %3;",
        "\tmov.u32 m, 0;", "\tmov.u32 i, 0;",
        "\tmov.f32 a, 1.0;", "\tmov.f32 b, 2.0;", "\tmov.f32 e, 0.5;",
        "\tmov.u32 n, %1;",
        "\tmov.u64 t0, %clock64;",
        "DISP:", "\tld.global.u32 k, [pp];",
        # 哨兵：该 warp 没有序列（值为 0xFFFFFFFF）就直接退出，避免它污染缓存
        "\tsetp.eq.u32 pd, k, 0xFFFFFFFF;", "\t@pd bra.uni DONE;",
        *tree, *blocks,
        "RET:", "\tadd.u32 i, i, 1;", "\tadd.u64 pp, pp, 4;",
        "\tsetp.lt.u32 pd, i, n;", "\t@pd bra.uni DISP;",
        "DONE:", "\tmov.u64 t1, %clock64;", "\tsub.u64 t1, t1, t0;",
        "\tcvt.u32.u64 k, t1;",
        "\tst.global.u32 [qq], k;",
        "\tst.u32 [%4], m;", "\tst.f32 [%5], a;",
        "}",
    ]) + "\n"
    esc = asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return (
        'extern "C" __global__ void ic_multi(unsigned n, const unsigned* sq,\n'
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


class MultiEngine:
    def __init__(self, gpu, body: int):
        self.gpu = gpu
        self.nwarp = 16
        self.mod = Module(gpu, build_src(body), f"ic16_{body}.cu")
        self.offs = block_offsets(self.mod.sass("ic_multi"))
        ks = sorted(self.offs)
        self.gap = abs(self.offs[ks[1]] - self.offs[ks[0]]) if len(ks) > 1 else 0
        self.fn = self.mod.func("ic_multi").fn
        self.stride = PER_WARP * REPEAT + 4
        self.d_mark = gpu.malloc(4)
        self.d_sink = gpu.malloc(4)
        self.total_words = self.stride * 16
        self.d_sq = gpu.malloc(self.total_words * 4)
        self.d_out = gpu.malloc(self.total_words * 4)

    def run(self, warps: list[int], k_per_warp: int, reps: int) -> dict[int, float]:
        """每个 warp 访问自己那 8 块里的前 k 个，返回 {warp: 每步周期}。"""
        n = k_per_warp * REPEAT
        # 全部槽位预填哨兵：未列出的 warp 一进来就读到 0xFFFFFFFF 并退出，
        # 否则它们会反复执行块 0，污染被测代码的缓存状态。
        flat = [0xFFFFFFFF] * self.total_words
        for w in warps:
            seq = [w * PER_WARP + i for i in range(k_per_warp)] * REPEAT
            flat[w * self.stride: w * self.stride + n] = seq
        self.gpu.htod(self.d_sq, struct.pack(f"<{self.total_words}I", *flat))
        best: dict[int, float] = {}
        for _ in range(reps):
            self.gpu.memset(self.d_out, 0xFF, self.total_words * 4)
            args = [ctypes.c_uint(n), ctypes.c_uint64(self.d_sq), ctypes.c_uint64(self.d_out),
                    ctypes.c_uint64(self.d_mark), ctypes.c_uint64(self.d_sink),
                    ctypes.c_uint(self.stride)]
            arr = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in args])
            cuda.chk(cuda._F["cuLaunchKernel"](self.fn, 1, 1, 1, 32 * self.nwarp, 1, 1,
                                               0, 0, arr, None), "launch")
            self.gpu.sync()
            raw = struct.unpack(f"<{self.total_words}I", self.gpu.dtoh(self.d_out, self.total_words * 4))
            for w in warps:
                v = raw[w * self.stride] / n
                best[w] = min(best.get(w, 1e9), v)
        return best


def run(log=print, reps: int = 5) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    log("=" * 96)
    log("[Round 16] L0 / L1.5：多 warp 各跑独立代码区")
    log("=" * 96)

    # 间距 S 由 body 决定：S = (body+4)*16
    for body, S in ((4, 128), (28, 512), (60, 1024), (124, 2048), (252, 4096), (508, 8192)):
        eng = MultiEngine(gpu, body)
        log(f"\n  间距 {eng.gap} B（body={body}）  每 warp 最多 {PER_WARP} 块 = "
            f"{PER_WARP * eng.gap / 1024:.1f} KiB")
        log(f"     {'配置':<22} " + " ".join(f"K={k}" for k in (1, 2, 4, 6, 8)))
        for label, warps in (("单 warp(0)", [0]),
                             ("同分区 0/4", [0, 4]),
                             ("同分区 0/4/8/12", [0, 4, 8, 12]),
                             ("跨分区 0/1/2/3", [0, 1, 2, 3])):
            if len(warps) > 4:
                continue
            row = []
            for k in (1, 2, 4, 6, 8):
                per = eng.run(warps, k, reps)
                row.append(max(per.values()) if per else float("nan"))
            total_kib = len(warps) * 8 * eng.gap / 1024
            log(f"     {label:<22} " + " ".join(f"{v:6.1f}" for v in row)
                + f"   （8 块全开时合计 {total_kib:.1f} KiB）")
            out[f"{S}B:{label}"] = [round(v, 1) for v in row]
    return out
