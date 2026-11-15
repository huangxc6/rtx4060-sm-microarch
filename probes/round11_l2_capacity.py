"""Round 11：L2 容量拐点（靠多 SM 取最小压干扰）。

问题
----
Round 3 的单 SM 版在 8 MiB 给 286、16 MiB 给 769——两者都远小于 24 MiB 的 L2 规格。
到底是"有效容量只有 ~10 MiB（桌面占了一半）"，还是"16 MiB 那次被干扰了"？

方法
----
用 Round 10 验证过的「多 SM 取最小」：24 个 CTA 共读同一区域、各跑独立链，
取跨 SM 最小值。然后扫 4/8/16/32/64 MiB：

* 若拐点在 8→16 MiB 之间 ⟹ 有效容量约 10 MiB，说明 L2 被桌面占掉一半；
* 若拐点在 16→32 MiB 之间 ⟹ 有效容量接近规格的 24 MiB。

限制
----
链的构造要求节点数是 2 的幂（否则无法保证"单一整环 + 地址双射模糊"），
所以尺寸只能是 2 的幂。这意味着拐点只能被**夹在相邻两个 2 的幂之间**，
不能给出精确容量。这是本轮的已知局限。
"""
from __future__ import annotations

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

from probes.round10_noise_multism import (_chase_kernel, _init_kernel,  # noqa: E402
                                         _mix_params)

SIZES = [22, 23, 24, 25, 26]      # 4, 8, 16, 32, 64 MiB
REPS = 5
ITERS_FACTOR = 4
ITERS_CAP = 1 << 21               # 最多 2M 跳（64 MiB 时 line 数已达 512K）


def _iters_for(k: int) -> tuple[int, int]:
    lines = (1 << k) // 128
    hi = min(ITERS_CAP, max(1024, ITERS_FACTOR * lines))
    return hi // 4, hi


def run(log=print, reps: int = REPS) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    out: dict = {"clock_ghz": round(ghz, 4), "sm_count": gpu.sm_count}

    src = PREAMBLE + "".join(_init_kernel(k) for k in SIZES)
    src += "".join(_chase_kernel(k, it, False) for k in SIZES for it in _iters_for(k))
    src += "".join(_chase_kernel(k, it, True) for k in SIZES for it in _iters_for(k))
    mod = Module(gpu, src, "l2cap.cu")

    buf = gpu.malloc(1 << 26)
    sink = gpu.malloc(8)
    gpu.memset(buf, 0, 1 << 26)
    res: dict = {}
    log("=" * 92)
    log("[Round 11] L2 容量拐点：单 SM vs 多 SM 取最小")
    log("=" * 92)
    log(f"  {'工作集':>10} {'跳数':>9} {'单 SM':>9} {'多SM min':>9} {'多SM 中位':>10} "
        f"{'多SM max':>9}")
    try:
        for k in SIZES:
            nodes, mask, a, shift = _mix_params(k)
            lo_it, hi_it = _iters_for(k)
            mod.func(f"ch_init_{k}").launch(min(512, max(1, nodes // 256)), 256,
                                            (0, buf, 0))
            gpu.sync()
            # 单 SM
            c_lo = min(b.measure1(mod.func(f"ch_{k}_{lo_it}_s"), 32, (buf, sink), reps))
            c_hi = min(b.measure1(mod.func(f"ch_{k}_{hi_it}_s"), 32, (buf, sink), reps))
            single = (c_hi - c_lo) / float(hi_it - lo_it)
            # 多 SM
            r_lo = b.measure(mod.func(f"ch_{k}_{lo_it}_m"), 24, 32, (buf, sink), reps)
            r_hi = b.measure(mod.func(f"ch_{k}_{hi_it}_m"), 24, 32, (buf, sink), reps)
            per = []
            for c in range(24):
                per.append(((min(row[c] for row in r_hi) - min(row[c] for row in r_lo))
                            / float(hi_it - lo_it)))
            per.sort()
            med = per[len(per) // 2]
            res[str(k)] = {
                "footprint_bytes": 1 << k,
                "single": round(single, 1),
                "multi_min": round(per[0], 1),
                "multi_median": round(med, 1),
                "multi_max": round(per[-1], 1),
            }
            log(f"  {1 << k:>10,} {hi_it:>9,} {single:>9.1f} {per[0]:>9.1f} "
                f"{med:>10.1f} {per[-1]:>9.1f}")
    finally:
        gpu.free(buf)
        gpu.free(sink)
    out["l2_capacity"] = res

    # 拐点判读
    log("")
    prev = None
    for k in SIZES:
        v = res[str(k)]["multi_min"]
        if prev is not None:
            ratio = v / prev[1]
            log(f"  {1 << prev[0]:>9,} → {1 << k:>9,}: {prev[1]:>6.0f} → {v:>6.0f} "
                f"（{ratio:.2f}×）" + ("   <<< 明显上台阶" if ratio > 1.5 else ""))
        prev = (k, v)
    return out
