"""Round 0：设备事实 + 主时钟 + 硬件自报的 SM 普查 + 占用率上限。

这一轮不产出"逆向结论"，只建立**可信的观测基线**：
* 驱动声称的规格（可能是营销数字，也可能是保守值）；
* 硬件在指令里自报的规格（``%smid`` / ``%nsmid`` / ``%nwarpid``，不经过驱动）；
* 内核内实测的 SM 频率（无法锁频，nvidia-smi 的采样值不能用来换算周期）。

三者互相印证的部分，后续各轮才有资格当"已知量"。
"""
from __future__ import annotations

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

CENSUS_SRC = r"""
// %smid 是硬件给出的 SM 编号，%nsmid 是硬件自报的 SM 总数。
// 用一次覆盖全部 SM 的网格，看实际出现了多少个不同 SM、以及 warp 编号如何分布。
extern "C" __global__ void census(unsigned long long* cyc, const void* in,
                                  unsigned long long* out) {
    unsigned int smid, nsmid, warpid, nwarpid, laneid;
    asm volatile("mov.u32 %0, %%smid;"    : "=r"(smid));
    asm volatile("mov.u32 %0, %%nsmid;"   : "=r"(nsmid));
    asm volatile("mov.u32 %0, %%warpid;"  : "=r"(warpid));
    asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(nwarpid));
    asm volatile("mov.u32 %0, %%laneid;"  : "=r"(laneid));
    if (threadIdx.x == 0) {
        unsigned long long* p = out + blockIdx.x * 8;
        p[0] = smid;
        p[1] = nsmid;
        p[2] = warpid;
        p[3] = nwarpid;
        p[4] = laneid;
        p[5] = blockIdx.x;
        p[6] = gridDim.x;
        p[7] = blockDim.x;
    }
}
"""

OCC_SRC = r"""
extern "C" __global__ void touch(unsigned long long* cyc, const void* in,
                                 unsigned long long* out) {
    if (threadIdx.x == 0xFFFFFFFFu) cyc[0] = 1;
}
"""


def run(log=print, reps: int = 7) -> dict:
    gpu = env.Gpu()
    facts = gpu.describe()
    out: dict = {"facts": facts}

    log("=" * 80)
    log(f"[Round 0]  {facts['name']}   {facts['arch']}   PCI {facts['pci']}")
    log(f"           driver={facts['driver_version']}  nvrtc={facts['nvrtc_version']}")
    log("=" * 80)

    human_units = {
        "smem_per_sm": "B", "smem_per_block": "B", "l2_bytes": "B",
        "total_mem_bytes": "B", "max_regs_per_sm": "个", "max_regs_per_block": "个",
        "nominal_sm_clock_khz": "kHz", "nominal_mem_clock_khz": "kHz",
        "bus_width_bits": "bit", "derived_mem_bandwidth_gbs": "GB/s",
    }
    for k, v in facts.items():
        unit = human_units.get(k, "")
        if k in ("total_mem_bytes", "l2_bytes", "smem_per_sm", "smem_per_block"):
            shown = f"{v:,} ({v / 1024:.0f} KiB)"
        elif k == "derived_mem_bandwidth_gbs":
            shown = f"{v:.1f}"
        elif isinstance(v, int) and not isinstance(v, bool):
            shown = f"{v:,}"
        else:
            shown = str(v)
        log(f"  {k:<28} = {shown} {unit}")

    b = bench_mod.Bench(gpu, log=log)

    # ---- 硬件自报普查 --------------------------------------------------
    census = Module(gpu, CENSUS_SRC, "census.cu").func("census")
    grid, block = 512, 32
    buf = gpu.malloc(grid * 8 * 8)
    try:
        for _ in range(3):
            census.launch(grid, block, (0, 0, buf))
        gpu.sync()
        vals = bench_mod.u64(gpu.dtoh(buf, grid * 8 * 8))
    finally:
        gpu.free(buf)
    recs = [tuple(vals[i * 8:(i + 1) * 8]) for i in range(grid)]
    smids = sorted({r[0] for r in recs})
    nsmid_hw = recs[0][1]
    nwarpid_hw = recs[0][3]

    # 每个 SM 上落了多少个不同 warp 编号
    out["census"] = {
        "distinct_smid": len(smids),
        "smid_values": smids,
        "nsmid_reported": int(nsmid_hw),
        "nwarpid_reported": int(nwarpid_hw),
        "blocks_per_smid": {int(s): sum(1 for r in recs if r[0] == s) for s in smids},
    }
    log("")
    log(f"  %nsmid 硬件自报           = {nsmid_hw}")
    log(f"  %smid 实测不同取值数      = {len(smids)}  (取值 {smids[0]}..{smids[-1]})")
    log(f"  %nwarpid 自报             = {nwarpid_hw}")
    log(f"  驱动 multiprocessor_count = {facts['sm_count']}")
    log(f"  每次启动 512 CTA 时，各 SM 落到的 CTA 数 = "
        f"{out['census']['blocks_per_smid']}")

    # ---- 内核内实测 SM 频率 --------------------------------------------
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    ghz2 = b.measure_clock_ghz()
    out["clock"] = {
        "ghz_after_prime": ghz,
        "ghz_second": ghz2,
        "nominal_ghz": facts["nominal_sm_clock_khz"] / 1e6,
    }
    log("")
    log(f"  实测 SM 频率（预热后）    = {ghz:.4f} GHz   (复测 {ghz2:.4f} GHz)")
    log(f"  驱动标称 clock_rate       = {facts['nominal_sm_clock_khz'] / 1e6:.4f} GHz")

    # ---- 占用率上限（驱动计算，不是实测）-------------------------------
    touch = Module(gpu, OCC_SRC, "occ.cu").func("touch")
    rows = []
    for bs in (32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536):
        nb = gpu.occupancy_blocks(touch.fn, bs, 0)
        rows.append((bs, nb, None if nb is None else nb * bs))
    out["occupancy"] = rows
    log("")
    log("  占用率上限（空内核，驱动计算）：block_size -> 每 SM 驻留 CTA / 线程 / warp")
    for bs, nb, th in rows:
        if nb is None:
            log(f"    {bs:>5}  -> 非法")
        else:
            log(f"    {bs:>5}  -> {nb:>3} CTA  {th:>5} 线程  {th // facts['warp_size']:>3} warp")

    return out
