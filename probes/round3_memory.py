"""Round 3：存储层次。

子实验
------
3.1 指针追逐：延迟 vs 工作集（4 KiB → 64 MiB）。台阶位置 = 各级容量。
    关键设计：链节点存的是**下一个节点的完整 64 位地址**，于是追逐步是
    单条 ``ld.global.ca.b64 %0, [%0]``——地址计算不在关键路径上，
    测到的就是纯 load-to-use 延迟（没有加法的 1~2 周期污染）。
3.2 共享内存 bank：固定 warp、扫地址步长，吞吐台阶 → bank 数/宽度/广播。
    另测 LDS 依赖链延迟（同样是单条 ``ld.shared.b32 %0, [%0]``）。
3.3 DRAM 带宽：512 MiB 流式读/写，墙钟计时，与驱动标称 272 GB/s 对照。
3.4 sector 尺寸：固定 footprint、变大跨步读，useful 带宽的第一个膝盖就是
    DRAM 的 sector 粒度（预期 32 B）。

为什么 3.3/3.4 用墙钟
---------------------
周期计数器是**每 SM 独立**的，跨全部 SM 的聚合带宽没法用 clock64 表示；
而这两个内核都是毫秒级，~50 µs 的启动抖动只占百分之几。
"""
from __future__ import annotations

import time

from harness import bench as bench_mod
from harness import env
from harness.jit import Module

PREAMBLE = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
"""

# ---- 3.1 指针追逐 --------------------------------------------------------
CHASE_K = list(range(12, 27))     # footprint = 2^k 字节，4 KiB .. 64 MiB
CG_K = [16, 20, 24]               # 只对这几个尺寸额外做 .cg（绕 L1）对照
CHASE_NODE = 8                    # 每个节点 8 字节（存 64 位地址）
CHASE_REPS = 10                   # 单 warp 追逐极易被桌面合成抢占，多取几次最小值


def _iters_for(k: int) -> tuple[int, int]:
    """每个尺寸的跳数。

    这是初版最大的坑：跳数决定了**实际碰到的 cache line 数**。
    当初固定 1024 跳时，即使名义 footprint 是 64 MiB，也只碰到 1024 条 line
    （≈128 KiB），于是 L1 正好被填满、每条 line 每轮都驱逐，所有大尺寸
    都塌缩到同一个 ~200 周期的 L2 值——看上去"DRAM 和 L2 一样快"。
    正确做法：跳数 ≈ 4 倍 line 数（每条 line 被访问约 4 次，覆盖率足够）。
    """
    lines = max(16, (1 << k) // 128)
    hi = max(1024, 4 * lines)
    return hi // 4, hi


def _chain_a(nodes: int) -> int:
    """链的步长（节点数）：奇数 → 在 2 的幂大小的环上是一个完整置换。"""
    return max(3, int(nodes * 0.6180339887)) | 1


def _init_kernel(k: int) -> str:
    """把链写进显存：节点 i 位于地址 g(i)，值为节点 (i+A) 的地址。

    两个约束必须同时满足，否则测出来的曲线会被假象污染：

    * **必须是单一整环**：否则链路只会绕行环的一部分，实际 footprint 小于名义值。
      做法是在**节点编号域**里做 ``i -> i+A``（A 为奇数、n 为 2 的幂 ⇒ 整环）。
    * **地址必须伪随机**：否则等步长会在 cache 的 set 上产生别名冲突，
      小 footprint 也能跑出大延迟（实测初版：32 KiB 的延迟比 64 KiB 还高）。
      做法是让节点 i 存放于地址 ``g(i)``，g 是一个**双射**的混淆函数：
      ``t = (i*C) & mask;  g = t ^ (t >> shift)``（奇数乘 + xorshift 都是双射）。
      g 作用在地址上 ⇒ 访问序列是乱序的；i -> i+A 作用在编号上 ⇒ 仍是整环。
    """
    nodes = 1 << (k - 3)
    a = _chain_a(nodes)
    mask = nodes - 1
    shift = max(1, (k - 3) // 2)
    c = 0x9E3779B1
    return f"""
extern "C" __global__ void ch_init_{k}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned long long* p = (unsigned long long*)in;
    unsigned int nthr = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < {nodes}u; i += nthr) {{
        unsigned int t0 = (i * {c}u) & {mask}u;
        unsigned int g0 = t0 ^ (t0 >> {shift});
        unsigned int t1 = (((i + {a}u) & {mask}u) * {c}u) & {mask}u;
        unsigned int g1 = t1 ^ (t1 >> {shift});
        p[g0] = base + 8ull * (unsigned long long)g1;
    }}
}}
"""


def _chase_kernel(k: int, iters: int, op: str) -> str:
    return f"""
extern "C" __global__ void ch_{op}_{k}_{iters}(unsigned long long* cyc, const void* in,
                                               void* out) {{
    unsigned long long addr = (unsigned long long)in;
    unsigned long long t0, t1;
    CLK(t0);
    #pragma unroll 1
    for (int i = 0; i < {iters}; ++i) {{
        asm volatile("ld.global.{op}.b64 %0, [%0];" : "+l"(addr));
    }}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.b64 [%0], %1;" :: "l"(out), "l"(addr));
}}
"""


# ---- 3.2 共享内存 --------------------------------------------------------
LDS_STRIDES = [0, 1, 2, 3, 4, 8, 16, 17, 31, 32, 33, 64]
# 每条 LDS 占一条不同偏移；N 上限由"16 KiB 内能放下多少个 +128B 偏移"决定
# （128 个 → 16 KiB）。N 太小的话差值会被 clock64 的几十周期噪声淹没。
LDS_N_LO, LDS_N_HI = 32, 128
LDS_LAT_A, LDS_LAT_NODES = 97, 1024
LDS_LAT_LO, LDS_LAT_HI = 64, 256
LDS_PEAK_LO, LDS_PEAK_HI = 8, 32     # 32 个独立目标寄存器 ⇒ 最多 32 条不重复的载入


def _lds_tp_kernel(stride: int, n: int) -> str:
    """n 条独立 LDS，每条用**不同立即数偏移**（+j*128 字节）并累加进 8 个累加器之一。

    两个坑都要绕开（都已用 diag_smem.py 验证）：

    1. 若 n 条语句完全相同（同地址 + 同目标寄存器），ptxas 的"多余载入消除"
       会把它们合并成 1 条 `LDS`。
    2. 即使地址不同，只要**都写同一个目标寄存器**，前面的结果就是死值，
       ptxas 会直接删掉（实测 128 条 → 只剩 1 条）。所以每条的结果必须被消费。

    处理办法：8 个累加器（把加法依赖链打散，64 周期 ≪ LDS 时间），
    并用 **8 个 warp**（256 线程）保证发射能力足够（每 2 条指令 1 次载入，
    4 IPC 下可达 2 条载入/周期 > LDS 的 1 条/周期上限，于是瓶颈仍在 LDS）。

    偏移取 128 字节整数倍 ⇒ +32 字 ⇒ **bank 编号不变**，冲突度对所有指令一致。
    """
    body = "\n    ".join(
        f'asm volatile("ld.shared.b32 %0, [%1+{j * 128}];" : "=r"(t) : "r"(addr));'
        f' v{j % 8} += t;'
        for j in range(n)
    )
    return f"""
extern "C" __global__ void lds_tp_{stride}_{n}(unsigned long long* cyc, const void* in,
                                               void* out) {{
    __shared__ unsigned int sm[8192];
    sm[threadIdx.x] = threadIdx.x;
    asm volatile("bar.sync 0;");
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned int addr = (unsigned int)sbase
        + (((threadIdx.x * {stride}u) & 2047u) * 4u);
    unsigned int v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0, t = 0;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "r"(v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7));
}}
"""


def _lds_peak_kernel(n: int) -> str:
    """直接测共享内存峰值：每条 LDS 写**自己的**目标寄存器，计时区里没有任何加法。

    前一版探针每条载入配一条加法（为了让结果不被当死值删掉），结果无冲突时只测到
    16 字/周期——那是探针的发射上限，不是硬件上限。这里改成 32 个独立目标寄存器
    r0..r31，归约（32 个 XOR）放在计时区之外，于是计时区里只剩纯 LDS。
    """
    loads = "\n    ".join(
        f'asm volatile("ld.shared.b32 %0, [%1+{j * 512}];" : "=r"(r{j}) : "r"(addr));'
        for j in range(n)
    )
    zerofill = "\n    ".join(f"r{j} = 0;" for j in range(n, 32))
    sink = " ^ ".join(f"r{j}" for j in range(32))
    return f"""
extern "C" __global__ void lds_peak_{n}(unsigned long long* cyc, const void* in, void* out) {{
    __shared__ unsigned int sm[8192];
    sm[threadIdx.x] = threadIdx.x;
    asm volatile("bar.sync 0;");
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned int addr = (unsigned int)sbase + ((threadIdx.x & 2047u) * 4u);
    unsigned int r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
                 r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29,
                 r30, r31;
    {zerofill}
    unsigned long long t0, t1;
    CLK(t0);
    {loads}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4),
                 "r"({sink}));
}}
"""


def _lds_lat_kernel(iters: int) -> str:
    body = "\n    ".join(['asm volatile("ld.shared.b32 %0, [%0];" : "+r"(addr));'] * iters)
    return f"""
extern "C" __global__ void lds_lat_{iters}(unsigned long long* cyc, const void* in,
                                           void* out) {{
    __shared__ unsigned int sm[{LDS_LAT_NODES}];
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned int base32 = (unsigned int)sbase;
    for (int i = 0; i < {LDS_LAT_NODES}; ++i)
        sm[i] = base32 + 4u * (((unsigned int)i + {LDS_LAT_A}u) & {LDS_LAT_NODES - 1}u);
    asm volatile("bar.sync 0;");
    unsigned int addr = base32;
    unsigned long long t0, t1;
    CLK(t0);
    {body}
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"(out), "r"(addr));
}}
"""


# ---- 3.3 带宽 -----------------------------------------------------------
BW_BYTES = 512 << 20
BW_NVEC = BW_BYTES // 16
BW_GRID, BW_BLOCK = 96, 256


def _bw_read_kernel() -> str:
    return f"""
extern "C" __global__ void bw_read(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    // 每个线程 4 条独立累加链：避免加法依赖把循环速度限制在 4 周期/迭代
    for (unsigned int i = tid; i < {BW_NVEC}u; i += stride) {{
        unsigned long long a = base + (unsigned long long)i * 16ull;
        unsigned int r0, r1, r2, r3;
        asm volatile("ld.global.v4.u32 {{%0,%1,%2,%3}}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(a));
        a0 += r0; a1 += r1; a2 += r2; a3 += r3;
    }}
    asm volatile("st.global.u32 [%0], %1;" :: "l"(out), "r"(a0 ^ a1 ^ a2 ^ a3));
}}
"""


def _bw_write_kernel() -> str:
    return f"""
extern "C" __global__ void bw_write(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int v = tid * 2654435761u + 1u;
    for (unsigned int i = tid; i < {BW_NVEC}u; i += stride) {{
        unsigned long long a = base + (unsigned long long)i * 16ull;
        asm volatile("st.global.v4.u32 [%0], {{%1,%1,%1,%1}};" :: "l"(a), "r"(v));
    }}
    asm volatile("st.global.u32 [%0], %1;" :: "l"(out), "r"(v));
}}
"""


# ---- 3.4 sector 尺寸 ----------------------------------------------------
SECTOR_FOOTPRINT = 256 << 20     # 固定 footprint
SECTOR_STRIDES = [1, 2, 4, 8, 16, 32, 64, 128, 256]   # 单位：word(4B)


def _sector_kernel(stride: int) -> str:
    nwords = SECTOR_FOOTPRINT // (4 * stride)
    return f"""
extern "C" __global__ void sec_{stride}(unsigned long long* cyc, const void* in, void* out) {{
    unsigned long long base = (unsigned long long)in;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nthr = gridDim.x * blockDim.x;
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (unsigned int i = tid; i < {nwords}u; i += nthr) {{
        unsigned long long a = base + (unsigned long long)i * {4 * stride}ull;
        unsigned int r0;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(r0) : "l"(a));
        a0 += r0;
    }}
    asm volatile("st.global.u32 [%0], %1;" :: "l"(out), "r"(a0 ^ a1 ^ a2 ^ a3));
}}
"""


def _time_launches(b, kernel, grid, block, pts, launches: int = 8,
                   groups: int = 3) -> float:
    """多次启动取最优，返回单次墙钟秒数。用于短内核的带宽测量。"""
    best = float("inf")
    for _ in range(groups):
        t0 = time.perf_counter()
        for _ in range(launches):
            kernel.launch(grid, block, pts)
        b.gpu.sync()
        best = min(best, (time.perf_counter() - t0) / launches)
    return best


def run(log=print, reps: int = 7) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    ghz = b.measure_clock_ghz()
    b.prime(250)
    facts = gpu.describe()
    out: dict = {"clock_ghz": round(ghz, 4)}

    # ================= 3.1 指针追逐 =================
    src = PREAMBLE + "".join(_init_kernel(k) for k in CHASE_K)
    src += "".join(_chase_kernel(k, it, "ca") for k in CHASE_K
                   for it in _iters_for(k))
    src += "".join(_chase_kernel(k, it, "cg") for k in CG_K
                   for it in _iters_for(k))
    mod = Module(gpu, src, "chase.cu")
    log("=" * 92)
    log("[Round 3.1] 指针追逐：延迟 vs 工作集（每个尺寸跑满≈4 倍 line 数）")
    log("=" * 92)

    buf = gpu.malloc(1 << 26)
    sink = gpu.malloc(8)
    gpu.memset(buf, 0, 1 << 26)
    pts = (buf, sink)
    chase: dict = {}
    try:
        log(f"  {'footprint':>10} {'跳数':>10} {'ca':>8} {'cg':>8}   ns(ca)   SASS")
        for k in CHASE_K:
            nodes = 1 << (k - 3)
            lo_it, hi_it = _iters_for(k)
            grid = max(1, min(512, (nodes + 255) // 256))
            mod.func(f"ch_init_{k}").launch(grid, 256, (0, buf, 0))
            gpu.sync()
            sass_mid = ""
            vals = {}
            for op in ("ca", "cg"):
                if op == "cg" and k not in CG_K:
                    continue
                k_lo = mod.func(f"ch_{op}_{k}_{lo_it}")
                k_hi = mod.func(f"ch_{op}_{k}_{hi_it}")
                c_lo = min(b.measure1(k_lo, 32, pts, CHASE_REPS))
                c_hi = min(b.measure1(k_hi, 32, pts, CHASE_REPS))
                vals[op] = (c_hi - c_lo) / float(hi_it - lo_it)
                if op == "ca":
                    for ln in mod.sass(f"ch_ca_{k}_{hi_it}").splitlines():
                        s = ln.strip()
                        if s.startswith("/*") and "*/" in s:
                            txt = s.split("*/", 1)[1].split(";")[0].strip()
                            if txt.startswith("LDG"):
                                sass_mid = txt
                                break
            ns = vals["ca"] / ghz
            cg_s = f"{vals['cg']:.2f}" if "cg" in vals else "-"
            chase[str(k)] = {
                "footprint_bytes": 1 << k,
                "hops": hi_it,
                "lat_ca": round(vals["ca"], 2),
                "lat_cg": round(vals["cg"], 2) if "cg" in vals else None,
                "ns_ca": round(ns, 1),
                "sass": sass_mid,
            }
            log(f"  {1 << k:>10,} {hi_it:>10,} {vals['ca']:>8.2f} {cg_s:>8}   "
                f"{ns:>6.1f}   {sass_mid}")
    finally:
        gpu.free(buf)
        gpu.free(sink)
    out["chase"] = chase

    # ================= 3.2 共享内存 =================
    log("")
    log("=" * 92)
    log("[Round 3.2] 共享内存：bank 冲突扫描（1 warp = 32 线程）")
    log("=" * 92)
    src = PREAMBLE
    src += "".join(_lds_tp_kernel(s, n) for s in LDS_STRIDES
                   for n in (LDS_N_LO, LDS_N_HI))
    src += "".join(_lds_lat_kernel(it) for it in (LDS_LAT_LO, LDS_LAT_HI))
    src += "".join(_lds_peak_kernel(n) for n in (LDS_PEAK_LO, LDS_PEAK_HI))
    mod2 = Module(gpu, src, "smem.cu")
    smem_out: dict = {}
    outbuf = gpu.malloc(4096)
    gpu.memset(outbuf, 0, 4096)
    try:
        log(f"  {'stride(字)':>10} {'理论 bank 度':>12} {'words/cyc':>10}  {'SASS'}")
        for s in LDS_STRIDES:
            k_lo = mod2.func(f"lds_tp_{s}_{LDS_N_LO}")
            k_hi = mod2.func(f"lds_tp_{s}_{LDS_N_HI}")
            c_lo = min(b.measure1(k_lo, 256, (outbuf, outbuf), reps))
            c_hi = min(b.measure1(k_hi, 256, (outbuf, outbuf), reps))
            d = c_hi - c_lo
            dn = LDS_N_HI - LDS_N_LO
            words = 256.0 * dn / d if d else float("nan")
            degree = 32 // _gcd(s % 32, 32) if s % 32 else 32
            smem_out[str(s)] = {"words_per_cycle": round(words, 2),
                                "expected_bank_degree": degree}
            log(f"  {s:>10} {degree:>12} {words:>10.2f}")
        # 证据：确认 SASS 里真的有 N 条 LDS（否则说明又被 ptxas 合并了）
        sass1 = mod2.sass(f"lds_tp_1_{LDS_N_HI}")
        nlds = sum(1 for ln in sass1.splitlines()
                   if ln.strip().startswith("/*") and "*/" in ln and "LDS" in ln)
        smem_out["sass_lds_count"] = nlds
        log(f"  证据: lds_tp_1_{LDS_N_HI} 的 SASS 含 {nlds} 条 LDS（期望 {LDS_N_HI}）")
    finally:
        gpu.free(outbuf)

    # LDS 延迟
    k_lo = mod2.func(f"lds_lat_{LDS_LAT_LO}")
    k_hi = mod2.func(f"lds_lat_{LDS_LAT_HI}")
    dummy = gpu.malloc(8)
    try:
        c_lo = min(b.measure1(k_lo, 32, (dummy, dummy), reps))
        c_hi = min(b.measure1(k_hi, 32, (dummy, dummy), reps))
        lds_lat = (c_hi - c_lo) / float(LDS_LAT_HI - LDS_LAT_LO)
    finally:
        gpu.free(dummy)
    smem_out["lds_latency_cycles"] = round(lds_lat, 2)
    log(f"  LDS 依赖链延迟 = {lds_lat:.2f} 周期")

    # 峰值：计时区里只有纯 LDS，没有配对的加法
    kp_lo = mod2.func(f"lds_peak_{LDS_PEAK_LO}")
    kp_hi = mod2.func(f"lds_peak_{LDS_PEAK_HI}")
    pkbuf = gpu.malloc(2048)
    gpu.memset(pkbuf, 0, 2048)
    try:
        c_lo = min(b.measure1(kp_lo, 256, (pkbuf, pkbuf), reps))
        c_hi = min(b.measure1(kp_hi, 256, (pkbuf, pkbuf), reps))
        d = c_hi - c_lo
        peak = 256.0 * (LDS_PEAK_HI - LDS_PEAK_LO) / d if d > 0 else float("nan")
        sass_pk = mod2.sass(f"lds_peak_{LDS_PEAK_HI}")
        n_lds = sum(1 for ln in sass_pk.splitlines()
                    if ln.strip().startswith("/*") and "*/" in ln and "LDS" in ln)
    finally:
        gpu.free(pkbuf)
    smem_out["peak_words_per_cycle"] = round(peak, 2)
    smem_out["peak_sass_lds_count"] = n_lds
    log(f"  峰值（纯 LDS，无配对加法）= {peak:.2f} 字/周期  → "
        f"{peak * 4:.0f} B/周期   [SASS 含 {n_lds} 条 LDS]")

    out["smem"] = smem_out

    # ================= 3.3 带宽 =================
    log("")
    log("=" * 92)
    log(f"[Round 3.3] DRAM 带宽（{BW_BYTES >> 20} MiB 流式，墙钟计时）")
    log("=" * 92)
    src = PREAMBLE + _bw_read_kernel() + _bw_write_kernel()
    mod3 = Module(gpu, src, "bw.cu")
    bbuf = gpu.malloc(BW_BYTES)
    bout = gpu.malloc(8)
    bw: dict = {}
    try:
        kr = mod3.func("bw_read")
        kw = mod3.func("bw_write")
        pts3 = (bout, bbuf, bout)
        # 预热：首次触碰显存页表，不能计入时间
        kr.launch(BW_GRID, BW_BLOCK, pts3)
        kw.launch(BW_GRID, BW_BLOCK, pts3)
        gpu.sync()
        tr = _time_launches(b, kr, BW_GRID, BW_BLOCK, pts3, launches=1, groups=5)
        tw = _time_launches(b, kw, BW_GRID, BW_BLOCK, pts3, launches=1, groups=5)
        bw = {
            "read_gbs": round(BW_BYTES / tr / 1e9, 1),
            "write_gbs": round(BW_BYTES / tw / 1e9, 1),
            "read_ms": round(tr * 1e3, 3),
            "write_ms": round(tw * 1e3, 3),
            "derived_nominal_gbs": facts["derived_mem_bandwidth_gbs"],
        }
        log(f"  读: {tr * 1e3:.2f} ms -> {bw['read_gbs']:.1f} GB/s")
        log(f"  写: {tw * 1e3:.2f} ms -> {bw['write_gbs']:.1f} GB/s")
        log(f"  驱动标称推算: {facts['derived_mem_bandwidth_gbs']:.1f} GB/s")
    finally:
        gpu.free(bbuf)
        gpu.free(bout)
    out["bandwidth"] = bw

    # ================= 3.4 sector 尺寸 =================
    log("")
    log("=" * 92)
    log(f"[Round 3.4] sector 尺寸：固定 footprint {SECTOR_FOOTPRINT >> 20} MiB，变大跨步")
    log("=" * 92)
    src = PREAMBLE + "".join(_sector_kernel(s) for s in SECTOR_STRIDES)
    mod4 = Module(gpu, src, "sector.cu")
    sbuf = gpu.malloc(SECTOR_FOOTPRINT)
    sout = gpu.malloc(8)
    sector: dict = {}
    try:
        log(f"  {'stride(B)':>10} {'读字数':>12} {'墙钟 ms':>9} {'useful GB/s':>12}")
        for s in SECTOR_STRIDES:
            k = mod4.func(f"sec_{s}")
            pts4 = (sout, sbuf, sout)
            k.launch(BW_GRID, BW_BLOCK, pts4)
            gpu.sync()
            t = _time_launches(b, k, BW_GRID, BW_BLOCK, pts4, launches=32, groups=3)
            nwords = SECTOR_FOOTPRINT // (4 * s)
            useful = nwords * 4 / t / 1e9
            sector[str(s * 4)] = {"bytes_stride": s * 4, "words_read": nwords,
                                  "ms": round(t * 1e3, 4),
                                  "useful_gbs": round(useful, 1)}
            log(f"  {s * 4:>10} {nwords:>12,} {t * 1e3:>9.3f} {useful:>12.1f}")
    finally:
        gpu.free(sbuf)
        gpu.free(sout)
    out["sector"] = sector

    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
