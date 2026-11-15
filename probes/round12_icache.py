"""Round 12：L1I / L0I（指令缓存）—— 移植自 icache_b300_pkg。

来源
----
`third_party/icache_b300_pkg/` 是一套面向 B300（sm_103）的指令缓存微基准，
需要 nvcc + Linux + 独占 sm_90+ 卡，本机都满足不了。但它的内核是**纯 device 代码**
（inline PTX 算子 + clock64 计时 + 一个按值传的 BenchArgs），所以这里用本工程的
nvrtc 链路重新驱动：内核生成器照用，host 编排换成 Python。

测什么
------
A. `fp`  家族：代码足迹 4 KiB → 512 KiB 扫描 → **L1I 容量**（每指令周期数的台阶）
B. `dual` 家族：同一/不同代码放在**同一/不同调度器分区**上 → **L0I 容量与分区归属**

B 的做法依据：Round 4/6 已确认 warp 4k 与 warp 4k+4 落在同一个调度器，
所以"同分区争用"要用 warp 0 与 warp 4；不同分区用 warp 0 与 warp 1。

注意
----
* 结构体参数按 ABI **按值**传递（168 字节），这是本移植的关键；
* 生成的头文件末尾含 host 侧名册（引用 KernelInfo/KIND_*），nvrtc 只编 device 代码，
  必须切掉；
* 本机无法独占 GPU，所以每个配置重复多次取最小值。
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import bench as bench_mod  # noqa: E402
from harness import cuda, env  # noqa: E402
from harness.jit import Module  # noqa: E402

PKG = ROOT / "third_party" / "icache_b300_pkg"
BUILD = PKG / "build"
HEADER = BUILD / "kernels_gen.cuh"

TARGET_INSTR = 1 << 18      # 每次启动约 26 万条指令的计时窗口
WARMUP = 4
REPS = 5

SHIM = r"""
#define kBlockThreads 512

struct BenchArgs {
    long long* t_start;
    long long* t_end;
    float* sink;
    unsigned long long roles;
    int warmup;
    int iters;
    int touch_const;
    int n_chunks;
    int flag;
    int bar_threads;
    float b;
    float c;
    const unsigned* seq;
    unsigned* seq_out;
    int seq_len;
    const int* ph_warp;
    const int* ph_off;
    const int* ph_len;
    int n_phases;
    int* xsm;
    const int* bw_body;
    long long* bw_out;
    const unsigned* bw_data;
    const unsigned* bw_dstart;
    int bw_dwords;
    int bw_dstride;
};

__constant__ float c_arr[60 * 1024 / 4];

__device__ __forceinline__ void record(const BenchArgs& args, unsigned warp,
                                       long long t0, long long t1, float acc) {
    if (args.touch_const) acc += c_arr[threadIdx.x];
    if ((threadIdx.x & 31u) == 0u) {
        args.t_start[warp] = t0;
        args.t_end[warp] = t1;
    }
    args.sink[threadIdx.x] = acc;
}

"""


class BenchArgs(ctypes.Structure):
    _fields_ = [
        ("t_start", ctypes.c_void_p), ("t_end", ctypes.c_void_p),
        ("sink", ctypes.c_void_p), ("roles", ctypes.c_uint64),
        ("warmup", ctypes.c_int), ("iters", ctypes.c_int),
        ("touch_const", ctypes.c_int), ("n_chunks", ctypes.c_int),
        ("flag", ctypes.c_int), ("bar_threads", ctypes.c_int),
        ("b", ctypes.c_float), ("c", ctypes.c_float),
        ("seq", ctypes.c_void_p), ("seq_out", ctypes.c_void_p),
        ("seq_len", ctypes.c_int),
        ("ph_warp", ctypes.c_void_p), ("ph_off", ctypes.c_void_p),
        ("ph_len", ctypes.c_void_p), ("n_phases", ctypes.c_int),
        ("xsm", ctypes.c_void_p), ("bw_body", ctypes.c_void_p),
        ("bw_out", ctypes.c_void_p), ("bw_data", ctypes.c_void_p),
        ("bw_dstart", ctypes.c_void_p),
        ("bw_dwords", ctypes.c_int), ("bw_dstride", ctypes.c_int),
    ]


def ensure_generated(log=print) -> bool:
    """确保生成的内核存在。外部包不在时返回 False（调用方跳过本轮）。"""
    if not (PKG / "gen_kernels.py").exists():
        log("  !! 找不到 third_party/icache_b300_pkg/gen_kernels.py —— 本轮跳过。")
        log("     见 third_party/icache_b300_pkg/PORTING.md（该外部包不随仓库分发）。")
        return False
    if HEADER.exists():
        head = HEADER.read_text(encoding="utf-8", errors="replace")[:400000]
        if "dual_0004k" in head:
            return True
    log("  生成内核（gen_kernels.py --families fp,dual）…")
    try:
        subprocess.run([sys.executable, str(PKG / "gen_kernels.py"),
                        "--out-dir", str(BUILD), "--families", "fp,dual"],
                       check=True, cwd=str(PKG),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        log(f"  !! 生成失败（{exc}）—— 本轮跳过。")
        return False
    return True


def device_source() -> str:
    h = HEADER.read_text(encoding="utf-8")
    cut = h.find("static const KernelInfo kKernels[]")
    return SHIM + (h[:cut] if cut > 0 else h)


def roster() -> list[tuple[str, str, int, int]]:
    """从名册里取 (名字, kind, code_kb, nominal_instr)。"""
    out = []
    h = HEADER.read_text(encoding="utf-8")
    import re
    for m in re.finditer(r'\{"(\w+)",\s*\w+,\s*KIND_(\w+),\s*(\d+),\s*(\d+)', h):
        out.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    return out


class IcacheBench:
    def __init__(self, gpu, mod):
        self.gpu = gpu
        self.mod = mod
        self.t_start = gpu.malloc(16 * 8)
        self.t_end = gpu.malloc(16 * 8)
        self.sink = gpu.malloc(512 * 4)
        self.args = BenchArgs()
        self.args.t_start = self.t_start
        self.args.t_end = self.t_end
        self.args.sink = self.sink
        self.args.flag = 1
        self.args.b = 1.000001
        self.args.c = 0.5

    def run(self, name: str, roles: int, iters: int, warmup: int = WARMUP) -> dict:
        k = self.mod.func(name)
        self.args.roles = roles
        self.args.iters = iters
        self.args.warmup = warmup
        self.gpu.memset(self.t_start, 0, 16 * 8)
        argptr = ctypes.cast(ctypes.byref(self.args), ctypes.c_void_p)
        arr = (ctypes.c_void_p * 1)(argptr)
        cuda.chk(cuda._F["cuLaunchKernel"](k.fn, 1, 1, 1, 512, 1, 1, 0, 0, arr, None),
                 f"launch {name}")
        self.gpu.sync()
        raw_t0 = self.gpu.dtoh(self.t_start, 16 * 8)
        raw_t1 = self.gpu.dtoh(self.t_end, 16 * 8)
        per = {}
        for w in range(16):
            t0 = ctypes.c_int64.from_buffer_copy(raw_t0[w * 8:(w + 1) * 8]).value
            t1 = ctypes.c_int64.from_buffer_copy(raw_t1[w * 8:(w + 1) * 8]).value
            if t1 > t0 > 0:
                per[w] = t1 - t0
        return per


def run(log=print, reps: int = REPS) -> dict:
    gpu = env.Gpu()
    b = bench_mod.Bench(gpu, log=log)
    b.prime(250)
    out: dict = {}

    log("=" * 96)
    log("[Round 12] L1I / L0I（移植自 icache_b300_pkg，内核用 nvrtc 重新编译到 sm_89）")
    log("=" * 96)
    if not ensure_generated(log):
        return {"skipped": "外部包 third_party/icache_b300_pkg 不存在"}
    mod = Module(gpu, device_source(), "icache.cu")
    log(f"  编译完成: cubin={len(mod.cubin):,} B  sizeof(BenchArgs)={ctypes.sizeof(BenchArgs)}")
    bench = IcacheBench(gpu, mod)
    table = roster()

    # ---------- A. fp：代码足迹 → L1I 容量 ----------
    log("")
    log("  A. 代码足迹扫描（单 warp，按名义指令数归一）")
    log(f"  {'内核':<12} {'足迹':>8} {'iters':>7} {'周期/指令':>10} {'SASS 条数':>9}")
    fp_res: dict = {}
    for name, kind, code_kb, ninstr in table:
        if kind != "FP":
            continue
        iters = max(16, TARGET_INSTR // ninstr)
        per = {}
        for _ in range(reps):
            r = bench.run(name, roles=0x1, iters=iters)
            if 0 in r:
                per[0] = min(per.get(0, 1 << 62), r[0])
        if 0 not in per:
            continue
        cpi = per[0] / (iters * ninstr)
        n_sass = sum(1 for ln in mod.sass(name).splitlines()
                     if ln.strip().startswith("/*") and "*/" in ln)
        fp_res[name] = {"code_kb": code_kb, "cpi": round(cpi, 4),
                        "sass_instr": n_sass, "iters": iters}
        log(f"  {name:<12} {code_kb:>6} KiB {iters:>7} {cpi:>10.3f} {n_sass:>9}")
    out["fp"] = fp_res

    # ---------- B. dual：同/不同分区 × 同/不同代码 ----------
    log("")
    log("  B. L0I：同一调度器分区上放同一份 vs 两份不同代码")
    log("     （warp0 与 warp4 同分区；warp0 与 warp1 不同分区）")
    l0_res: dict = {}
    configs = [
        ("warp0 单独跑 A",            0x1, [0]),
        ("warp0+1 同代码（不同分区）", 0x11, [0, 1]),
        ("warp0+1 不同代码（不同分区）", 0x21, [0, 1]),
        ("warp0+4 同代码（同分区）",   0x1_0001, [0, 4]),
        ("warp0+4 不同代码（同分区）", 0x2_0001, [0, 4]),
    ]
    for code_kb in (4, 8, 16):
        name = f"dual_{code_kb:04d}k"
        if name not in [t[0] for t in table]:
            continue
        ninstr = next(t[3] for t in table if t[0] == name)
        iters = max(16, TARGET_INSTR // ninstr)
        rows = []
        for label, roles, warps in configs:
            best = {}
            for _ in range(reps):
                r = bench.run(name, roles=roles, iters=iters)
                for w in warps:
                    if w in r:
                        best[w] = min(best.get(w, 1 << 62), r[w])
            if not best:
                continue
            cpis = {w: best[w] / (iters * ninstr) for w in best}
            rows.append((label, cpis))
            l0_res[f"{code_kb}k:{label}"] = {str(w): round(v, 3) for w, v in cpis.items()}
        log("")
        log(f"    dual_{code_kb:04d}k（每份代码 {code_kb} KiB）")
        for label, cpis in rows:
            txt = "  ".join(f"w{w}={v:.3f}" for w, v in cpis.items())
            log(f"      {label:<26} {txt}")
    out["l0"] = l0_res

    gpu.free(bench.t_start)
    gpu.free(bench.t_end)
    gpu.free(bench.sink)
    return out
