"""最小链路自检：nvrtc 编译 -> cubin -> 驱动加载 -> 启动 -> 读回 -> SASS。

任何一环坏掉，后面的微基准都不成立，所以先用这个脚本把链路跑通。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import bench, env, jit  # noqa: E402

SRC = r"""
extern "C" __global__ void fill(unsigned long long* out, const void* in, void* o2) {
    out[threadIdx.x] = threadIdx.x * 7 + blockIdx.x;
}
"""


def main() -> int:
    env.prepare()
    gpu = env.Gpu()
    d = gpu.describe()
    print(f"device      : {d['name']}  {d['arch']}  sm={d['sm_count']}  l2={d['l2_bytes']}")
    print(f"driver      : {d['driver_version']}   nvrtc: {d['nvrtc_version']}")

    m = jit.Module(gpu, SRC, "smoke.cu")
    print(f"compile     : cubin={len(m.cubin)}B ptx={len(m.ptx)}B log={m.log.strip()!r}")
    k = m.func("fill")
    print(f"kernel      : {k} attrs={ {kk: vv for kk, vv in k.attrs().items() if kk in ('num_regs','max_threads_per_block','binary_version')} }")

    n = 64
    buf = gpu.malloc(n * 8)
    try:
        k.launch(1, n, (buf, 0, 0))
        gpu.sync()
        vals = bench.u64(gpu.dtoh(buf, n * 8))
    finally:
        gpu.free(buf)
    expect = [i * 7 for i in range(n)]
    got = list(vals)
    print(f"launch      : {'OK' if got == expect else 'MISMATCH'}  first8={got[:8]}")

    sass = m.sass("fill")
    lines = [ln for ln in sass.splitlines() if "/*" in ln and "*/" in ln]
    print("sass        :")
    for ln in lines[:12]:
        print("   " + ln.strip())
    return 0 if got == expect else 1


if __name__ == "__main__":
    raise SystemExit(main())
