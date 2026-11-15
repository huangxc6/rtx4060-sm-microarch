"""诊断脚本：为什么 lds_tp 的差值是 0？chase 的 SASS 到底长什么样？"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import bench, env  # noqa: E402
from harness.jit import Module  # noqa: E402

SRC = r"""
#define CLK(x) asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));

// A: 64 位地址
extern "C" __global__ void tp_u64(unsigned long long* cyc, const void* in, void* out) {
    __shared__ unsigned int sm[4096];
    sm[threadIdx.x] = threadIdx.x;
    asm volatile("bar.sync 0;");
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned long long addr = sbase + (unsigned long long)((threadIdx.x & 4095u) * 4u);
    unsigned int v = 0;
    unsigned long long t0, t1;
    CLK(t0);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"(addr));
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(v));
}

// B: 32 位 shared 地址
extern "C" __global__ void tp_u32(unsigned long long* cyc, const void* in, void* out) {
    __shared__ unsigned int sm[4096];
    sm[threadIdx.x] = threadIdx.x;
    asm volatile("bar.sync 0;");
    unsigned long long sbase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sbase) : "l"(&sm[0]));
    unsigned int addr = (unsigned int)sbase + (threadIdx.x & 4095u) * 4u;
    unsigned int v = 0;
    unsigned long long t0, t1;
    CLK(t0);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    CLK(t1);
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"((char*)out + threadIdx.x * 4), "r"(v));
}
"""


def main():
    env.prepare()
    gpu = env.Gpu()
    b = bench.Bench(gpu)
    m = Module(gpu, SRC, "diag.cu")
    out = gpu.malloc(4096)
    gpu.memset(out, 0, 4096)
    for name in ("tp_u64", "tp_u32"):
        k = m.func(name)
        sass = m.sass(name)
        lines = [ln.strip() for ln in sass.splitlines()
                 if ln.strip().startswith("/*") and "*/" in ln]
        lds = [ln for ln in lines if "LDS" in ln]
        print(f"--- {name}: 共 {len(lines)} 条指令, 其中 LDS {len(lds)} 条 ---")
        for ln in lines:
            txt = ln.split("*/", 1)[1].split(";")[0].strip()
            print("   ", txt)
        vals = b.measure1(k, 32, (out, out), 3)
        print(f"    cycles per rep = {vals}")
    gpu.free(out)


if __name__ == "__main__":
    main()
