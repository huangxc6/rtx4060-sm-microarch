# Round 0：设备事实与观测基线

数据来源：`results/20261115-210955-final/results.json` 的 `round0` 段。
每一行都标出**是谁说的**——驱动属性、硬件自报寄存器、还是我们实测。

## 1. 身份与驱动

| 项 | 值 | 来源 |
|---|---|---|
| 名称 | NVIDIA GeForce RTX 4060 | `cuDeviceGetName` |
| 计算能力 | 8.9（AD107，Ada Lovelace） | `cuDeviceGetAttribute` |
| 编译目标 | `sm_89` | 由计算能力推得 |
| PCI | 0000:01:00 | `cuDeviceGetAttribute` |
| 显存 | 8,585,216,000 B = 8188 MiB | `cuDeviceTotalMem` |
| 驱动 | 13,030（= CUDA 13.3 UMD，KMD 610.88） | `cuDriverGetVersion` |
| NVRTC | 12.9 | `nvrtcVersion` |
| 驱动模型 | WDDM（非独占） | `nvidia-smi` |

> NVRTC 12.9 编译 `sm_89` 的 cubin 能被 CUDA 13.3 的驱动加载，说明 NVRTC 产物向下兼容。
> 这也意味着本工程不依赖 CUDA 13 的工具链。

## 2. SM 顶层参数（三方交叉验证）

| 项 | 驱动属性 | 硬件自报 | 实测 | 结论 |
|---|---|---|---|---|
| SM 数量 | 24 | `%nsmid` = 24 | `%smid` 出现 24 个不同取值（0..23） | **24**，三方一致 |
| warp 宽度 | 32 | — | 每个 CTA 的 `%laneid` 落在 0..31 | **32** |
| 每 SM warp 上限 | 1536/32 = 48 | `%nwarpid` = 48 | 占用率表见下 | **48** |
| 每 SM 线程上限 | 1536 | — | — | **1536** |
| 每 SM CTA 上限 | 24 | — | 占用率表见下（block=32 时卡在 24） | **24** |

`%smid` 的 24 个取值是硬件给出的物理 SM 编号，不经过驱动；它和 `cuDeviceGetAttribute`
的 `MULTIPROCESSOR_COUNT` 完全一致，因此可以放心把「24」当已知量用于后续换算
（GPU 全局吞吐 = 24 × 单 SM 吞吐）。

512 个 CTA 的落点分布为每 SM 21~22 个，说明 CTA 分发是硬件轮询、基本均匀。

## 3. 存储与寄存器

| 项 | 值 | 备注 |
|---|---|---|
| 每 SM 寄存器 | 65,536 × 32 bit = **256 KiB** | `MAX_REGISTERS_PER_MULTIPROCESSOR` |
| 每线程寄存器上限 | 65,536（受 block 上限约束） | 实际由占用率决定 |
| 每 SM 共享内存 | 102,400 B = **100 KiB** | `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` |
| 每 CTA 共享内存 | 49,152 B = 48 KiB（静态） | `MAX_SHARED_MEMORY_PER_BLOCK` |
| L2 容量 | 25,165,824 B = **24 MiB** | `L2_CACHE_SIZE` |
| 显存位宽 | 128 bit | `GLOBAL_MEMORY_BUS_WIDTH` |
| 标称显存频率 | 8,501,000 kHz = 8.5 GHz（等效 17 Gbps） | `MEMORY_CLOCK_RATE` |
| 由此推得带宽 | 2 × 8.5e9 × 128/8 = **272 GB/s** | 仅驱动标称值推算，**待 Round 3 实测** |

> 100 KiB 而不是 128 KiB：Ada 的 `sm_89` 每 SM 物理上有 128 KiB 统一 L1/smem，
> 但驱动只允许 100 KiB 用于 smem（其余留给 L1）。这个数字是**驱动上限**，
> 实际可分配量要用 Round 3 的容量扫描确认。

## 4. 占用率上限（驱动计算，不是实测）

空内核、0 动态 smem 时，每个 block 大小能驻留多少 CTA：

| block_size | CTA/SM | 线程/SM | warp/SM | 卡在哪个上限 |
|---:|---:|---:|---:|---|
| 32 | 24 | 768 | 24 | **CTA 数（24）** |
| 64 | 24 | 1536 | 48 | warp 数 |
| 128 | 12 | 1536 | 48 | warp 数 |
| 192 | 8 | 1536 | 48 | warp 数 |
| 256 | 6 | 1536 | 48 | warp 数 |
| 384 | 4 | 1536 | 48 | warp 数 |
| 512 | 3 | 1536 | 48 | warp 数 |
| 768 | 2 | 1536 | 48 | warp 数 |
| 1024 | 1 | 1024 | 32 | 每 CTA 线程上限 1024 |
| 1536 | 0 | — | — | 超过每 CTA 线程上限 |

**一个容易被忽略的推论**：如果只用 32 线程的小 CTA，即使 SM 能容纳 48 个 warp，
也永远只能跑到 24 个 warp（768 线程）——因为 CTA 数先到上限了。
要打满 48 warp，每 CTA 至少 64 线程。

## 5. 实测 SM 频率（内核内采样）

因为非管理员无法 `nvidia-smi -lgc` 锁频，驱动报的 `clock_rate`（2.46 GHz）只是标称值；
GPU 实际在 P3 空闲态会掉到 285 MHz。用内核内 `%globaltimer`（纳秒）与 `%clock64`（周期）
同步采样，得到预热后的瞬时频率：

| 场景 | 实测 | nvidia-smi 标称 |
|---|---|---|
| 250 ms 满载预热后 | **2.7466 GHz** | 2.4600 GHz |
| 紧接着复测 | **2.7016 GHz** | — |

结论：

1. 这块卡**会跑到标称之上**（约 +11%），所以用 2.46 GHz 算理论算力会低估。
2. 频率在两次测量间就漂了 ~1.6%，**任何纳秒级结论都不可复现**——这正是全部改用周期的原因。
3. 后续所有换算都注明所用的频率。按实测 2.7466 GHz：
   - FP32 峰值 = 24 SM × 128 lane × 2 flop × 2.7466 GHz = **16.87 TFLOPS**
   - SFU 峰值 = 24 × 16 = 384 ops/cycle → 1.055 Tops/s
   - FP64 峰值 = 24 × 2 = 48 ops/cycle → 0.132 Tflop/s（是 FP32 的 1/64）

## 6. 本轮的"待验证"

- 272 GB/s 是驱动标称值推算，未实测。
- 100 KiB smem 是驱动上限，实际可分配量与 L1/smem 划分未实测。
- 24 CTA/SM 的上限只由驱动占用率 API 给出，未用实测反证（Round 2 的 warp 扫描
  用的是单 CTA，没有触及 CTA 上限）。
