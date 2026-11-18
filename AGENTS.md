# GPU 微架构逆向（E:\2026\4060-sm-microarch）

## 这个文件夹是什么

用微基准测量反推 **RTX 4060（AD107 / sm_89）的 SM 微架构**。
不是抄规格书：每条结论都要有「内核内实测周期 + SASS 反汇编证据 + 反证检验」。

主文档：`docs/FINAL-REPORT.md`（发布版，配 12 张代码生成的 SVG）。
自包含 HTML：`docs/final-report.html`。
方法：`docs/00-methodology.md`。证据链：`notes/inference-log.md`。下一步：`docs/03-roadmap.md`。

改完数据后重新出图/出报告：

```powershell
.\.venv\Scripts\python.exe docs\figures\make_figures.py   # 生成 12 张 SVG
.\.venv\Scripts\python.exe docs\make_html.py              # MD -> 自包含 HTML
```

## 怎么跑

```powershell
cd E:\2026\4060-sm-microarch
.\.venv\Scripts\python.exe smoke.py         # 链路自检
.\.venv\Scripts\python.exe run_all.py       # 全部轮次（0–13，约 16 分钟）
.\.venv\Scripts\python.exe run_all.py --round 1 2
.\.venv\Scripts\python.exe run_all.py --round 3 --tag mem
```

Round 3 会分配 64 MiB + 512 MiB + 256 MiB 三块显存，跑之前确认卡上余量够（8 GiB 卡没问题）。

结果落到 `results/<时间戳>[-tag]/`，含 `run.log` 与 `results.json`。
分析文档里引用数据时**必须写明结果目录**，否则无法复现。

## 环境事实（这几条决定了整个工程的做法）

- **没有 CUDA Toolkit，没有 cl/gcc/clang/cmake**，DSH 会话**非管理员**。
- CUDA 编译靠 `nvrtc64_120_0.dll`（从 pip wheel 手工解压到 `.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin`）。
- 驱动 API 靠 `harness/cuda.py` 里的**纯 ctypes** 绑定，不用 `cuda-python`。
- SASS 反汇编靠 `build/tools/` 下解压的 `cuobjdump.exe` / `nvdisasm.exe`（NVIDIA redist 独立 zip）。
- **这台机器的 pip 会卡死**（CPU 占满但几十分钟不动）。要装包就
  `Invoke-WebRequest` 下 wheel + `[System.IO.Compression.ZipFile]::ExtractToDirectory` 解压。
- **无法 `nvidia-smi -lgc` 锁频**（非管理员）。空闲态 285 MHz，满载实测 ~2.70–2.75 GHz 且会漂。

## 约定（写新探针时必须遵守）

1. **只信周期，不信纳秒**。用内核内 `%clock64`，不要用 CUDA event 或墙钟。
2. **差值法**：同一内核取 N1/N2 两个迭代数相减，消固定开销。
3. **一切吞吐按真实执行的指令条数归一**（不是源码条数），并且要**精确到指令类别**：
   `add.u32` 会被融合进 `IADD3`，而 MMA 会被插入 MOV/LDG 搬运指令——
   两类情形下"按总条数归一"都会算错。
4. **每条延迟/吞吐都要打印 SASS 证据行**，并在结果里存 `sass_*` 字段。
5. **内核参数只能是设备指针**（8 字节），配置常量在 JIT 编译期固化。
   这样 `cuLaunchKernel` 的参数编组永远只有 `void*` 数组，行为确定。
6. **预热 250 ms 再测**（`Bench.prime`），否则读到的是 P3 低频态。
7. 测量内核**禁止** `printf`、越界访问、`use_fast_math`；结果必须写回显存否则会被优化掉。

## 已踩过的坑（别再踩）

| 坑 | 现象 | 对策 |
|---|---|---|
| 常量输入被折叠 | 延迟测出 0（`a+=1`×320 被折成 `IADD3 tid,0x140`） | 操作数一律从显存读，编译期不可知 |
| 代数化简 | `xor` 两次自逆、`and` 幂等、`rcp∘rcp` 恒等 → `dSASS=0` | 换不可化简形式（环状/交替）或改用 SASS 归一 |
| 指令融合 | `add.u32`×320 变成 160 条 `IADD3`（三操作数） | 用 `cyc/SASS` 而不是 `cyc/src` |
| 指令替换 | `fma.rn.f16x2` 第三操作数为 0 → 变成 `HADD2` | 第三操作数取运行时值 |
| 非 ftz 的 MUFU 被展开 | `rcp/lg2/rsqrt.approx.f32` → 4~7 条指令 | 用 `.ftz` 变体 |
| sink 只引用部分累加器 | 其余链被当死代码删，测到的是延迟不是吞吐 | sink 必须消费**全部** NACC 个累加器 |
| **重复访存被合并/删除** | n 条相同 `ld.shared` → 只剩 1 条；写同一寄存器时前面全被删 | 每条结果都要消费（多累加器）+ SASS 条数校验 |
| **跳数不足** | 追逐的大 footprint 全塌到同一延迟（"DRAM 跟 L2 一样快"） | 跳数 ≥ 4 × footprint/128B |
| **等步长环别名** | 32 KiB 延迟比 64 KiB 还高 | 编号域整环 + 地址域双射混淆 |
| `min` 撞上坏点 | `ops/cyc` 出现 384、2621、52428 | 吞吐用中位数，并做物理上限过滤（>512 判坏点） |
| **按"总指令数"归一** | Tensor Core 吞吐被高估 1.5×（768 vs 真实 512）：编译器会为 MMA 插入 MOV/LDG 搬运指令，且条数随迭代增长 | 计数必须精确到指令类别（只数 `HMMA`/`QMMA`） |
| **常量初始化让源变成立即数** | `b = 1.000001f` 被折成立即数，`FFMA R7, R0, R7, 0.5`——"三源全不同"的实验直接失效 | 所有源都要从显存读入（运行时未知） |
| **SASS 取样取到初始化代码** | 取"第一条 FFMA"取到了 `bv + k*0.5f` 的编译结果 | 取函数**中段**的指令做证据 |
| **桌面干扰** | 同一次指针追逐测量 3 秒内差 3.4×（min 272 / max 930） | 一律取 min；关键实验用「多 SM 取最小」 |
| 传空指针/少传参数 | `CUDA_ERROR_ILLEGAL_ADDRESS` 污染上下文，或 `INVALID_VALUE` | 直接 launch 时也要凑满 3 个参数（cyc/in/out） |
| PowerShell 改写文件 | `Get-Content/Set-Content` 按 ANSI 解码，把中文变乱码 | 编辑文件只用 edit/write 工具，别用 pwsh 读写；读 JSON 也用 Python |
| 控制台中文乱码 | 运行输出是 `΢�ܹ�` | 运行前 `[Console]::OutputEncoding = [Text.Encoding]::UTF8` 且 `$env:PYTHONIOENCODING="utf-8"` |

## 已知未验证项（不要当成结论）

- **L2 容量**：Round 11 测得拐点夹在 **8–16 MiB**（249 → 458 周期，1.84×），
  推算可用容量约 **8–12 MiB**（规格 24 MiB）。但尺寸只能取 2 的幂（链构造限制），
  所以给不出精确值；且无法区分"桌面占用 / L2 分区 / 流式替换"三种原因。
- **桌面干扰已量化**：同一测量 3 秒内差 **3.4×**。任何单点延迟都必须取 min；
  本机无法独占（i5-13490F 无核显，4060 同时驱动两块屏），除非加一张亮机卡。
- **L1I**：容量 **54–64 KiB**（Round 12/13 两法一致），**预取深度 1 行**（Round 13）。
  **替换策略与 set 索引位未测出**：包里的 `seq` 族依赖 `brx.idx`（994 个内联目标），
  在 sm_89 + nvrtc 上必崩（`seq_len=0` 也崩，id 未越界、结构体布局已核对）。
  指令缓存延迟、取指带宽也未测；**L0I / L1.5 未分离**（只给了下界：2×16 KiB 内无限制）。
- **L1D 精确容量**只给了 32–64 KiB 区间。
- **共享内存 bank 带宽 32 字/周期是反推的**（无冲突时只到 16，那是 LSU 的 0.5 条/周期接受率）。
- **warp→调度器的定量映射**未拿到：≤8 warp 时启动同步（≤13 周期）已证明双峰来自
  "共享调度器"，但 per-warp 时间戳换算的 IPC 之和（6 warp 时 4.51）超过实测的 4.00 硬上限，
  所以"2+2+1+1"只能当定性解释。需要"统计发射槽"的新方法。
- **warp 发射节流**：Round 8 用极小 warp 体复测，32 warp 只跨 30 周期 → **未观察到节流**；
  Round 6 那次 807 周期的原因仍未查清（离群值？寄存器压力？）。
- **操作数供给机制**：Round 8 的统计拟合倾向"每 lane 2 次读"的端口模型（预测 85.3 vs 实测 78，
  差 9.4%），所有 bank 模型拟合都更差（18–64%）。但 9.4% 残差未解释，仍不能定论。
- **TLB 容量本身未测出**：Round 9 只证明 256 页 × 2 MiB（512 MiB 跨度）内零代价，
  再往上受显存（8 GiB）与页大小限制。
- `cp.async`/LDGSTS、常量/纹理路径、原子单元、DPX、64 KiB 页：未测。
- 稀疏的其它形状（m16n8k16 稀疏、稀疏 TF32/FP8）、MMA 与向量指令并发、
  多 CTA 聚合张量吞吐、INT8/INT4 的延迟：未测。
- **FP8 为何只有 INT8 的一半发射率**：只观察到现象，机制未查明。
- **FP32 的 128 只在"操作数可复用"时成立**，不要当成任何 FFMA 流的峰值。
- LOP3 精确吞吐（58.7 受折叠污染，推断 64）。
- IADD3 = 128/cyc 与公开资料冲突，需换探针复核。
- MUFU.RCP 延迟原理上不可测（`rcp∘rcp` 恒等）。
