# 复现指南

这台机器上**没有** CUDA Toolkit、没有任何 C/C++ 编译器、也不是管理员。
整个工程靠三个免安装组件跑起来：NVRTC（编译）、纯 ctypes 的驱动绑定（加载/启动）、
NVIDIA redist 的独立 zip（反汇编）。下面的脚本把这三样装好。

---

## 0. 前置条件

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（脚本用 PowerShell；其他平台需自行改写路径） |
| Python | 3.10+（**不需要**任何第三方包，只用标准库 + ctypes） |
| GPU | NVIDIA 卡 + 驱动（本工程在 RTX 4060 / sm_89 / 驱动 610.88 上完成） |
| 权限 | **不需要管理员** |
| 网络 | 需要一次下载（约 100 MB：nvrtc wheel + cuobjdump/nvdisasm） |

---

## 1. 一键准备工具链

```powershell
cd <仓库根目录>
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

它会做四件事：

1. 建 `.venv/`；
2. 从清华镜像下 `nvidia-cuda-nvrtc-cu12` 的 wheel，**当 zip 解压**到 site-packages
   （不走 `pip install`：本机 pip 在解析阶段会卡死，见 README「工具链」一节）；
3. 从 NVIDIA redist 下 `cuda_cuobjdump` 与 `cuda_nvdisasm` 的独立 zip，解压到 `build/tools/`；
4. 跑 `smoke.py` 自检：编译 → 加载 → 启动 → 反汇编，全链路打通才算成功。

预期输出：

```
device      : NVIDIA GeForce RTX 4060  sm_89  sm=24  l2=25165824
compile     : cubin=2728B ptx=635B log=''
launch      : OK  first8=[0, 7, 14, 21, 28, 35, 42, 49]
sass        : /*0000*/  MOV R1, c[0x0][0x28] ;
```

---

## 2. 跑全部测量

```powershell
.\.venv\Scripts\python.exe run_all.py            # 13 轮，约 14 分钟
.\.venv\Scripts\python.exe run_all.py --round 2  # 只跑某一轮
.\.venv\Scripts\python.exe run_all.py --round 3 --tag mem
```

结果写到 `results/<时间戳>-<tag>/`：

* `run.log` —— 人读的完整输出；
* `results.json` —— 机读，含每条结论的原始数值与 SASS 证据行。

---

## 3. 重新生成图表与 HTML 报告

```powershell
.\.venv\Scripts\python.exe docs\figures\make_figures.py   # 12 张 SVG
.\.venv\Scripts\python.exe docs\make_html.py              # 自包含 HTML
```

图表是**纯代码生成**的 SVG（`docs/figures/make_figures.py` 里一个小工具箱，
无第三方依赖），数据点写死在脚本顶部，便于对照 `results/` 里的原始 JSON 审阅。

---

## 4. Round 12 额外需要外部包

Round 12（L1I / L0I）复用了外部测试包 `icache_b300_pkg` 的**内核生成器**。
该包不随本仓库分发，需要自己放进去：

```
third_party/icache_b300_pkg/
    gen_kernels.py  icache_bench.cu  ...   <- 解压自 icache_b300_pkg.tgz
```

放好之后 `run_all.py --round 12` 会自动生成内核并编译。
**没有这个包时 Round 12 会打印提示并跳过，其余 12 轮不受影响。**

移植细节（为什么不用它的 `nvcc` 流程、结构体怎么按值传、为什么要截断头文件）
见 `third_party/icache_b300_pkg/PORTING.md`。

---

## 5. 已知会被环境影响的测量

* **无法锁频**（非管理员）：GPU 空闲态 285 MHz，满载实测 2.70–2.75 GHz 且会漂。
  所以全工程以**周期**为单位，不报纳秒。
* **无法独占 GPU**（本机 i5-13490F 无核显，4060 同时驱动两块屏）：
  同一测量在 3 秒内可差 **3.4×**。对策：重复取最小值；关键实验用「多 SM 取最小」。
  详见 `docs/10-noise-and-multism.md`。
* 若换到别的卡上跑，数值当然会变，但 `probes/` 里的方法与 `notes/inference-log.md`
  里的反驳逻辑仍然适用。

---

## 6. 目录速查

```
harness/    测量框架（ctypes + nvrtc，无第三方依赖）
probes/     13 个探针，round0..round12
docs/       13 篇分析文档 + 发布版报告（md + 自包含 html）+ figures/
notes/      inference-log.md：40 条结论的完整证据链
results/    每次运行的有效原始数据（失败的运行已剔除）
scripts/    bootstrap.ps1（装工具链）、run_all.ps1（跑测量）
```
