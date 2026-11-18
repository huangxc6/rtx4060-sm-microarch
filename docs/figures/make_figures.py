"""生成发布版报告的全部 SVG 图。

数据来源（写死在下面，便于审阅时对照）：
  A = results/20261115-221820-final/results.json   （Round 0–10 全量）
  B = results/20261115-223836/results.json         （Round 11 L2 容量）

风格：浅色、单一调色板、Segoe UI + 微软雅黑。
运行：python docs/figures/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---- 调色板 ----
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BG = "#ffffff"
BAND = "#f9fafb"
BLUE = "#2563eb"
AMBER = "#d97706"
GREEN = "#059669"
RED = "#dc2626"
PURPLE = "#7c3aed"
TEAL = "#0891b2"
FONT = "Segoe UI, Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"
MONO = "Consolas, Cascadia Mono, monospace"


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts: list[str] = []

    def add(self, s: str):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=0, sw=1, op=1):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'opacity="{op}"/>')

    def text(self, x, y, s, size=13, anchor="start", fill=INK, weight="400",
             mono=False, op=1):
        fam = MONO if mono else FONT
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
                 f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
                 f'opacity="{op}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1, dash=None, op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')

    def poly(self, pts, stroke, sw=2.2, fill="none", dash=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-linejoin="round"{d}/>')

    def circle(self, x, y, r, fill, stroke="none", sw=1):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def path(self, d, fill="none", stroke=INK, sw=1):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def render(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" font-family="{FONT}">')
        return "\n".join([head, f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>',
                          *self.parts, "</svg>"])


def save(name: str, svg: Svg):
    p = OUT / name
    p.write_text(svg.render(), encoding="utf-8")
    print(f"  {name}  {len(p.read_bytes()) / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# 通用坐标轴
# ---------------------------------------------------------------------------
def axes(s: Svg, box, xlab="", ylab="", ymax=1.0, ymin=0.0, yticks=5,
         yfmt=lambda v: f"{v:g}", xpos=None):
    x0, y0, x1, y1 = box
    s.rect(x0, y0, x1 - x0, y1 - y0, fill=BG)
    if yticks:
        for i in range(yticks + 1):
            v = ymin + (ymax - ymin) * i / yticks
            y = y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
            s.line(x0, y, x1, y, GRID, 1)
            s.text(x0 - 10, y + 4, yfmt(v), 11, "end", MUTED)
    if xpos:
        for x, lab in xpos:
            s.text(x, y1 + 18, lab, 11, "middle", MUTED)
    if xlab:
        s.text((x0 + x1) / 2, y1 + 42, xlab, 12, "middle", MUTED)
    if ylab:
        s.add(f'<text x="{x0 - 52}" y="{(y0 + y1) / 2:.1f}" font-family="{FONT}" '
              f'font-size="12" fill="{MUTED}" text-anchor="middle" '
              f'transform="rotate(-90 {x0 - 52} {(y0 + y1) / 2:.1f})">{esc(ylab)}</text>')


def header(s: Svg, title: str, sub: str = ""):
    s.text(36, 38, title, 19, weight="600")
    if sub:
        s.text(36, 60, sub, 12, fill=MUTED)


def footer(s: Svg, txt: str):
    s.text(s.w - 24, s.h - 14, txt, 10.5, "end", MUTED)


# ===========================================================================
# Fig 1：SM 结构示意
# ===========================================================================
def fig1():
    W, H = 980, 560
    s = Svg(W, H)
    header(s, "图 1  SM 结构与存储层次（据实测重建）",
           "所有数字都来自本工程的实测；括号内是驱动/规格声称但本工程未能独立验证的部分")

    # 外框
    s.rect(36, 82, 600, 300, "#f8fafc", "#cbd5e1", 10)
    s.text(52, 106, "一个 SM（全卡共 24 个）", 13, weight="600")

    pipes = [("FP32（FMA）", 32, BLUE), ("INT 加（IADD3）", 32, BLUE),
             ("INT 乘（IMAD）", 16, TEAL), ("移位 / 逻辑", 16, TEAL),
             ("SFU（MUFU）", 4, PURPLE), ("FP64", 0.5, AMBER)]
    for i in range(4):
        x = 52 + i * 145
        s.rect(x, 118, 132, 28, "#eef2ff", "#c7d2fe", 6)
        s.text(x + 66, 137, f"调度器 {i}", 12, "middle", weight="600")
        s.text(x + 66, 152, "1 条 IPC", 10, "middle", MUTED)
        for j, (nm, lanes, col) in enumerate(pipes):
            y = 162 + j * 34
            frac = max(0.06, lanes / 32)
            s.rect(x, y, 132, 26, "#ffffff", "#e5e7eb", 4)
            s.rect(x, y, 132 * frac, 26, col, "none", 4, op=0.18)
            s.text(x + 6, y + 17, nm, 10.5)
            s.text(x + 126, y + 17, f"{lanes:g}", 10.5, "end", col, "600")

    s.text(52, 374, "每调度器每周期发 1 条 warp 指令 ⟹ SM 总发射上限 4 IPC（实测 4.00）", 11.5, fill=MUTED)

    # 共享资源条
    s.rect(36, 396, 600, 66, "#fffbeb", "#fde68a", 10)
    items = [("寄存器文件", "64K × 32 bit = 256 KiB"), ("共享内存", "100 KiB（驱动上限）"),
             ("warp / 线程 / CTA", "48 / 1536 / 24")]
    for i, (a, b) in enumerate(items):
        x = 56 + i * 196
        s.text(x, 420, a, 11.5, weight="600")
        s.text(x, 440, b, 11, fill=MUTED)

    # 存储层次
    levels = [
        ("L1 数据", "32–64 KiB 可用", "37 周期", BLUE),
        ("L2", "可用约 8–12 MiB（规格 24）", "240–249 周期", TEAL),
        ("DRAM", "8 GiB GDDR6 / 272 GB/s", "679–750 周期", AMBER),
    ]
    for i, (nm, cap, lat, col) in enumerate(levels):
        y = 96 + i * 108
        s.rect(700, y, 244, 84, "#ffffff", col, 10, 1.6)
        s.rect(700, y, 6, 84, col, "none", 3)
        s.text(716, y + 26, nm, 14, weight="600", fill=col)
        s.text(716, y + 48, cap, 11.5, fill=MUTED)
        s.text(716, y + 68, "命中延迟 " + lat, 12, weight="600")
        if i < 2:
            s.line(822, y + 84, 822, y + 108, col, 2)
            s.add(f'<polygon points="822,{y + 112} 817,{y + 104} 827,{y + 104}" fill="{col}"/>')

    s.text(700, 404, "每级都是「命中延迟」，单 warp 依赖链实测；", 11, fill=MUTED)
    s.text(700, 422, "多 SM 取最小、跨 SM 的最小值。", 11, fill=MUTED)
    footer(s, "来源：Round 0/1/2/3/10/11   工程：E:\\2026\\4060-sm-microarch")
    save("fig1-sm-structure.svg", s)


# ===========================================================================
# Fig 2：FFMA 吞吐 vs warp 数（4 调度器的证据）
# ===========================================================================
def fig2():
    W, H = 900, 470
    s = Svg(W, H)
    header(s, "图 2  4 个 warp 调度器：FFMA 吞吐随 warp 数的台阶",
           "单 CTA、单 SM、每线程 256 条互相独立的 FFMA；4 warp 时恰好饱和到 128 ops/cycle")
    data = [(1, 32.0), (2, 63.5), (4, 127.0), (6, 95.3), (8, 126.0),
            (12, 125.1), (16, 126.5), (24, 126.5), (32, 127.6)]
    box = (80, 86, 860, 386)
    xs = [d[0] for d in data]
    xpos = []
    for w in [1, 2, 4, 8, 16, 32]:
        x = box[0] + (box[2] - box[0]) * (w ** 0.5 - 1) / (32 ** 0.5 - 1)
        xpos.append((x, str(w)))
    axes(s, box, "warp 数（√ 轴）", "FFMA ops / cycle / SM", ymax=150, ymin=0,
         yticks=6, xpos=xpos)

    def X(w):
        return box[0] + (box[2] - box[0]) * (w ** 0.5 - 1) / (32 ** 0.5 - 1)

    def Y(v):
        return box[3] - (box[3] - box[1]) * v / 150

    s.line(box[0], Y(128), box[2], Y(128), RED, 1.4, "6 4")
    s.text(box[2] - 40, Y(128) - 18, "理论峰值 128 = 4 调度器 × 32 lane", 11, "end", RED)
    s.line(box[0], Y(32), box[2], Y(32), BLUE, 1.2, "4 4")
    s.poly([(X(w), Y(v)) for w, v in data], BLUE, 2.6)
    for w, v in data:
        s.circle(X(w), Y(v), 4.5, BG if w == 6 else BLUE, BLUE, 2)
        s.text(X(w), Y(v) - 12, f"{v:.0f}", 11, "middle", BLUE, "600")
    # 标注
    s.text(X(1) + 12, Y(32) + 22, "1 warp = 32 ⟹ 每周期 1 条 warp 指令", 11, fill=BLUE)
    s.text(X(4) + 10, Y(127) - 26, "4 warp = 127 ⟹ 饱和", 11, fill=RED, weight="600")
    s.text(X(6) - 4, Y(95) + 26, "6 warp 下探到 3 IPC\n（调度器负载不均）", 11, "middle", MUTED)
    footer(s, "来源：Round 2（results/20261115-221820-final）")
    save("fig2-scheduler-staircase.svg", s)


# ===========================================================================
# Fig 3：各流水线宽度
# ===========================================================================
def fig3():
    W, H = 900, 430
    s = Svg(W, H)
    header(s, "图 3  各流水线的并行通道数（每 SM、每周期）",
           "warp 数扫描的饱和值；括号里是每调度器的通道数")
    rows = [
        ("FP32  FMA", 127.6, BLUE, "32 / 调度器"),
        ("INT 加  IADD3", 128.0, BLUE, "32 / 调度器"),
        ("INT 乘  IMAD", 67.8, TEAL, "16 / 调度器"),
        ("FP16x2  HFMA2", 67.9, TEAL, "16 / 调度器（每条 2 个 FP16）"),
        ("移位  SHF", 64.0, TEAL, "16 / 调度器"),
        ("逻辑  LOP3", 58.7, TEAL, "≈16（受可折叠性影响）"),
        ("SFU  MUFU", 16.0, PURPLE, "4 / 调度器"),
        ("FP64  DFMA", 2.0, AMBER, "0.5 / 调度器（全 SM 2 个）"),
    ]
    x0, x1 = 200, 830
    vmax = 128.0
    for i, (nm, v, col, note) in enumerate(rows):
        y = 92 + i * 38
        s.text(x0 - 12, y + 16, nm, 12, "end")
        s.rect(x0, y, x1 - x0, 22, "#f3f4f6", "none", 4)
        s.rect(x0, y, max(2, (x1 - x0) * v / vmax), 22, col, "none", 4, op=0.85)
        s.text(x0 + 8, y + 16, f"{v:.0f}", 11.5, weight="600", fill="#ffffff")
        s.text(x1 + 8, y + 16, note, 10.5, fill=MUTED)
    s.line(x0 + (x1 - x0) * 128 / vmax, 86, x0 + (x1 - x0) * 128 / vmax, 400, RED, 1.2, "5 4")
    s.text(x0 + (x1 - x0) * 128 / vmax + 6, 82, "128", 11, fill=RED)
    footer(s, "来源：Round 2。注意 FP32 的 128 需要操作数可复用，见 图 5")
    save("fig3-pipeline-widths.svg", s)


# ===========================================================================
# Fig 4：指令延迟
# ===========================================================================
def fig4():
    W, H = 900, 450
    s = Svg(W, H)
    header(s, "图 4  指令延迟（单 warp 串行依赖链，单位：周期）",
           "差值法消固定开销；每条都附 SASS 证据确认真实指令")
    rows = [
        ("FFMA / FADD / FMUL", 4.0, BLUE, "fma.rn.f32"),
        ("IMAD（INT32 乘加）", 4.0, BLUE, "mad.lo.u32"),
        ("IADD3（INT32 加）", 4.0, BLUE, "被 ptxas 两两融合，按 SASS 归一"),
        ("SHF（移位）", 4.0, BLUE, "shl.b32"),
        ("HFMA2（FP16x2）", 4.0, BLUE, "fma.rn.f16x2"),
        ("F2I / I2F（转换）", 11.5, TEAL, "cvt 往返，每条分摊"),
        ("MUFU（EX2/LG2/RSQ）", 17.0, PURPLE, "ftz 变体，避免被展开成多条"),
        ("MMA（Tensor Core）", 40.6, GREEN, "HMMA.16816，D→C 依赖链"),
        ("DADD（FP64）", 48.0, AMBER, "add.rn.f64"),
    ]
    x0, x1 = 240, 800
    vmax = 50.0
    for i, (nm, v, col, note) in enumerate(rows):
        y = 92 + i * 36
        s.text(x0 - 12, y + 15, nm, 12, "end")
        s.rect(x0, y, max(2, (x1 - x0) * v / vmax), 20, col, "none", 4, op=0.85)
        s.text(x0 + 8, y + 15, f"{v:g}", 11.5, weight="600", fill="#ffffff")
        s.text(x1 + 8, y + 15, note, 10.5, fill=MUTED)
    footer(s, "来源：Round 1 / Round 5（results/20261115-221820-final）")
    save("fig4-instruction-latency.svg", s)


# ===========================================================================
# Fig 5：操作数供给（78 vs 125）
# ===========================================================================
def fig5():
    W, H = 900, 430
    s = Svg(W, H)
    header(s, "图 5  「FP32 峰值 128」的前提：操作数必须可复用",
           "同一形状的 FFMA，只改源寄存器的复用关系；无溢出、指令数相同")
    rows = [
        ("三源全不同", 78.0, RED, "FFMA R23, R21, R23, R16"),
        ("两源相同", 125.1, GREEN, "FFMA R9, R7.reuse, R14, R7"),
        ("三源同一", 125.1, GREEN, "FFMA R0, R0, R0, R0"),
        ("跨 3 取（两源）", 125.1, GREEN, "FFMA R17, R0, R5, R0"),
    ]
    x0, x1 = 190, 640
    for i, (nm, v, col, sass) in enumerate(rows):
        y = 100 + i * 62
        s.text(x0 - 12, y + 20, nm, 12.5, "end", weight="600")
        s.rect(x0, y, (x1 - x0) * v / 128, 26, col, "none", 5, op=0.85)
        s.text(x0 + 8, y + 19, f"{v:.1f} ops/cyc", 12, weight="600", fill="#ffffff")
        s.text(x0, y + 44, sass, 10.5, fill=MUTED, mono=True)
    s.line(x0 + (x1 - x0) * 128 / 128, 92, x0 + (x1 - x0) * 128 / 128, 350, MUTED, 1.2, "5 4")
    s.text(x0 + (x1 - x0) * 128 / 128 + 6, 88, "128", 11, fill=MUTED)
    s.text(x0, 392, "代价 1.60×：每条指令需要 3 次寄存器读，而寄存器堆每 lane 每周期只供得起约 2 次。", 12)
    s.text(x0, 410, "（bank 冲突模型全部拟合失败：2 bank 预测 64、4/8 bank 预测 93、16/32 bank 预测 128）", 10.5, fill=MUTED)
    footer(s, "来源：Round 7 / Round 8")
    save("fig5-operand-supply.svg", s)


# ===========================================================================
# Fig 6：存储层次阶梯
# ===========================================================================
def fig6():
    W, H = 920, 500
    s = Svg(W, H)
    header(s, "图 6  存储层次：延迟 vs 工作集",
           "日志横轴；蓝线=多 SM 取最小（降噪后），灰线=单 SM 单次（受桌面干扰）")
    # (footprint, single, multi)
    data = [
        (4096, 37.0, None), (8192, 48.7, None), (16384, 95.9, None), (32768, 154.6, None),
        (65536, 228.7, 178.5), (262144, 271.5, 210.1), (1048576, 323.7, 210.6),
        (2097152, 483.0, 233.4), (4194304, 507.8, 240.0), (8388608, 471.8, 249.3),
        (16777216, 699.3, 458.4), (33554432, 732.4, 577.0), (67108864, 742.3, 678.8),
    ]
    box = (86, 88, 870, 400)
    import math

    def X(b):
        lo, hi = math.log2(4096), math.log2(67108864)
        return box[0] + (box[2] - box[0]) * (math.log2(b) - lo) / (hi - lo)

    def Y(v):
        return box[3] - (box[3] - box[1]) * v / 800

    def yfmt(v):
        return f"{v:g}"

    xpos = []
    for b, lab in [(4096, "4K"), (65536, "64K"), (1048576, "1M"),
                   (16777216, "16M"), (67108864, "64M")]:
        xpos.append((X(b), lab))
    axes(s, box, "工作集（字节）", "命中延迟（周期）", ymax=800, ymin=0, yticks=8, xpos=xpos)

    # 分层背景
    bands = [(4096, 32768, "L1", "#eff6ff"), (32768, 8388608, "L2（可用 ≈8–12 MiB）", "#ecfdf5"),
             (8388608, 67108864, "DRAM 区间", "#fffbeb")]
    for a, b, lab, col in bands:
        s.rect(X(a), box[1], X(b) - X(a), box[3] - box[1], col, "none")
        s.text((X(a) + X(b)) / 2, box[1] + 18, lab, 11.5, "middle", MUTED, "600")

    # 单 SM
    pts = [(X(b), Y(v)) for b, v, _ in data if v]
    s.poly(pts, "#9ca3af", 1.8, dash="5 4")
    for b, v, _ in data:
        s.circle(X(b), Y(v), 3.2, "#9ca3af")
    # 多 SM
    mpts = [(X(b), Y(m)) for b, _, m in data if m]
    s.poly(mpts, BLUE, 2.8)
    for b, _, m in data:
        if m:
            s.circle(X(b), Y(m), 4.2, BG, BLUE, 2)
            s.text(X(b), Y(m) - 11, f"{m:.0f}", 10.5, "middle", BLUE, "600")

    s.text(X(8388608) - 12, Y(458) - 18, "容量拐点 8→16 MiB（1.84×）", 11.5, "end", RED, "600")
    s.line(X(8388608), box[1] + 34, X(8388608), box[3], RED, 1.2, "4 4")
    s.text(X(4096) + 14, Y(37) - 14, "L1 命中 37 周期", 11, fill=BLUE, weight="600")
    footer(s, "来源：Round 3 / Round 10 / Round 11")
    save("fig6-memory-hierarchy.svg", s)


# ===========================================================================
# Fig 7：DRAM sector 尺寸
# ===========================================================================
def fig7():
    W, H = 900, 430
    s = Svg(W, H)
    header(s, "图 7  DRAM 的最小搬运单位是 32 B（sector）",
           "固定 256 MiB footprint，只改跨步；有用带宽在 32 B 处停止下降")
    data = [(4, 215.6), (8, 109.5), (16, 55.7), (32, 27.8), (64, 29.8),
            (128, 30.0), (256, 26.0), (512, 23.0), (1024, 12.9)]
    box = (86, 88, 870, 360)
    import math

    def X(b):
        lo, hi = math.log2(4), math.log2(1024)
        return box[0] + (box[2] - box[0]) * (math.log2(b) - lo) / (hi - lo)

    def Y(v):
        return box[3] - (box[3] - box[1]) * v / 240

    xpos = [(X(b), f"{b:g}") for b in [4, 8, 16, 32, 64, 128, 256, 512, 1024]]
    axes(s, box, "跨步（字节）", "有用带宽（GB/s）", ymax=240, ymin=0, yticks=6, xpos=xpos)
    s.rect(X(32), box[1], X(1024) - X(32), box[3] - box[1], "#fef2f2", "none")
    s.poly([(X(b), Y(v)) for b, v in data], BLUE, 2.8)
    for b, v in data:
        s.circle(X(b), Y(v), 4.2, BG, BLUE, 2)
        s.text(X(b), Y(v) - 12, f"{v:.0f}", 10.5, "middle", BLUE, "600")
    s.text(X(32) - 8, box[1] + 20, "32 B 处停止下降 ⟹ sector = 32 B", 12, "end", RED, "600")
    s.text(X(256), box[1] + 46, "再大也不变：128 B 的 line 只是 4 个 sector", 11, "middle", RED)
    footer(s, "来源：Round 3（results/20261115-221820-final）")
    save("fig7-sector-size.svg", s)


# ===========================================================================
# Fig 8：共享内存 bank
# ===========================================================================
def fig8():
    W, H = 900, 430
    s = Svg(W, H)
    header(s, "图 8  共享内存：32 个 bank × 4 B",
           "1 warp 内 32 个线程的地址跨步 vs 吞吐；冲突度 = 32 / gcd(跨步, 32)")
    data = [(0, 16.0), (1, 16.0), (2, 16.0), (3, 16.0), (4, 8.0), (8, 4.0),
            (16, 2.0), (17, 16.0), (31, 16.0), (32, 1.0), (33, 16.0), (64, 1.0)]
    box = (86, 92, 870, 350)
    xpos = []
    bw = (box[2] - box[0]) / len(data)
    for i, (st, _) in enumerate(data):
        xpos.append((box[0] + bw * (i + 0.5), str(st)))
    axes(s, box, "跨步（字 = 4 B）", "吞吐（字 / 周期）", ymax=20, ymin=0, yticks=5, xpos=xpos)
    for i, (st, v) in enumerate(data):
        x = box[0] + bw * (i + 0.5)
        w = bw * 0.55
        col = GREEN if v >= 15 else (TEAL if v >= 4 else RED)
        s.rect(x - w / 2, box[3] - (box[3] - box[1]) * v / 20, w,
               (box[3] - box[1]) * v / 20, col, "none", 4, op=0.85)
        s.text(x, box[3] - (box[3] - box[1]) * v / 20 - 8, f"{v:g}", 11, "middle", col, "600")
    s.line(box[0], box[3] - (box[3] - box[1]) * 16 / 20, box[2],
           box[3] - (box[3] - box[1]) * 16 / 20, MUTED, 1.2, "5 4")
    s.text(box[2] - 4, box[3] - (box[3] - box[1]) * 16 / 20 - 8,
           "无冲突上限 16（= LSU 每 2 周期 1 条）", 10.5, "end", MUTED)
    s.text(box[0] + 4, box[1] + 20, "跨步 32（32 路冲突）→ 1 字/周期 ⟹ bank 侧峰值 = 32 字/周期", 11.5, fill=RED)
    footer(s, "来源：Round 3。LDS 延迟 23.97 周期")
    save("fig8-smem-banks.svg", s)


# ===========================================================================
# Fig 9：Tensor Core 精度谱
# ===========================================================================
def fig9():
    W, H = 940, 470
    s = Svg(W, H)
    header(s, "图 9  Tensor Core：吞吐严格正比于 K，但 INT8 是 FP8 的两倍",
           "每 SM 每周期 FLOP（MMA 按 2·M·N·K 计）；标签下方是实测发射率")
    data = [
        ("TF32\nm16n8k8", 257, TEAL, "0.0626 MMA/cyc"),
        ("FP16\nm16n8k16", 514, BLUE, "0.1255 MMA/cyc"),
        ("FP8\nm16n8k32", 1025, PURPLE, "0.1248 MMA/cyc"),
        ("INT8\nm16n8k32", 2047, GREEN, "0.2499 MMA/cyc"),
        ("INT4\nm16n8k64", 4095, AMBER, "0.2499 MMA/cyc"),
        ("稀疏 FP16\nm16n8k32", 2049, "#94a3b8", "0.2500 MMA/cyc（等效）"),
    ]
    box = (100, 100, 880, 330)
    import math

    def Y(v):
        return box[3] - (box[3] - box[1]) * (math.log10(max(v, 100)) - 2.3) / (math.log10(4095) - 2.3)

    bw = (box[2] - box[0]) / len(data)
    axes(s, box, "", "FLOP / cycle / SM（对数）", ymax=1, ymin=0, yticks=0)
    for i, (nm, v, col, note) in enumerate(data):
        x = box[0] + bw * (i + 0.5)
        w = bw * 0.5
        top = Y(v)
        s.rect(x - w / 2, top, w, box[3] - top, col, "none", 5, op=0.85)
        s.text(x, top - 10, f"{v:,}", 12, "middle", col, "600")
        for j, line in enumerate(nm.split("\n")):
            s.text(x, box[3] + 20 + j * 15, line, 11, "middle", INK if j == 0 else MUTED)
        s.text(x, box[3] + 52, note.split("（")[0], 9.5, "middle", MUTED)
    for lv in [256, 512, 1024, 2048, 4096]:
        y = Y(lv)
        s.line(box[0], y, box[2], y, GRID, 1)
        s.text(box[0] - 10, y + 4, f"{lv:,}", 10, "end", MUTED)
    s.text(box[0] + 6, box[1] + 18, "FP16 张量 = FP32 向量（256）的 2×；FP8 = 4×；INT8 = 8×；INT4 = 16×", 11.5, fill=MUTED)
    footer(s, "来源：Round 5 / Round 6 / Round 9。MMA 延迟 ≈40 周期")
    save("fig9-tensor-precision.svg", s)


# ===========================================================================
# Fig 10：桌面干扰与降噪
# ===========================================================================
def fig10():
    W, H = 940, 460
    s = Svg(W, H)
    header(s, "图 10  桌面干扰的量化，以及「多 SM 取最小」的降噪效果",
           "左：同一测量连做 40 次的分布；右：降噪前后对比")
    # 左：直方图
    hist = [3, 6, 8, 4, 3, 7, 2, 3, 1, 3]
    edges = [272, 330, 388, 446, 505, 563, 621, 679, 737, 795]
    box = (96, 100, 470, 370)
    axes(s, box, "单次测量得到的延迟（周期）", "次数", ymax=10, ymin=0, yticks=5)
    bw = (box[2] - box[0]) / len(hist)
    for i, n in enumerate(hist):
        x = box[0] + bw * (i + 0.5)
        s.rect(x - bw * 0.36, box[3] - (box[3] - box[1]) * n / 10, bw * 0.72,
               (box[3] - box[1]) * n / 10, AMBER, "none", 4, op=0.85)
        s.text(x, box[3] + 16, str(edges[i]), 9.5, "middle", MUTED)
    s.text(box[0] + 6, box[1] + 16, "min 272 / median 506 / max 930  ⟹ 3.4×", 12, fill=RED, weight="600")
    s.text(box[0] + 6, box[1] + 36, "（3 秒内，GPU 利用率中位数 23%）", 10.5, fill=MUTED)

    # 右：降噪前后
    data = [("64K", 228.7, 178.5), ("256K", 271.5, 210.1), ("1M", 323.7, 210.6),
            ("2M", 483.0, 233.4), ("4M", 507.8, 240.0), ("8M", 471.8, 249.3)]
    box2 = (570, 100, 900, 370)
    import math
    vmax = 550

    def Y2(v):
        return box2[3] - (box2[3] - box2[1]) * v / vmax

    s.rect(box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1], BG)
    for v in [0, 200, 400]:
        y = Y2(v)
        s.line(box2[0], y, box2[2], y, GRID, 1)
        s.text(box2[0] - 8, y + 4, str(v), 10, "end", MUTED)
    bw2 = (box2[2] - box2[0]) / len(data)
    for i, (nm, single, multi) in enumerate(data):
        x = box2[0] + bw2 * (i + 0.5)
        s.rect(x - bw2 * 0.34, Y2(single), bw2 * 0.30, box2[3] - Y2(single),
               "#9ca3af", "none", 4, op=0.8)
        s.rect(x - bw2 * 0.02, Y2(multi), bw2 * 0.30, box2[3] - Y2(multi),
               BLUE, "none", 4, op=0.9)
        s.text(x - bw2 * 0.19, Y2(single) - 6, f"{single:.0f}", 9, "middle", MUTED)
        s.text(x + bw2 * 0.13, Y2(multi) - 6, f"{multi:.0f}", 9, "middle", BLUE, "600")
        s.text(x, box2[3] + 16, nm, 10.5, "middle", MUTED)
    s.rect(box2[0] + 6, box2[1] + 8, 12, 10, "#9ca3af")
    s.text(box2[0] + 24, box2[1] + 17, "单 SM 单次", 10.5, fill=MUTED)
    s.rect(box2[0] + 100, box2[1] + 8, 12, 10, BLUE)
    s.text(box2[0] + 118, box2[1] + 17, "多 SM 取最小", 10.5, fill=BLUE)
    s.text(box2[0], box2[3] + 40, "2M/4M 改善 2.1×/1.7× ⟹ 原来的「台阶」是干扰伪像", 11, fill=RED, weight="600")
    footer(s, "来源：Round 10（results/20261115-221820-final）")
    save("fig10-noise-and-denoise.svg", s)


# ===========================================================================
# Fig 11：全卡扩展性
# ===========================================================================
def fig11():
    W, H = 900, 420
    s = Svg(W, H)
    header(s, "图 11  24 个 SM：固定每 CTA 工作量的扩展性台阶",
           "grid 1→24 时间恒定，grid=25 立刻翻倍 ⟹ 这是直接测出 SM 数的方法")
    data = [(1, 0.839), (2, 0.651), (4, 0.650), (8, 0.662), (12, 0.647),
            (24, 0.649), (25, 1.285), (48, 1.279)]
    box = (96, 96, 860, 350)
    vmax = 1.5

    def Y(v):
        return box[3] - (box[3] - box[1]) * v / vmax

    bw = (box[2] - box[0]) / len(data)
    axes(s, box, "grid（CTA 数）", "内核时间（ms）", ymax=vmax, ymin=0, yticks=5,
         xpos=[(box[0] + bw * (i + 0.5), str(g)) for i, (g, _) in enumerate(data)])
    for i, (g, t) in enumerate(data):
        x = box[0] + bw * (i + 0.5)
        col = BLUE if g <= 24 else RED
        s.rect(x - bw * 0.28, Y(t), bw * 0.56, box[3] - Y(t), col, "none", 4, op=0.85)
        s.text(x, Y(t) - 8, f"{t:.2f}", 10.5, "middle", col, "600")
    s.line(box[0], Y(0.65), box[2], Y(0.65), GREEN, 1.2, "5 4")
    s.text(box[0] + 6, Y(0.65) - 10, "1 波：24 个 CTA 全部并行（0.65 ms）", 11, fill=GREEN)
    s.text(box[0] + bw * 6.1, box[1] + 20, "第 25 个 CTA 起需要第二波 ⟹ 只有 24 个 SM", 11.5, fill=RED, weight="600")
    footer(s, "来源：Round 7。全卡 FP32 实测 15.1 TFLOPS（理想 16.86 的 89%）")
    save("fig11-cta-scaling.svg", s)



# ===========================================================================
# Fig 12：L1I 容量 与 L0I 分区争用
# ===========================================================================
def fig12():
    W, H = 1000, 470
    s = Svg(W, H)
    header(s, "图 12  指令缓存：L1I 容量约 64-80 KiB，跨分区互不干扰",
           "左：代码足迹 vs 每指令周期；右：同一/不同调度器上放两份代码")
    import math
    # (SASS 实测代码量 KiB, cycles/指令)
    data = [(5.1, 1.128), (13.1, 1.055), (25.1, 1.029), (33.1, 1.022),
            (49.1, 1.014), (57.1, 1.012), (65.1, 1.086), (81.1, 2.431),
            (97.1, 2.431), (129.1, 2.522), (193.1, 3.251), (513.1, 3.924)]
    box = (86, 96, 560, 360)

    def X(v):
        lo, hi = math.log2(4), math.log2(560)
        return box[0] + (box[2] - box[0]) * (math.log2(max(v, 4)) - lo) / (hi - lo)

    def Y(v):
        return box[3] - (box[3] - box[1]) * v / 4.2

    axes(s, box, "SASS 实测代码量（KiB，对数轴）", "cycles / 指令",
         ymax=4.2, ymin=0, yticks=4,
         xpos=[(X(4), "4"), (X(16), "16"), (X(64), "64"), (X(256), "256")])
    s.rect(X(64), box[1], X(80) - X(64), box[3] - box[1], "#fef2f2", "none")
    s.line(box[0], Y(1.0), box[2], Y(1.0), MUTED, 1.2, "5 4")
    s.text(box[0] + 30, Y(1.0) - 12, "理论下限 1.00", 10.5, fill=MUTED)
    s.poly([(X(a), Y(b)) for a, b in data], BLUE, 2.6)
    for a, b in data:
        s.circle(X(a), Y(b), 4.2, BG, BLUE, 2)
    s.text(X(57) - 6, Y(1.012) + 26, "64 KiB 以内：1.01-1.13", 11, "end", BLUE, "600")
    s.text(X(81) + 8, Y(2.431) - 24, "80 KiB：2.43（2.2x）", 11, fill=RED, weight="600")
    s.line(X(64), box[1] + 30, X(64), box[3], RED, 1.2, "4 4")
    s.text(X(64) - 6, box[1] + 24, "容量拐点 64-80 KiB", 11, "end", RED, "600")

    # 右：L0I 分组柱
    box2 = (660, 96, 950, 360)
    s.rect(box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1], BG)
    for v in [0, 1, 2]:
        y = box2[3] - (box2[3] - box2[1]) * v / 2.6
        s.line(box2[0], y, box2[2], y, GRID, 1)
        s.text(box2[0] - 8, y + 4, str(v), 10, "end", MUTED)
    cfgs = [("跨分区·同码", 1.128, 1.044), ("跨分区·异码", 1.128, 1.045),
            ("同分区·同码", 2.131, 2.034), ("同分区·异码", 2.156, 2.040)]
    bw = (box2[2] - box2[0]) / len(cfgs)
    for i, (lab, v4, v16) in enumerate(cfgs):
        x = box2[0] + bw * (i + 0.5)
        for j, (v, col) in enumerate(((v4, TEAL), (v16, AMBER))):
            h = (box2[3] - box2[1]) * v / 2.6
            s.rect(x - bw * 0.30 + j * bw * 0.30, box2[3] - h, bw * 0.26, h, col,
                   "none", 3, op=0.85)
            s.text(x - bw * 0.17 + j * bw * 0.30, box2[3] - h - 6, f"{v:.2f}",
                   9.5, "middle", col, "600")
        s.text(x, box2[3] + 18, lab, 9.5, "middle", MUTED)
    y2 = box2[3] - (box2[3] - box2[1]) * 2.0 / 2.6
    s.line(box2[0], y2, box2[2], y2, RED, 1.2, "5 4")
    s.text(box2[0] + 4, y2 - 7, "2 warp 抢一调度器的下限", 9.5, RED)
    s.rect(box2[0], box2[1] - 18, 10, 8, TEAL)
    s.text(box2[0] + 14, box2[1] - 11, "4 KiB/份", 10, fill=MUTED)
    s.rect(box2[0] + 80, box2[1] - 18, 10, 8, AMBER)
    s.text(box2[0] + 94, box2[1] - 11, "16 KiB/份", 10, fill=MUTED)
    footer(s, "来源：Round 12（移植 icache_b300_pkg，results/20261115-230119）")
    save("fig12-icache.svg", s)


# ===========================================================================
# Fig 13：L1I 容量细扫（57 点，每点 1424 B）
# ===========================================================================
def fig13():
    import json
    import math
    data = json.loads((OUT / "data" / "cap_curve.json").read_text(encoding="utf-8"))
    pts = [(c["bytes"], c["per_step"]) for c in data["curve"]]
    base = min(v for _, v in pts)

    W, H = 1000, 470
    s = Svg(W, H)
    header(s, "图 13  指令缓存容量细扫：每步代价随足迹的拐点",
           "统一尺寸的填充块（1424 B）× 57 个足迹点；纵轴是循环稳态的每步周期")
    box = (96, 98, 930, 372)

    xlo, xhi = 0, 88000

    def X(v):
        return box[0] + (box[2] - box[0]) * (v - xlo) / (xhi - xlo)

    def Y(v):
        lo, hi = base - 12, 690
        return box[3] - (box[3] - box[1]) * (v - lo) / (hi - lo)

    axes(s, box, "足迹（字节）", "每步周期", ymax=1, ymin=0, yticks=0,
         xpos=[(X(v), f"{v // 1024}K") for v in (0, 16384, 32768, 49152, 65536, 81920)])
    for v in (600, 620, 640, 660, 680):
        y = Y(v)
        s.line(box[0], y, box[2], y, GRID, 1)
        s.text(box[0] - 10, y + 4, str(v), 10.5, "end", MUTED)

    # 平区与平台的参考线
    s.line(box[0], Y(base), box[2], Y(base), BLUE, 1.2, "5 4")
    s.text(box[0] + 6, Y(base) - 8, f"平区基线 {base:.0f} 周期", 10.5, fill=BLUE)
    plateau = max(v for _, v in pts)
    s.line(X(76000), Y(plateau), box[2], Y(plateau), RED, 1.2, "5 4")
    s.text(box[2] - 4, Y(plateau) - 8, f"平台 {plateau:.0f}（+{100 * (plateau / base - 1):.0f}%）",
           10.5, "end", RED)

    # 拐点区域
    s.rect(X(58400), box[1], X(64000) - X(58400), box[3] - box[1], "#fef2f2", "none")
    s.text(X(61200), box[3] + 40, "拐点 58.4–64 KiB", 11.5, "middle", RED, "600")

    s.poly([(X(a), Y(b)) for a, b in pts], BLUE, 2.6)
    for a, b in pts[::2]:
        s.circle(X(a), Y(b), 3.0, BG, BLUE, 1.6)
    footer(s, "来源：Round 13b（results/20261118-*，数据 docs/figures/data/cap_curve.json）")
    save("fig13-icache-capacity.svg", s)


if __name__ == "__main__":
    print("生成 SVG：")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13):
        fn()
    print(f"完成，输出目录 {OUT}")
