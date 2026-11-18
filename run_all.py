"""跑一轮或多轮微基准，把原始数据与日志落到 results/<timestamp>/。

用法:
    python run_all.py                 # 全部轮次
    python run_all.py --round 0 1     # 只跑第 0、1 轮
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import env  # noqa: E402

ROUNDS = {
    "0": ("probes.round0", "设备事实 / 主时钟 / SM 普查 / 占用率"),
    "1": ("probes.round1_latency", "指令延迟"),
    "2": ("probes.round2_throughput", "指令吞吐 / 发射宽度"),
    "3": ("probes.round3_memory", "存储层次：L1/L2/DRAM 延迟、smem bank、带宽、sector"),
    "4": ("probes.round4_scheduler", "warp 调度器行为 / 发射宽度"),
    "5": ("probes.round5_tensor", "第 4 代 Tensor Core：f16/tf32/fp8 吞吐与延迟"),
    "6": ("probes.round6_sparse_dispatch", "稀疏 MMA / warp 分配规则"),
    "7": ("probes.round7_bank_agg", "寄存器 bank / 操作数端口、多 CTA 聚合扩展性"),
    "8": ("probes.round8_bankfit_dispatch", "warp 发射节流 / 寄存器 bank 统计拟合"),
    "9": ("probes.round9_int_tensor_tlb", "INT8/INT4 张量 / TLB"),
    "10": ("probes.round10_noise_multism", "桌面干扰量化 / 多 SM 降噪"),
    "11": ("probes.round11_l2_capacity", "L2 容量拐点"),
    "12": ("probes.round12_icache", "L1I 容量 / L0I 分区争用（移植 b300_pkg）"),
    "13": ("probes.round13_icache_struct", "指令缓存分级容量 / 预取 / 替换"),
    "14": ("probes.round14_capacity_fine", "L1I 容量细扫（57 点，二分派发）"),
}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "tolist"):          # numpy 标量/数组（若环境里恰好有）
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", nargs="*", default=None, help="要跑的轮次编号，默认全部")
    ap.add_argument("--reps", type=int, default=7, help="每项测量重复次数")
    ap.add_argument("--out", default=None, help="结果目录（默认自动时间戳）")
    ap.add_argument("--tag", default="", help="结果目录后缀标签")
    args = ap.parse_args(argv)

    env.prepare()
    env.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tag:
        ts += "-" + args.tag
    outdir = Path(args.out) if args.out else env.RESULTS_DIR / ts
    outdir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    log(f"# 4060 SM 微架构逆向  {_dt.datetime.now().isoformat(timespec='seconds')}")
    log(f"# 结果目录: {outdir}")

    wanted = args.round if args.round else list(ROUNDS)
    results: dict = {"meta": {"started": _dt.datetime.now().isoformat(timespec="seconds"),
                              "rounds": wanted, "reps": args.reps}}
    rc = 0
    for r in wanted:
        if r not in ROUNDS:
            log(f"!! 未知轮次 {r}")
            continue
        mod_name, title = ROUNDS[r]
        log("")
        log(f"########## Round {r}: {title} ##########")
        try:
            mod = __import__(mod_name, fromlist=["run"])
            res = mod.run(log=log, reps=args.reps)
            results[f"round{r}"] = res
        except Exception as exc:  # noqa: BLE001
            rc = 1
            log(f"!! Round {r} 失败: {exc.__class__.__name__}: {exc}")
            log(traceback.format_exc())
            results[f"round{r}_error"] = f"{exc.__class__.__name__}: {exc}"

    results["meta"]["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    (outdir / "results.json").write_text(
        json.dumps(_jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("")
    log(f"# 已写出 {outdir/'results.json'} 与 run.log")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
