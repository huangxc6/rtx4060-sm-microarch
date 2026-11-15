"""GPU 访问层：DLL 搜索路径、设备属性、显存。

依赖只有 :mod:`harness.cuda`（纯 ctypes）。本文件负责把散落的
``CU_DEVICE_ATTRIBUTE_*`` 数字变成可读的名字，并给出 :class:`Gpu` 门面。
"""
from __future__ import annotations

import os
import site
from pathlib import Path

from . import cuda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "build" / "tools"
RESULTS_DIR = PROJECT_ROOT / "results"

CudaError = cuda.CudaError
chk = cuda.chk

# CU_DEVICE_ATTRIBUTE_* -> 语义名
ATTRS = {
    "max_threads_per_block": 1,
    "max_block_dim_x": 2,
    "max_block_dim_y": 3,
    "max_grid_dim_x": 5,
    "max_shared_memory_per_block": 8,
    "warp_size": 10,
    "max_registers_per_block": 12,
    "clock_rate_khz": 13,
    "multiprocessor_count": 16,
    "kernel_overhead_us": 22,
    "memory_clock_rate_khz": 36,
    "global_memory_bus_width": 37,
    "l2_cache_size": 38,
    "max_threads_per_multiprocessor": 39,
    "async_engine_count": 40,
    "unified_addressing": 41,
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "max_shared_memory_per_multiprocessor": 81,
    "max_registers_per_multiprocessor": 82,
    "concurrent_kernels": 84,
    "pci_bus_id": 33,
    "pci_device_id": 34,
    "pci_domain_id": 35,
    "max_blocks_per_multiprocessor": 106,
    "host_native_atomic_supported": 86,
}

_prepared = False


def _nvidia_bins() -> list[Path]:
    bins: list[Path] = []
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for p in roots:
        nv = Path(p) / "nvidia"
        if not nv.is_dir():
            continue
        for sub in nv.iterdir():
            for cand in (sub / "bin", sub / "lib"):
                if cand.is_dir():
                    bins.append(cand)
    return bins


def _tool_bins() -> list[Path]:
    if not TOOLS_DIR.is_dir():
        return []
    return sorted({p.parent for p in TOOLS_DIR.rglob("*.exe")})


def prepare() -> None:
    """把 nvrtc / nvdisasm 所在目录登记到进程的 DLL 搜索路径。"""
    global _prepared
    if _prepared:
        return
    for d in _nvidia_bins() + _tool_bins():
        try:
            os.add_dll_directory(str(d))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    _prepared = True


class Gpu:
    """单卡门面：设备属性 + 上下文 + 显存 + 模块。"""

    def __init__(self, ordinal: int = 0):
        prepare()
        self._c = cuda.Ctx(ordinal)

    # -- 属性 ------------------------------------------------------------
    def attr(self, name: str) -> int:
        return self._c.attr(ATTRS[name])

    def name(self) -> str:
        return self._c.name()

    def total_mem(self) -> int:
        return self._c.total_mem()

    @property
    def arch(self) -> str:
        return (f"sm_{self.attr('compute_capability_major')}"
                f"{self.attr('compute_capability_minor')}")

    @property
    def sm_count(self) -> int:
        return self.attr("multiprocessor_count")

    @property
    def warp_size(self) -> int:
        return self.attr("warp_size")

    def describe(self) -> dict:
        major, minor = self.attr("compute_capability_major"), self.attr("compute_capability_minor")
        d = {
            "name": self.name(),
            "compute_capability": f"{major}.{minor}",
            "arch": self.arch,
            "total_mem_bytes": self.total_mem(),
            "sm_count": self.attr("multiprocessor_count"),
            "warp_size": self.attr("warp_size"),
            "max_threads_per_sm": self.attr("max_threads_per_multiprocessor"),
            "max_blocks_per_sm": self.attr("max_blocks_per_multiprocessor"),
            "max_warps_per_sm": (self.attr("max_threads_per_multiprocessor")
                                 // self.attr("warp_size")),
            "max_regs_per_sm": self.attr("max_registers_per_multiprocessor"),
            "max_regs_per_block": self.attr("max_registers_per_block"),
            "smem_per_sm": self.attr("max_shared_memory_per_multiprocessor"),
            "smem_per_block": self.attr("max_shared_memory_per_block"),
            "l2_bytes": self.attr("l2_cache_size"),
            "nominal_sm_clock_khz": self.attr("clock_rate_khz"),
            "nominal_mem_clock_khz": self.attr("memory_clock_rate_khz"),
            "bus_width_bits": self.attr("global_memory_bus_width"),
            "max_threads_per_block": self.attr("max_threads_per_block"),
            "max_grid_dim_x": self.attr("max_grid_dim_x"),
            "async_engines": self.attr("async_engine_count"),
            "concurrent_kernels": self.attr("concurrent_kernels"),
            "unified_addressing": bool(self.attr("unified_addressing")),
            "pci": (f"{self.attr('pci_domain_id'):04x}:"
                    f"{self.attr('pci_bus_id'):02x}:{self.attr('pci_device_id'):02x}"),
            "driver_version": cuda.driver_version(),
            "nvrtc_version": cuda.nvrtc_version(),
        }
        mem_hz = d["nominal_mem_clock_khz"] * 1000.0
        d["derived_mem_bandwidth_gbs"] = 2 * mem_hz * d["bus_width_bits"] / 8 / 1e9
        return d

    # -- 显存 ------------------------------------------------------------
    def malloc(self, nbytes: int) -> int:
        return self._c.malloc(nbytes)

    def free(self, ptr: int) -> None:
        self._c.free(ptr)

    def htod(self, dst: int, data) -> None:
        self._c.htod(dst, data)

    def dtoh(self, src: int, nbytes: int) -> bytes:
        return self._c.dtoh(src, nbytes)

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        self._c.memset(dst, value, nbytes)

    def sync(self) -> None:
        self._c.sync()

    # -- 模块 ------------------------------------------------------------
    def module(self, cubin: bytes):
        return self._c.module(cubin)

    def function(self, mod, name: str):
        return self._c.function(mod, name)

    def func_attr(self, fn, code: int):
        return self._c.func_attr(fn, code)

    def occupancy_blocks(self, fn, block_size: int, dyn_smem: int = 0):
        return self._c.occupancy_blocks(fn, block_size, dyn_smem)

    def launch(self, fn, grid, block, args, smem: int = 0, stream=None) -> None:
        self._c.launch(fn, grid, block, args, smem=smem, stream=stream)
