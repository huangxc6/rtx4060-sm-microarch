"""NVRTC JIT 编译 + 模块加载 + SASS 反汇编。

SASS 是本工程的核心证据：微基准里写的 ``asm volatile`` 到底被 ptxas 变成了
什么指令、有没有被插桩或重排，只能靠 ``cuobjdump -sass`` / ``nvdisasm`` 回答。
这两个工具来自 NVIDIA redist 的独立 zip 包（见 ``build/tools``），不需要装 Toolkit。
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from . import cuda, env

# CUfunction_attribute
FUNC_ATTR = {
    "max_threads_per_block": 0,
    "shared_size_bytes": 1,
    "const_size_bytes": 2,
    "local_size_bytes": 3,
    "num_regs": 4,
    "ptx_version": 5,
    "binary_version": 6,
    "cache_mode_ca": 7,
    "max_dynamic_shared_size_bytes": 8,
    "preferred_smem_carveout": 9,
}

CompileError = cuda.CompileError


def _tool(name: str) -> Path | None:
    found = sorted(env.TOOLS_DIR.rglob(f"{name}.exe")) if env.TOOLS_DIR.is_dir() else []
    return found[0] if found else None


class Module:
    """一份 CUDA 源码编译出的模块。"""

    def __init__(self, gpu: env.Gpu, src: str, name: str = "bench.cu", opts=()):
        self.gpu = gpu
        self.src = src
        self.name = name
        self.options = [
            f"--gpu-architecture={gpu.arch}",
            "--std=c++17",
            # 不做 fast-math：除法/开方必须保留完整指令，否则测的不是我们想测的
            "--fmad=true",
            *opts,
        ]
        self.cubin, self.ptx, self.log = cuda.nvrtc_compile(src, name, self.options)
        self._mod = gpu.module(self.cubin)

    def func(self, name: str) -> "Kernel":
        return Kernel(self.gpu, self, name)

    def regs(self, name: str) -> dict:
        return self.func(name).attrs()

    # -- 反汇编 ----------------------------------------------------------
    def sass(self, func: str | None = None) -> str:
        """返回 SASS 文本；``func`` 给定时只返回该函数的段落。

        ``cuobjdump -fun`` 要的是函数*索引*而不是名字，所以这里拿到全量
        反汇编后按 ``.text.<name>`` 段切分。
        """
        text = self._sass_all()
        return extract_function(text, func) if func else text

    def _sass_all(self) -> str:
        with tempfile.TemporaryDirectory(prefix="sass_") as td:
            path = Path(td) / "module.cubin"
            path.write_bytes(self.cubin)
            cuobjdump = _tool("cuobjdump")
            if cuobjdump is not None:
                out = subprocess.run([str(cuobjdump), "-sass", str(path)],
                                     capture_output=True, text=True, errors="replace")
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout
            nvdisasm = _tool("nvdisasm")
            if nvdisasm is None:
                return "<未找到 cuobjdump / nvdisasm>"
            out = subprocess.run([str(nvdisasm), "-c", str(path)],
                                 capture_output=True, text=True, errors="replace")
            return out.stdout if out.returncode == 0 else out.stdout + out.stderr


def extract_function(sass_text: str, func: str) -> str:
    """从全量反汇编里取 ``.text.<func>`` 段。"""
    marker = f".text.{func}"
    lines = sass_text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if marker in ln and ln.lstrip().startswith("//"):
            start = i
            break
    if start is None:
        for i, ln in enumerate(lines):
            if ln.strip().startswith(f"{func}:"):
                start = i
                break
    if start is None:
        return f"<SASS 中未找到函数 {func}>"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith("//---"):
            end = j
            break
    return "\n".join(lines[start:end])


class Kernel:
    """CUfunction + 静态资源信息 + 启动。"""

    def __init__(self, gpu: env.Gpu, module: Module, name: str):
        self.gpu = gpu
        self.module = module
        self.name = name
        self.fn = gpu.function(module._mod, name)

    def attrs(self) -> dict:
        return {k: self.gpu.func_attr(self.fn, code) for k, code in FUNC_ATTR.items()}

    @property
    def num_regs(self) -> int | None:
        return self.gpu.func_attr(self.fn, FUNC_ATTR["num_regs"])

    def launch(self, grid, block, ptrs=(), smem: int = 0, stream=None) -> None:
        self.gpu.launch(self.fn, grid, block, ptrs, smem=smem, stream=stream)

    def __repr__(self) -> str:
        return f"<Kernel {self.name} regs={self.num_regs}>"
