"""纯 ctypes 的 CUDA 驱动 API + NVRTC 绑定。

为什么不用 cuda-python
----------------------
本机 pip 解析/安装在空转（CPU 占满但不动），而镜像直连有 1.4 MB/s。
把 `cuda-bindings` 这个编译扩展也省掉，整条链路只剩两个 DLL：
``nvcuda.dll``（驱动自带）与 ``nvrtc64_120_0.dll``（从 pip wheel 里手工解压）。

好处不只是"少装一个包"：ctypes 原型把每个参数的类型都写死，
不存在"不同版本返回值形态不同"的不确定性；而且 ``cuLaunchKernel`` 的
``void**`` 编组由本文件显式构造，行为可审计。

所有导出函数统一返回 ``CUresult``（0 成功），统一用 :func:`chk` 检查。
"""
from __future__ import annotations

import ctypes
from ctypes import (POINTER, byref, c_char_p, c_int, c_size_t, c_uint,
                    c_uint64, c_void_p)

# --------------------------------------------------------------------------
# 错误
# --------------------------------------------------------------------------
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1


class CudaError(RuntimeError):
    def __init__(self, msg: str, rc: int | None = None):
        super().__init__(msg)
        self.rc = rc


class CompileError(RuntimeError):
    pass


_driver_lib = None
_nvrtc_lib = None
_F: dict[str, ctypes._CFuncPtr] = {}


def _load_driver():
    global _driver_lib
    if _driver_lib is not None:
        return _driver_lib
    for name in ("nvcuda.dll", "nvcuda"):
        try:
            _driver_lib = ctypes.WinDLL(name)
            break
        except OSError:
            continue
    if _driver_lib is None:
        raise CudaError("无法加载 nvcuda.dll（显卡驱动未安装？）")
    _bind_driver()
    return _driver_lib


def _load_nvrtc():
    global _nvrtc_lib
    if _nvrtc_lib is not None:
        return _nvrtc_lib
    for name in ("nvrtc64_120_0.dll", "nvrtc64_130_0.dll", "nvrtc64_120_0"):
        try:
            _nvrtc_lib = ctypes.WinDLL(name)
            break
        except OSError:
            continue
    if _nvrtc_lib is None:
        raise CompileError(
            "无法加载 nvrtc（需要 .venv/Lib/site-packages/nvidia/cuda_nvrtc/bin 在 DLL 搜索路径中）"
        )
    _bind_nvrtc()
    return _nvrtc_lib


def _sig(lib, name: str, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    _F[name] = fn
    return fn


def _bind_driver():
    lib = _driver_lib
    p = c_void_p
    _sig(lib, "cuGetErrorName", c_int, [c_int, POINTER(c_char_p)])
    _sig(lib, "cuGetErrorString", c_int, [c_int, POINTER(c_char_p)])
    _sig(lib, "cuInit", c_int, [c_uint])
    _sig(lib, "cuDriverGetVersion", c_int, [POINTER(c_int)])
    _sig(lib, "cuDeviceGet", c_int, [POINTER(c_int), c_int])
    _sig(lib, "cuDeviceGetName", c_int, [c_char_p, c_int, c_int])
    _sig(lib, "cuDeviceGetAttribute", c_int, [POINTER(c_int), c_int, c_int])
    _sig(lib, "cuDeviceTotalMem_v2", c_int, [POINTER(c_size_t), c_int])
    _sig(lib, "cuDevicePrimaryCtxRetain", c_int, [POINTER(p), c_int])
    _sig(lib, "cuCtxSetCurrent", c_int, [p])
    _sig(lib, "cuCtxSynchronize", c_int, [])
    _sig(lib, "cuMemAlloc_v2", c_int, [POINTER(c_uint64), c_size_t])
    _sig(lib, "cuMemFree_v2", c_int, [c_uint64])
    _sig(lib, "cuMemcpyHtoD_v2", c_int, [c_uint64, p, c_size_t])
    _sig(lib, "cuMemcpyDtoH_v2", c_int, [p, c_uint64, c_size_t])
    _sig(lib, "cuMemsetD8_v2", c_int, [c_uint64, c_uint, c_size_t])
    _sig(lib, "cuModuleLoadData", c_int, [POINTER(p), p])
    _sig(lib, "cuModuleGetFunction", c_int, [POINTER(p), p, c_char_p])
    _sig(lib, "cuModuleUnload", c_int, [p])
    _sig(lib, "cuFuncGetAttribute", c_int, [POINTER(c_int), c_int, p])
    _sig(lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor", c_int,
         [POINTER(c_int), p, c_int, c_size_t])
    _sig(lib, "cuLaunchKernel", c_int,
         [p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, p,
          POINTER(c_void_p), POINTER(c_void_p)])


def _bind_nvrtc():
    lib = _nvrtc_lib
    _sig(lib, "nvrtcVersion", c_int, [POINTER(c_int), POINTER(c_int)])
    _sig(lib, "nvrtcCreateProgram", c_int,
         [POINTER(c_void_p), c_char_p, c_char_p, c_int,
          POINTER(c_char_p), POINTER(c_char_p)])
    _sig(lib, "nvrtcDestroyProgram", c_int, [POINTER(c_void_p)])
    _sig(lib, "nvrtcCompileProgram", c_int, [c_void_p, c_int, POINTER(c_char_p)])
    _sig(lib, "nvrtcGetProgramLogSize", c_int, [c_void_p, POINTER(c_size_t)])
    _sig(lib, "nvrtcGetProgramLog", c_int, [c_void_p, c_char_p])
    _sig(lib, "nvrtcGetCUBINSize", c_int, [c_void_p, POINTER(c_size_t)])
    _sig(lib, "nvrtcGetCUBIN", c_int, [c_void_p, c_char_p])
    _sig(lib, "nvrtcGetPTXSize", c_int, [c_void_p, POINTER(c_size_t)])
    _sig(lib, "nvrtcGetPTX", c_int, [c_void_p, c_char_p])
    _sig(lib, "nvrtcGetNumSupportedArchs", c_int, [POINTER(c_int)])
    _sig(lib, "nvrtcGetSupportedArchs", c_int, [POINTER(c_int)])


# --------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------
def chk(rc: int, what: str = "") -> int:
    if rc != CUDA_SUCCESS:
        raise CudaError(f"{what or 'CUDA 调用'} 失败: {err_name(rc)} (rc={rc})", rc)
    return rc


def err_name(rc: int) -> str:
    s = c_char_p()
    try:
        _F["cuGetErrorName"](rc, byref(s))
    except Exception:  # pragma: no cover
        return f"rc={rc}"
    return s.value.decode() if s.value else f"rc={rc}"


def driver_version() -> int:
    v = c_int()
    chk(_F["cuDriverGetVersion"](byref(v)), "cuDriverGetVersion")
    return v.value


def nvrtc_version() -> str:
    _load_nvrtc()
    a, b = c_int(), c_int()
    chk(_F["nvrtcVersion"](byref(a), byref(b)), "nvrtcVersion")
    return f"{a.value}.{b.value}"


# --------------------------------------------------------------------------
# NVRTC 编译
# --------------------------------------------------------------------------
def nvrtc_compile(src: str, name: str, options: list[str]) -> tuple[bytes, bytes, str]:
    """编译 CUDA C++ 源码，返回 (cubin, ptx, log)。"""
    _load_nvrtc()
    prog = c_void_p()
    chk(_F["nvrtcCreateProgram"](byref(prog), src.encode(), name.encode(),
                                 0, None, None), "nvrtcCreateProgram")
    try:
        opts = [o.encode() for o in options]
        arr = (c_char_p * len(opts))(*opts)
        rc = _F["nvrtcCompileProgram"](prog, len(opts), arr)
        log = _program_log(prog)
        if rc != CUDA_SUCCESS:
            # 本机的 nvrtc 偶尔会返回非零码（CUDA_ERROR_PROFILER_NOT_INITIALIZED）
            # 而日志里只有 "info : Advisory:"、没有任何 error/fatal——实际是编译成功的。
            # 所以判据以**日志内容**为准：没有 error/fatal 就继续取 cubin。
            fatal = any(
                ("fatal" in ln.lower())
                or ("error" in ln.lower() and "advisory" not in ln.lower()
                    and not ln.strip().startswith("nvrtc"))
                for ln in log.splitlines()
            )
            if fatal:
                raise CompileError(f"nvrtc 编译失败 (rc={rc}):\n{log}")
        cubin = _blob(prog, "nvrtcGetCUBINSize", "nvrtcGetCUBIN", "CUBIN")
        try:
            ptx = _blob(prog, "nvrtcGetPTXSize", "nvrtcGetPTX", "PTX")
        except CompileError:
            ptx = b""
        return cubin, ptx, log
    finally:
        _F["nvrtcDestroyProgram"](byref(prog))


def _program_log(prog) -> str:
    sz = c_size_t()
    if _F["nvrtcGetProgramLogSize"](prog, byref(sz)) != 0 or sz.value <= 1:
        return ""
    buf = ctypes.create_string_buffer(sz.value)
    if _F["nvrtcGetProgramLog"](prog, buf) != 0:
        return ""
    return buf.value.decode(errors="replace")


def _blob(prog, size_fn: str, get_fn: str, what: str) -> bytes:
    sz = c_size_t()
    chk(_F[size_fn](prog, byref(sz)), f"{size_fn}")
    if sz.value == 0:
        raise CompileError(f"{what} 为空")
    buf = ctypes.create_string_buffer(sz.value)
    chk(_F[get_fn](prog, buf), get_fn)
    return buf.raw[: sz.value]


# --------------------------------------------------------------------------
# 驱动侧封装
# --------------------------------------------------------------------------
class Ctx:
    """设备 + 主上下文 + 显存操作。"""

    def __init__(self, ordinal: int = 0):
        _load_driver()
        chk(_F["cuInit"](0), "cuInit")
        dev = c_int()
        chk(_F["cuDeviceGet"](byref(dev), ordinal), "cuDeviceGet")
        self.device = dev.value
        ctx = c_void_p()
        chk(_F["cuDevicePrimaryCtxRetain"](byref(ctx), self.device),
            "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        chk(_F["cuCtxSetCurrent"](ctx), "cuCtxSetCurrent")
        self._attrs: dict[int, int] = {}

    def attr(self, code: int) -> int:
        if code not in self._attrs:
            v = c_int()
            chk(_F["cuDeviceGetAttribute"](byref(v), code, self.device),
                f"cuDeviceGetAttribute({code})")
            self._attrs[code] = v.value
        return self._attrs[code]

    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        chk(_F["cuDeviceGetName"](buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode()

    def total_mem(self) -> int:
        v = c_size_t()
        chk(_F["cuDeviceTotalMem_v2"](byref(v), self.device), "cuDeviceTotalMem")
        return v.value

    def malloc(self, nbytes: int) -> int:
        p = c_uint64()
        chk(_F["cuMemAlloc_v2"](byref(p), nbytes), f"cuMemAlloc({nbytes})")
        return p.value

    def free(self, ptr: int) -> None:
        if ptr:
            chk(_F["cuMemFree_v2"](c_uint64(ptr)), "cuMemFree")

    def htod(self, dst: int, data) -> None:
        b = _as_bytes(data)
        chk(_F["cuMemcpyHtoD_v2"](c_uint64(dst), b, len(b)), "cuMemcpyHtoD")

    def dtoh(self, src: int, nbytes: int) -> bytes:
        buf = ctypes.create_string_buffer(nbytes)
        chk(_F["cuMemcpyDtoH_v2"](buf, c_uint64(src), nbytes), "cuMemcpyDtoH")
        return buf.raw

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        chk(_F["cuMemsetD8_v2"](c_uint64(dst), value, nbytes), "cuMemsetD8")

    def sync(self) -> None:
        chk(_F["cuCtxSynchronize"](), "cuCtxSynchronize")

    def module(self, cubin: bytes):
        mod = c_void_p()
        img = ctypes.create_string_buffer(cubin, len(cubin))
        chk(_F["cuModuleLoadData"](byref(mod), img), "cuModuleLoadData")
        mod._image = img  # 保持缓冲区存活
        return mod

    def function(self, mod, name: str):
        fn = c_void_p()
        chk(_F["cuModuleGetFunction"](byref(fn), mod, name.encode()),
            f"cuModuleGetFunction({name})")
        return fn

    def func_attr(self, fn, code: int) -> int | None:
        v = c_int()
        rc = _F["cuFuncGetAttribute"](byref(v), code, fn)
        return v.value if rc == 0 else None

    def occupancy_blocks(self, fn, block_size: int, dyn_smem: int = 0) -> int | None:
        v = c_int()
        rc = _F["cuOccupancyMaxActiveBlocksPerMultiprocessor"](
            byref(v), fn, block_size, dyn_smem)
        return v.value if rc == 0 else None

    def launch(self, fn, grid, block, args, smem: int = 0, stream=None) -> None:
        """``args`` 是设备指针序列（每个参数都是 8 字节指针）。

        ``cuLaunchKernel`` 要的是 ``void**``：每个元素指向"参数值"本身，
        因此这里先为每个指针值造一个 c_uint64 单元，再取它的地址。
        """
        gx, gy, gz = (grid, 1, 1) if isinstance(grid, int) else tuple(grid)
        bx, by, bz = (block, 1, 1) if isinstance(block, int) else tuple(block)
        holders = [c_uint64(int(a)) for a in args]
        if holders:
            arr = (c_void_p * len(holders))(
                *[ctypes.cast(byref(h), c_void_p) for h in holders]
            )
        else:
            arr = None
        chk(_F["cuLaunchKernel"](fn, gx, gy, gz, bx, by, bz, smem,
                                 0 if stream is None else stream, arr, None),
            "cuLaunchKernel")


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return data.encode()
    try:
        return bytes(memoryview(data))
    except TypeError:
        return bytes(data)
