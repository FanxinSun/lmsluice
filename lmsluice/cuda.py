"""The device end of the path, through the CUDA driver and nothing else.

This package has no dependencies and is not about to acquire one to allocate a
buffer. So the device side is the **driver API** -- `libcuda.so.1`, which ships
with the driver itself -- reached through `ctypes`. Not the runtime API, which
lives in the CUDA toolkit: a machine with an NVIDIA GPU always has the former
and frequently lacks the latter, and requiring a toolkit to *load a model* would
repeat the mistake the probe was written to avoid.

Nothing here is a tensor library. The job is to put bytes in device memory and
hand back something a real tensor library can adopt without copying, which is
what `__cuda_array_interface__` is for: a dict of pointer, shape and dtype that
torch, cupy and numba all understand. lmsluice never imports any of them, and a
consumer that has one pays nothing to use it.

**Loading is done in a child process first, and that is not paranoia.** A CUDA
driver that is half-removed or mid-upgrade leaves `libcuda.so.1` on disk with an
initialiser that faults, and `ctypes.CDLL` on it takes the whole process down --
no exception, no return code, just a dead interpreter. lmz hit this and solved it
the same way; a segmentation fault in the caller's training script is not an
acceptable way to report "no GPU here."

What this does not do: decide what stays resident. It fills a device allocation
and returns it. Residency across VRAM, RAM and SSD is `vram/`'s charter and the
boundary is deliberate.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading

# Driver API result codes worth naming; the rest are reported by number.
CUDA_SUCCESS = 0
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101

# cuMemHostAlloc flags.
CU_MEMHOSTALLOC_PORTABLE = 0x01

_lib = None
_state = "not probed"
_why = ""
_lock = threading.Lock()

# The library name per platform. Linux and Windows ship the driver under these
# names; macOS has had no CUDA driver since 10.13, so it is absent by design
# rather than by omission -- Apple silicon's path is Metal, which is lmz's item.
_NAMES = {
    "linux": ("libcuda.so.1", "libcuda.so"),
    "win32": ("nvcuda.dll",),
    "darwin": (),
}


class CudaError(RuntimeError):
    """A driver call failed. Carries the numeric code the driver returned."""

    def __init__(self, fn: str, code: int):
        self.code = code
        super().__init__(f"{fn} failed with CUDA error {code}")


def _names() -> tuple:
    for key, names in _NAMES.items():
        if sys.platform.startswith(key):
            return names
    return ()


def _probe_elsewhere() -> tuple[bool, str]:
    """Load the driver in a process that is allowed to die.

    Returns (loadable, why not). A non-zero exit or a signal means the library
    is there and loading it is fatal, which is exactly the case that must not
    be discovered in the caller's process.
    """
    names = _names()
    if not names:
        return False, f"no CUDA driver on {sys.platform}"
    code = (
        "import ctypes,sys\n"
        f"names={names!r}\n"
        "for n in names:\n"
        "    try:\n"
        "        lib=ctypes.CDLL(n)\n"
        "    except OSError:\n"
        "        continue\n"
        "    r=lib.cuInit(0)\n"
        "    sys.exit(0 if r==0 else 10+min(r,200))\n"
        "sys.exit(9)\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], timeout=30,
                           capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not probe the driver: {exc}"
    if p.returncode == 0:
        return True, ""
    if p.returncode == 9:
        return False, f"driver library not found ({', '.join(names)})"
    if p.returncode < 0:
        return False, (f"loading the driver killed a child process with signal "
                       f"{-p.returncode} -- the driver is present but broken, "
                       f"probably mid-upgrade")
    return False, f"cuInit returned CUDA error {p.returncode - 10}"


def _load():
    global _lib, _state, _why
    with _lock:
        if _lib is not None or _state not in ("not probed",):
            return _lib
        if os.environ.get("LMSLUICE_NO_CUDA"):
            _state, _why = "disabled", "disabled by LMSLUICE_NO_CUDA"
            return None
        ok, why = _probe_elsewhere()
        if not ok:
            _state, _why = "unavailable", why
            return None
        for name in _names():
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if lib.cuInit(0) != CUDA_SUCCESS:
                continue
            _bind(lib)
            _lib, _state = lib, "ready"
            return lib
        _state, _why = "unavailable", "driver vanished between probe and load"
        return None


def _bind(lib) -> None:
    """Argument types for the calls used.

    Declared rather than left to ctypes' defaults because a device pointer is
    64-bit and an int is not: on a machine with more than 4 GB of VRAM,
    letting `cuMemAlloc` return a truncated handle produces a pointer that is
    wrong only for large allocations, which is the worst possible failure mode
    to debug.
    """
    P = ctypes.POINTER
    u64 = ctypes.c_ulonglong
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuDeviceGet.argtypes = [P(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.cuDeviceTotalMem_v2.argtypes = [P(ctypes.c_size_t), ctypes.c_int]
    lib.cuDeviceGetAttribute.argtypes = [P(ctypes.c_int), ctypes.c_int,
                                         ctypes.c_int]
    lib.cuDevicePrimaryCtxRetain.argtypes = [P(ctypes.c_void_p), ctypes.c_int]
    lib.cuDevicePrimaryCtxRelease_v2.argtypes = [ctypes.c_int]
    lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    lib.cuMemAlloc_v2.argtypes = [P(u64), ctypes.c_size_t]
    lib.cuMemFree_v2.argtypes = [u64]
    lib.cuMemHostAlloc.argtypes = [P(ctypes.c_void_p), ctypes.c_size_t,
                                   ctypes.c_uint]
    lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
    lib.cuMemcpyHtoDAsync_v2.argtypes = [u64, ctypes.c_void_p,
                                         ctypes.c_size_t, ctypes.c_void_p]
    lib.cuMemcpyHtoD_v2.argtypes = [u64, ctypes.c_void_p, ctypes.c_size_t]
    lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, u64, ctypes.c_size_t]
    lib.cuStreamCreate.argtypes = [P(ctypes.c_void_p), ctypes.c_uint]
    lib.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cuCtxSynchronize.argtypes = []
    lib.cuModuleLoadData.argtypes = [P(ctypes.c_void_p), ctypes.c_char_p]
    lib.cuModuleGetFunction.argtypes = [P(ctypes.c_void_p), ctypes.c_void_p,
                                        ctypes.c_char_p]
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, P(ctypes.c_void_p), P(ctypes.c_void_p)]


def _check(fn: str, code: int) -> None:
    if code != CUDA_SUCCESS:
        raise CudaError(fn, code)


def available() -> tuple[bool, str]:
    """(usable, why not). Never raises, never crashes the caller."""
    lib = _load()
    return (lib is not None), _why


def state() -> str:
    """What is known without going to look: 'not probed' until something does."""
    return _state


def device_info(ordinal: int = 0) -> dict:
    """Name, memory and PCIe generation of one device."""
    lib = _load()
    if lib is None:
        raise RuntimeError(f"no CUDA device: {_why}")
    dev = ctypes.c_int()
    _check("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(dev), ordinal))
    name = ctypes.create_string_buffer(256)
    _check("cuDeviceGetName", lib.cuDeviceGetName(name, 256, dev))
    total = ctypes.c_size_t()
    _check("cuDeviceTotalMem", lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev))
    out = {"ordinal": ordinal, "name": name.value.decode(errors="replace"),
           "total_bytes": int(total.value)}
    # 75 = COMPUTE_CAPABILITY_MAJOR, 76 = MINOR, in the driver's attribute enum.
    for attr, key in ((75, "cc_major"), (76, "cc_minor")):
        v = ctypes.c_int()
        if lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev) == CUDA_SUCCESS:
            out[key] = int(v.value)
    return out


class Context:
    """The device's primary context, retained rather than created.

    `cuCtxCreate` would make a second context that torch's own would not share,
    so a buffer filled here could not be adopted there without a copy -- which
    is the entire point of handing back `__cuda_array_interface__`. Retaining
    the primary context is what every other library on the machine also does.
    """

    def __init__(self, ordinal: int = 0):
        lib = _load()
        if lib is None:
            raise RuntimeError(f"no CUDA device: {_why}")
        self.lib = lib
        self.ordinal = ordinal
        dev = ctypes.c_int()
        _check("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(dev), ordinal))
        self.device = dev
        ctx = ctypes.c_void_p()
        _check("cuDevicePrimaryCtxRetain",
               lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
        self.handle = ctx
        _check("cuCtxSetCurrent", lib.cuCtxSetCurrent(ctx))
        self._closed = False

    def bind(self) -> None:
        """Make this context current on the calling thread.

        Every thread that touches the driver needs this, and the transport runs
        its stages on pools -- so a copy issued from a worker that never bound
        would fail with CUDA_ERROR_INVALID_CONTEXT rather than doing nothing.
        """
        _check("cuCtxSetCurrent", self.lib.cuCtxSetCurrent(self.handle))

    def allocate(self, nbytes: int) -> "DeviceBuffer":
        return DeviceBuffer(self, nbytes)

    def pinned(self, nbytes: int) -> "PinnedBuffer":
        return PinnedBuffer(self, nbytes)

    def stream(self) -> "Stream":
        return Stream(self)

    def synchronize(self) -> None:
        _check("cuCtxSynchronize", self.lib.cuCtxSynchronize())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.lib.cuDevicePrimaryCtxRelease_v2(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class Module:
    """A PTX module loaded through the driver, and its kernels.

    PTX rather than a cubin or a shared library: the driver JITs it for
    whatever card is present, so one artifact covers every architecture instead
    of one per `sm_`, and loading it needs no CUDA runtime -- which keeps the
    whole device path on the driver API that is always installed.
    """

    def __init__(self, ctx: "Context", ptx: bytes):
        self.ctx = ctx
        mod = ctypes.c_void_p()
        _check("cuModuleLoadData",
               ctx.lib.cuModuleLoadData(ctypes.byref(mod), ptx))
        self.handle = mod
        self._fns: dict[str, ctypes.c_void_p] = {}

    def function(self, name: str):
        fn = self._fns.get(name)
        if fn is None:
            fn = ctypes.c_void_p()
            _check("cuModuleGetFunction",
                   self.ctx.lib.cuModuleGetFunction(
                       ctypes.byref(fn), self.handle, name.encode()))
            self._fns[name] = fn
        return fn

    def launch(self, name: str, grid: int, block: int, args,
               stream: "Stream | None" = None) -> None:
        """Launch with `args` as a list of ctypes values, already typed.

        The driver takes an array of *pointers to* arguments, so every value
        has to outlive the call -- which it does here because the array holds
        references to the ctypes objects rather than to their addresses alone.
        """
        holders = list(args)
        arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        _check("cuLaunchKernel", self.ctx.lib.cuLaunchKernel(
            self.function(name), grid, 1, 1, block, 1, 1, 0,
            stream.handle if stream is not None else None, arr, None))


class Stream:
    """One CUDA stream, so copies can overlap the work that feeds them."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        h = ctypes.c_void_p()
        _check("cuStreamCreate", ctx.lib.cuStreamCreate(ctypes.byref(h), 0))
        self.handle = h

    def synchronize(self) -> None:
        _check("cuStreamSynchronize",
               self.ctx.lib.cuStreamSynchronize(self.handle))

    def close(self) -> None:
        if self.handle:
            self.ctx.lib.cuStreamDestroy_v2(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class PinnedBuffer:
    """Page-locked host memory, which is what makes a copy fast and async.

    A copy out of ordinary pageable memory is staged by the driver through its
    own pinned buffer, costing roughly half the link's rate and forbidding
    overlap. Pinning is not free -- it is a kernel-level page lock and a large
    one can fail -- so callers should size it as a staging window rather than as
    the whole model.
    """

    def __init__(self, ctx: Context, nbytes: int):
        self.ctx = ctx
        self.nbytes = nbytes
        p = ctypes.c_void_p()
        _check("cuMemHostAlloc",
               ctx.lib.cuMemHostAlloc(ctypes.byref(p), nbytes,
                                      CU_MEMHOSTALLOC_PORTABLE))
        self.ptr = p
        self._view = (ctypes.c_char * nbytes).from_address(p.value)

    def view(self) -> memoryview:
        """A writable memoryview, so a decoder can decode straight into it."""
        return memoryview(self._view).cast("B")

    def close(self) -> None:
        if self.ptr:
            self._view = None
            self.ctx.lib.cuMemFreeHost(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class DeviceBuffer:
    """Bytes in VRAM, and the interface that lets a tensor library adopt them.

    `__cuda_array_interface__` is the whole reason this class is worth having:
    `torch.as_tensor(buf, device="cuda")` and `cupy.asarray(buf)` both consume
    it with no copy and no import on this side. The buffer is exposed as raw
    `|u1` bytes because that is what it is -- what those bytes mean is the
    checkpoint's business, and a caller reshapes and reinterprets per tensor.
    """

    def __init__(self, ctx: Context, nbytes: int):
        self.ctx = ctx
        self.nbytes = nbytes
        d = ctypes.c_ulonglong()
        _check("cuMemAlloc", ctx.lib.cuMemAlloc_v2(ctypes.byref(d), nbytes))
        self.ptr = int(d.value)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {"shape": (self.nbytes,), "typestr": "|u1",
                "data": (self.ptr, False), "version": 3, "strides": None}

    def copy_from(self, src, offset: int = 0, stream: "Stream | None" = None,
                  nbytes: int | None = None) -> int:
        """Host to device. `src` is a buffer or a `PinnedBuffer`."""
        keepalive = None
        if isinstance(src, PinnedBuffer):
            host, n = src.ptr, nbytes if nbytes is not None else src.nbytes
        else:
            view = memoryview(src).cast("B")
            n = nbytes if nbytes is not None else len(view)
            if view.readonly:
                # ctypes will only take an address from a writable buffer, and
                # a copy is the only way to give it one. This is the
                # convenience path -- `bytes` handed in by a caller or a test.
                # The real routes always arrive with a pinned buffer or the
                # bytearray a decoder wrote into, and copy nothing. An async
                # copy out of this temporary would outlive it, so a read-only
                # source is always copied synchronously.
                keepalive = bytearray(view[:n])
                view, stream = memoryview(keepalive), None
            arr = (ctypes.c_char * n).from_buffer(view)
            keepalive = (keepalive, arr)
            host = ctypes.cast(arr, ctypes.c_void_p)
        if offset + n > self.nbytes:
            raise ValueError(f"copy of {n} at {offset} exceeds {self.nbytes}")
        dst = ctypes.c_ulonglong(self.ptr + offset)
        if stream is not None:
            _check("cuMemcpyHtoDAsync",
                   self.ctx.lib.cuMemcpyHtoDAsync_v2(dst, host, n,
                                                     stream.handle))
        else:
            _check("cuMemcpyHtoD",
                   self.ctx.lib.cuMemcpyHtoD_v2(dst, host, n))
        del keepalive
        return n

    def to_host(self, offset: int = 0, nbytes: int | None = None) -> bytearray:
        """Device to host, synchronous. This is how byte-identity is checked."""
        n = self.nbytes - offset if nbytes is None else nbytes
        out = bytearray(n)
        buf = (ctypes.c_char * n).from_buffer(out)
        _check("cuMemcpyDtoH",
               self.ctx.lib.cuMemcpyDtoH_v2(
                   ctypes.cast(buf, ctypes.c_void_p),
                   ctypes.c_ulonglong(self.ptr + offset), n))
        del buf
        return out

    def close(self) -> None:
        if self.ptr:
            self.ctx.lib.cuMemFree_v2(ctypes.c_ulonglong(self.ptr))
            self.ptr = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
