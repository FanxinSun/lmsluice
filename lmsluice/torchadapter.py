"""Torch tensors, for the callers who have torch -- and the import that must not spread.

`safetensors.torch.load_file(path)` returns a dict of `torch.Tensor`. This
returns the same thing from the same call, so a loader that already exists
becomes an lmsluice loader by changing which module the name is imported from,
and every measurement in `MEASURED.md` becomes reachable without anybody
rewriting anything. Until this existed, the numbers on the front page were true
and unusable.

**Why torch is confined to this file, enforced by a test.** "Runs with no torch"
is not a boast, it is a recorded competitive difference: `docs/competition.md`
§4 lists ZipNN, tensorizer, Run:ai and fastsafetensors as unable to *import*
without a tensor framework, and §7 counts "no install required" as a row only
lmsluice ticks. On a phone, an iGPU laptop, or a box with no CUDA, `pip install
torch` is frequently the whole obstacle -- so the core must keep importing
nothing, and a single stray `import torch` in a core module would retire the
claim silently. `tests/test_lmsluice.py` walks the package source and fails if
any module but this one names torch, and imports the whole core with torch made
unavailable to prove the claim rather than assert it.

The import is also lazy *within* this module, so importing `lmsluice.torchadapter`
on a machine with no torch raises only when something is actually asked for, and
raises an error that says what to install.

**Both routes are zero-copy, which is the point of doing this here rather than
in a wrapper.** On the host, `torch.frombuffer` adopts the decoded bytes in
place: the tensors are views into the one buffer the transport already filled,
so a 1.87 GB checkpoint costs 1.87 GB and not twice that. On CUDA, the device
buffer is adopted through `__cuda_array_interface__` and reinterpreted per
tensor with `.view(dtype)`, so the bytes are never copied on the host at all.

That has one consequence a caller should know, and it is the one difference from
safetensors worth stating: **the tensors share one buffer, so the buffer lives as
long as the longest-lived tensor.** Dropping nine tensors out of ten frees
nothing. `safetensors` copies each tensor into its own allocation and so does
release them piecemeal; this trades that for not copying at all. Where the
piecewise behaviour is wanted, `Model.stream()` in `model.py` yields bounded
windows instead.
"""

from __future__ import annotations

from .model import open_model

# safetensors' dtype names, which is what the tensor index carries, against the
# torch attribute that means the same thing. Resolved by `getattr` rather than
# by importing the objects, so a torch too old to have `float8_e4m3fn` reports
# that dtype as unsupported on that version instead of failing to import.
_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8", "I8": "int8",
    "U16": "uint16", "I16": "int16", "F16": "float16", "BF16": "bfloat16",
    "U32": "uint32", "I32": "int32", "F32": "float32",
    "U64": "uint64", "I64": "int64", "F64": "float64",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}

HINT = ("torch is not installed. It is an optional extra -- lmsluice's core "
        "needs nothing at all -- so install it only if you want tensors back: "
        "pip install lmsluice[torch]")


def available() -> tuple[bool, str]:
    """(ok, why). Never raises, so a caller can offer this path conditionally."""
    try:
        import torch
    except Exception as exc:                  # noqa: BLE001 -- report, never raise
        return False, str(exc)
    return True, f"torch {torch.__version__}"


def _torch():
    """The only import of torch in this package."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(HINT) from exc
    return torch


def _dtype(torch, name: str, tensor_name: str):
    attr = _DTYPES.get(name)
    got = getattr(torch, attr, None) if attr else None
    if got is None:
        raise TypeError(
            f"{tensor_name!r} has dtype {name!r}, which this torch "
            f"({torch.__version__}) does not provide"
            + (f" as torch.{attr}" if attr else " and lmsluice does not map")
            + ". The bytes are readable with `lmsluice.model.open_model(...)"
            ".tensor(name)` regardless.")
    return got


def _shape(t) -> tuple:
    return tuple(t.shape) if t.shape else ()


def load_file(filename, device="cpu"):
    """Load a checkpoint into torch tensors. `safetensors.torch.load_file`'s call.

    Takes a plain `.safetensors` file or an lmsluice/lmz archive and decides
    between them the way every other load here does -- through `plan.py`,
    against the machine's own measured rates. On a fast NVMe with an archive
    that does not pay, the plan says so and the read is still correct; the
    decision is about speed, never about the bytes.

    `device` accepts what safetensors accepts: `"cpu"`, `"cuda"`, `"cuda:1"`, or
    a `torch.device`.
    """
    torch = _torch()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    m = open_model(str(filename))
    try:
        if dev.type == "cpu":
            # Deliberately not closed: on the mapped route the tensors are
            # views into the mapping, and closing it would leave them pointing
            # at unmapped memory. They hold the model open instead, and it is
            # released when the last tensor is. `Model.close()` would refuse
            # anyway while a memoryview is outstanding, which is the same
            # invariant enforced from the other side.
            return _host(m, torch)
        if dev.type == "cuda":
            out = _cuda(m, torch, dev)
            m.close()          # the bytes are in VRAM; the file is finished with
            return out
        raise ValueError(
            f"device {dev.type!r} is not one this adapter can fill. It moves "
            f"bytes to host memory or to CUDA; anything else is a copy the "
            f"caller can make from the CPU result.")
    except BaseException:
        m.close()
        raise


def stream_tensors(filename, *, budget: int | None = None):
    """Yield `(name, tensor)` a window at a time, holding a bounded amount.

    `load_file` materialises a whole checkpoint, which is what safetensors does
    and what most callers want. An inference server does not: vLLM's
    `model.load_weights` consumes a generator precisely so that a 70 B model
    never exists twice, once in the loader and once in the parameters. Feeding
    it a dict would defeat the design and put the whole checkpoint in host RAM.

    So this drives `Model.stream()`, which reads a window, yields the tensors
    inside it, and drops the window before reading the next. Each tensor holds
    its own window open, so memory falls as the consumer lets go rather than
    when this generator finishes -- a consumer that copies each tensor into a
    parameter and moves on, which is exactly what a weight loader does, never
    holds more than one window.
    """
    torch = _torch()
    m = open_model(str(filename))
    try:
        kw = {} if budget is None else {"budget": budget}
        for name, mv in m.stream(**kw):
            t = m.tensors[name]
            dt = _dtype(torch, t.dtype, name)
            if t.nbytes == 0:
                yield name, torch.empty(_shape(t), dtype=dt)
                continue
            ten = torch.frombuffer(mv, dtype=dt).reshape(_shape(t))
            # The slice is what keeps the window alive; naming it here means a
            # consumer that keeps the tensor keeps its bytes, and one that does
            # not lets the window go at the next iteration.
            ten._lmsluice_keepalive = mv
            yield name, ten
    finally:
        m.close()


def _host(m, torch, prefer_map: bool = True) -> dict:
    """Tensors over the plain bytes, by whichever of two routes fits.

    **A plain local file is mapped, not read**, which is what `safetensors`
    does and the reason it is so hard to beat on a warm cache: mapping moves no
    bytes at all, and pages the kernel already holds cost nothing to reach.
    Measured on this box warm, mapping is 21.8 GB/s against 3.3 for reading the
    same file through the transport -- a loader that ignored that would be 6.6x
    slower than the incumbent at the thing people benchmark first.

    **Anything else is read through the transport**: a coded archive cannot be
    mapped at all, and a remote source has nothing to map. That is also where
    the transport earns its keep, because the pipeline is what turns a slow
    link into a fast load.

    The trade is the usual one and it runs the other way when the cache is
    cold: mapping faults pages in one at a time on first touch, while the
    transport fetches them in parallel and has already been measured beating a
    cold read. So this is not "mapping is better", it is "mapping is better
    warm, and warm is the case where the file is local and already resident".
    A caller who wants the eager path regardless has `Model.load()`.
    """
    from .model import NotMappable

    base, mapped = None, False
    if prefer_map:
        try:
            base = m.map(writable=True)
            mapped = True
        except NotMappable:
            base = None
    if base is None:
        base = m.load()

    out = {}
    for name, t in m.tensors.items():
        dt = _dtype(torch, t.dtype, name)
        if t.nbytes == 0:
            out[name] = torch.empty(_shape(t), dtype=dt)
            continue
        # Mapped, the view spans the file and tensor offsets are absolute.
        # Read, the buffer covers the member and they are relative to `base`.
        mv = base[t.start:t.end] if mapped else m.tensor(name, buffer=base)
        ten = torch.frombuffer(mv, dtype=dt).reshape(_shape(t))
        # Belt and braces on the read path, load-bearing on the mapped one:
        # the memoryview is what holds the mapping open, and `Model.close()`
        # refuses while it does. A mapping dropped under a live tensor is a
        # segfault rather than an exception.
        ten._lmsluice_keepalive = (m, base) if mapped else base
        ten._lmsluice_mapped = mapped
        out[name] = ten
    _note(out, 0)
    return out


def _cuda(m, torch, dev) -> dict:
    """The device buffer, adopted and reinterpreted. No host copy at all."""
    from . import cuda

    ok, why = cuda.available()
    if not ok:
        raise RuntimeError(f"no CUDA device: {why}")
    ordinal = 0 if dev.index is None else dev.index
    ctx = cuda.Context(ordinal)
    try:
        buf = m.to_device(context=ctx)
    except Exception:
        ctx.close()
        raise

    # `|u1` is what the buffer publishes, because raw bytes are what it holds,
    # and each tensor is a slice of it reinterpreted to its own dtype.
    #
    # **Whether that reinterpretation can be free is decided by the file, not
    # by us.** A tensor may only be viewed as `dtype` at a byte offset that
    # `dtype`'s width divides -- CUDA requires natural alignment, and torch
    # refuses the view rather than emitting an access the hardware may fault
    # on. A `.safetensors` file puts its tensor data immediately after a JSON
    # header of whatever length the JSON happened to be: 925 bytes in the
    # fixture this was first run against, which divides by no dtype width at
    # all, so every tensor in that file is misaligned by the same 1 byte.
    #
    # There is no way to fix that without moving the bytes, which is why
    # safetensors, tensorizer and the rest all copy unconditionally. This
    # copies only where it must -- a device-to-device copy at VRAM bandwidth,
    # a few milliseconds for a whole checkpoint -- and stays free where the
    # layout allows. `copied_bytes` on the result says which happened, because
    # a caller counting VRAM deserves to know that the peak was briefly double.
    flat = torch.as_tensor(buf, device=f"cuda:{ordinal}")
    out, copied = {}, 0
    for name, t in m.tensors.items():
        dt = _dtype(torch, t.dtype, name)
        if t.nbytes == 0:
            out[name] = torch.empty(_shape(t), dtype=dt, device=f"cuda:{ordinal}")
            continue
        lo = t.start - m.base
        raw = flat[lo:lo + t.nbytes]
        if lo % dt.itemsize == 0:
            ten = raw.view(dt).reshape(_shape(t))
            # Load-bearing here rather than belt and braces: the allocation
            # belongs to `buf`, and `buf` holds the context. Let both go while
            # a tensor still points into VRAM and the next kernel reads freed
            # memory.
            ten._lmsluice_keepalive = buf
        else:
            ten = torch.empty(_shape(t), dtype=dt, device=f"cuda:{ordinal}")
            ten.view(torch.uint8).reshape(-1).copy_(raw)
            copied += t.nbytes
        out[name] = ten
    _note(out, copied)
    return out


def _note(out: dict, copied: int) -> None:
    """Record how many bytes had to be moved, on every tensor of the result.

    On a dict rather than returned separately so `load_file` keeps
    safetensors' signature exactly, and on the tensors rather than the dict so
    it survives `dict(result)`.
    """
    for ten in out.values():
        ten._lmsluice_copied_bytes = copied
