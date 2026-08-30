"""Decoding in VRAM: coded bytes cross the link, plaintext never does.

The route this package was built to reach. The other two send N plain bytes over
PCIe; this one sends fN coded bytes and turns them into plaintext where they
land, so it is the only route that shortens *both* links -- and its decode runs
on a processor that is not competing with the host for memory bandwidth, which
is the contention that put the host route 23% below its predicted rate.

## What lmz gives, and what it stops short of

`lmz_gpu_decode_batch_dev(hdr, streams, off, nstr, plane, out, stream, tpb)`
decodes a batch of equal-length rANS streams into device memory without
synchronising. That is the expensive half and it is lmz's, shipped, already
built on this machine.

What it does not do is put the bytes back in order. lmz splits a tensor by byte
position before coding -- plane p holds byte p of every element -- and the batch
decoder returns the planes plane-major. The original bytes are the transpose,
and `merge.cu` is that transpose, at 264 GB/s against the driver's own strided
copy at 1.87. It sits on lmz's side of the boundary and is here on loan; the
reasoning is in its own header.

## The format's own limit, which is the real blocker

This works on chunks whose planes are independent rANS streams -- `CODEC_SPLIT`
and `CODEC_BF16`. It does **not** work on `CODEC_BF16C`, the conditioned form,
where sign and mantissa are coded per exponent bucket and the sub-streams are
not independent.

That matters more than it sounds, because `CODEC_BF16C` is what lmz chooses for
BF16 weights whenever it wins on size, which on real LLM checkpoints is always.
So the flagship case -- a BF16 model -- currently cannot use the GPU decoder at
all, and the reason is not missing plumbing here: **lmz's best codec for the
data this project exists to move is one its own GPU kernel cannot read.**
Measured on a real archive: an fp32 checkpoint comes out 19/19 `CODEC_SPLIT`,
and a BF16 one 224/224 `CODEC_BF16C`.

Chunks this path cannot take are decoded on the host and copied over, so an
archive is never refused -- it just gets less of the win, and `Stats` says how
much.
"""

from __future__ import annotations

import ctypes
import os
import struct
import threading
import time

from . import cuda

# Format constants come from the adapter, which is the module allowed to know
# them (I4). This file is itself a registered loan -- see the loan register in
# `docs/boundary.md` -- so it may use them; nothing else in the package may.
from .lmzcodec import (CODEC_BF16, CODEC_BF16C, CODEC_SPLIT,  # noqa: E402
                       CODEC_SPLIT_ST, COND_MIN_ELEMS, METHOD_RANS,
                       METHOD_STORED)

GPU_CODECS = (CODEC_SPLIT, CODEC_BF16)
DEVICE_CODECS = GPU_CODECS

PTX_NAME = "lmz_merge.ptx"
SOURCE_NAME = "merge.cu"

# Threads per block for the merge. Not tuned -- the kernel is bandwidth-bound
# and measured flat from 128 to 512 on the one card available, so this is a
# reasonable default rather than a finding.
BLOCK = 256


class Stats:
    """What the run actually managed, which is the honest headline."""

    def __init__(self):
        self.chunks_on_device = 0
        self.chunks_on_host = 0
        self.bytes_on_device = 0
        self.bytes_on_host = 0
        self.coded_bytes_uploaded = 0
        self.upload_seconds = 0.0
        self.kernel_seconds = 0.0     # rANS decode plus merge, on the device
        self.why_host: dict[str, int] = {}

    def note_host(self, reason: str, nbytes: int) -> None:
        self.chunks_on_host += 1
        self.bytes_on_host += nbytes
        self.why_host[reason] = self.why_host.get(reason, 0) + 1

    @property
    def device_rate(self) -> float:
        """Plain bytes per second through the device stage alone.

        The number `vram_plan` wants for `device_decode`: the kernels only,
        with the read and the upload measured separately, because the plan
        prices them as separate stages that overlap.
        """
        return (self.bytes_on_device / self.kernel_seconds
                if self.kernel_seconds else 0.0)

    @property
    def device_share(self) -> float:
        total = self.bytes_on_device + self.bytes_on_host
        return self.bytes_on_device / total if total else 0.0

    def __repr__(self) -> str:
        return (f"<Stats device={self.chunks_on_device} host={self.chunks_on_host} "
                f"share={self.device_share:.1%} why={self.why_host}>")


def _rec(chunk):
    """The codec's own record for a unit.

    `devdecode` is a loan and is allowed to look inside; every other module
    sees only the `Unit` fields. Kept as one accessor so the places that do
    are countable.
    """
    return getattr(chunk, "opaque", None) or chunk


def planes_of(chunk, payload):
    """(nplanes, nelem, [(method, offset, length)]) for a split chunk.

    The payload is `nplanes` little-endian uint32 lengths followed by the plane
    data in order, and the chunk's 16 flag bits carry two per plane saying how
    each was coded. Read here rather than through lmz because the whole point
    is to hand the coded bytes to the device without materialising them.
    """
    rec = _rec(chunk)
    nplanes = 2 if rec.codec == CODEC_BF16 else rec.esize
    if nplanes <= 0 or chunk.rlen % nplanes:
        return None
    head = struct.Struct(f"<{nplanes}I")
    if len(payload) < head.size:
        return None
    lengths = head.unpack_from(payload, 0)
    out, off = [], head.size
    for p, ln in enumerate(lengths):
        method = (rec.flags >> (2 * p)) & 3
        out.append((method, off, ln))
        off += ln
    if off > len(payload):
        return None
    return nplanes, chunk.rlen // nplanes, out


def gpu_chunk_size(esize: int = 2) -> int:
    """The largest chunk size that keeps an archive's chunks GPU-decodable.

    **LOAN — lmz's, held here.** See the loan register in `docs/boundary.md`.
    This is an *encoder-side* threshold living in the transport layer: it tells
    a writer which flag to set so that the decoder it will later use can read
    the result. Invariant I3 says the archive should declare which decoders can
    read it, and then nothing here would need to know. Retired by a per-archive
    decoder-capability declaration in lmz's manifest, at which point
    `LmzCodec.capabilities()` reads it instead of deriving it and this function
    goes away.

    The whole of the BF16 fix, and it is a writer-side choice rather than a
    decoder-side one. Below this, lmz never reaches its conditioned codec and
    emits `CODEC_BF16` instead -- two independent planes, which the device
    decoder reads. Measured on a 1.87 GB Llama checkpoint the difference is
    32.9% saved against 32.7%, so **two tenths of a point of ratio buys the
    entire GPU decode path**.
    """
    try:
        from ._lmz import require

        (codec,) = require("codec")
        floor = getattr(codec, "COND_MIN_ELEMS", COND_MIN_ELEMS)
    except Exception:                     # noqa: BLE001
        floor = COND_MIN_ELEMS
    return floor * esize // 2


def advice(archive) -> str:
    """What to do about an archive the device decoder cannot read, or ""."""
    from . import cuda

    ok, _why = cuda.available()
    if not ok:
        return ""
    chunks = archive.chunks_for([(0, archive.plain_bytes)])
    cond = [c for c in chunks if getattr(c, "opaque", c).codec == CODEC_BF16C]
    if not cond:
        return ""
    esize = getattr(cond[0], "opaque", cond[0]).esize or 2
    size = gpu_chunk_size(esize)
    return (f"{len(cond)} of {len(chunks)} chunks use lmz's conditioned BF16 "
            f"codec, which no GPU kernel can read, so the device route is "
            f"unavailable for them. Recompressing with "
            f"chunk_size={size >> 20} MiB or less makes the whole archive "
            f"GPU-decodable; measured cost on a 1.87 GB checkpoint was 0.2 "
            f"points of ratio.")


_cache: dict = {}


def for_context(ctx) -> "DeviceDecoder":
    """One decoder per context, kept.

    Building it probes lmz, reads the PTX and hands it to the driver to JIT --
    a fixed cost of roughly half a second that was being paid on every single
    call to `to_device`, which is to say once per model load. It is the same
    work every time and the result is immutable, so it is built once and kept
    against the context it was loaded into.
    """
    key = id(ctx)
    dd = _cache.get(key)
    if dd is None:
        dd = _cache[key] = DeviceDecoder(ctx)
    return dd


class DeviceDecoder:
    """lmz's device decoder plus the merge, driven from this package."""

    def __init__(self, ctx: "cuda.Context"):
        self.ctx = ctx
        self.lib = None
        self.module = None
        self.entry = ""
        self.why = ""
        self.grain = 128
        self.pad = 576
        self._prepare()

    # -- setting up --------------------------------------------------------
    def _prepare(self) -> None:
        try:
            from ._lmz import require

            # `lmz.gpu` is a submodule, not an attribute: importing `lmz` does
            # not bind it, deliberately, because probing for a GPU builds a
            # library and that is not what `import lmz` should do.
            (gpu,) = require("gpu")
        except Exception as exc:          # noqa: BLE001
            self.why = f"lmz.gpu not importable: {exc}"
            return
        ok, why = gpu.available()
        if not ok:
            self.why = f"lmz.gpu unavailable: {why}"
            return
        # lmz publishes `decode_batch_dev` as of bd3fb42 -- device pointers
        # throughout, allocating nothing, synchronising nothing. Prefer it:
        # it is public, it is the codec's own declaration of the call, and it
        # removes the second ctypes binding to a symbol lmz already binds.
        #
        # Guarded rather than assumed, because that commit is on the
        # `shared-frequency-tables` branch and a consumer with a released lmz
        # will not have it. The ctypes path below is the fallback and is
        # deletable once the branch lands; `self.entry` says which is live,
        # the way `Archive.route` does.
        self.gpu = gpu
        if hasattr(gpu, "decode_batch_dev"):
            self.entry = "lmz.gpu.decode_batch_dev"
            self.lib = gpu._load()
            if self.lib is None:
                self.why = "lmz.gpu would not load"
                return
        else:
            lib = gpu._load()
            if lib is None or not hasattr(lib, "lmz_gpu_decode_batch_dev"):
                self.why = "this lmz build has no device-pointer entry point"
                return
            lib.lmz_gpu_decode_batch_dev.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_uint]
            self.entry = "ctypes lmz_gpu_decode_batch_dev (fallback)"
            self.lib = lib
        self.grain, self.pad = gpu.grain(), gpu.pad_bytes()
        ptx = self._ptx()
        if ptx is None:
            self.lib = None
            return
        try:
            self.module = cuda.Module(self.ctx, ptx)
        except Exception as exc:          # noqa: BLE001
            self.lib, self.why = None, f"the merge kernel would not load: {exc}"

    def _ptx(self) -> bytes | None:
        """The merge kernel, compiled once if nvcc is here.

        Built beside the package and gitignored, which is the bargain lmz
        already makes with a C compiler and with nvcc. No nvcc means no device
        route, not a failure: the host routes are unaffected and say so.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        ptx, src = os.path.join(here, PTX_NAME), os.path.join(here, SOURCE_NAME)
        if os.path.exists(ptx) and (not os.path.exists(src)
                                    or os.path.getmtime(ptx) >= os.path.getmtime(src)):
            with open(ptx, "rb") as fh:
                return fh.read()
        import shutil
        import subprocess

        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(nvcc) and shutil.which("nvcc") is None:
            self.why = "no nvcc, so the merge kernel cannot be built"
            return None
        try:
            subprocess.run([nvcc, "-ptx", "-O3", "-o", ptx, src], check=True,
                           capture_output=True, timeout=180)
        except Exception as exc:          # noqa: BLE001
            self.why = f"nvcc failed on the merge kernel: {exc}"
            return None
        with open(ptx, "rb") as fh:
            return fh.read()

    @property
    def available(self) -> bool:
        return self.lib is not None and self.module is not None

    # -- deciding ----------------------------------------------------------
    @staticmethod
    def _coded_mask(chunk, nplanes) -> tuple:
        """Which planes are rANS, as a tuple of plane indices."""
        rec = _rec(chunk)
        return tuple(p for p in range(nplanes)
                     if ((rec.flags >> (2 * p)) & 3) == METHOD_RANS)

    def takes(self, chunk) -> str:
        """"" if this chunk can be decoded on the device, else why not."""
        rec = _rec(chunk)
        if rec.codec == CODEC_SPLIT_ST:
            return "shared-table chunks need a table the kernel is not given here"
        if rec.codec not in GPU_CODECS:
            if rec.codec == CODEC_BF16C:
                return ("conditioned bf16 (CODEC_BF16C): its sub-streams are "
                        "not independent rANS, and no shipped kernel reads it")
            return f"codec {rec.codec} is not a plane split"
        nplanes = 2 if rec.codec == CODEC_BF16 else rec.esize
        if nplanes <= 0 or chunk.rlen % nplanes:
            return "chunk length is not a whole number of elements"
        nelem = chunk.rlen // nplanes
        if nelem % self.grain:
            return f"plane of {nelem} is not a multiple of the grain {self.grain}"
        if nplanes > 16:
            return f"{nplanes} planes is more than the merge takes"
        for p in range(nplanes):
            if ((rec.flags >> (2 * p)) & 3) not in (METHOD_RANS, METHOD_STORED):
                return "a plane is zstd- or deflate-coded, which the kernel "\
                       "does not decode"
        coded = self._coded_mask(chunk, nplanes)
        if rec.codec == CODEC_BF16 and coded != (0,) and len(coded) != nplanes:
            return "an unexpected plane mask for a bf16 field split"
        return ""

    # -- doing it ----------------------------------------------------------
    def decode(self, archive, chunks, out: "cuda.DeviceBuffer", base: int,
               stream: "cuda.Stream | None" = None) -> Stats:
        """Decode `chunks` straight into `out`, which begins at plain `base`.

        Chunks the device cannot take are decoded on the host and copied, so
        the caller gets the whole range either way and `Stats` records what
        fraction really went the fast way.
        """
        stats = Stats()
        if not self.available:
            raise RuntimeError(self.why or "no device decoder")

        groups: dict[int, list] = {}
        for chunk in chunks:
            why = self.takes(chunk)
            if why:
                stats.note_host(why, chunk.rlen)
                self._host_chunk(archive, chunk, out, base, stream)
                continue
            rec = _rec(chunk)
            nplanes = 2 if rec.codec == CODEC_BF16 else rec.esize
            # Keyed by both, so a batch is one index space in the merge: every
            # job has the same element count and the same width.
            # Keyed by codec too: CODEC_BF16 reassembles on bfloat16's field
            # boundaries and CODEC_SPLIT on byte boundaries, so two chunks of
            # the same shape still need different merges.
            # The method mask is part of the key: which planes are coded and
            # which are stored decides where each plane's bytes come from, so
            # jobs in one batch must agree on it.
            mask = rec.flags & ((1 << (2 * nplanes)) - 1)
            groups.setdefault((chunk.rlen // nplanes, nplanes, rec.codec,
                               mask), []).append(chunk)

        for (nelem, _nplanes, _codec, _mask), batch in groups.items():
            self._device_group(archive, batch, nelem, out, base, stream, stats)
        return stats

    # Coded bytes one job of the fetch stage should carry. The same figure the
    # host routes use, for the same reason: enough that dispatching it costs
    # nothing next to the read.
    FETCH_BYTES = 4 << 20

    def _fetch_all(self, archive, chunks) -> dict:
        """Every chunk's payload, read in parallel and coalesced.

        Reading them one at a time was what held this route back once the
        kernels were fixed: an archive in 1 MiB blocks has 1787 of them, and
        1787 serial `pread`s at queue depth one left the decode -- by then
        running at 81 GB/s -- waiting on the disk. This reuses the same
        run-coalescing and the same two-stage transport as every other route,
        which is what they were written for.
        """
        from .probe import default_threads
        from .transport import transport

        runs = archive.runs(sorted(chunks, key=lambda c: c.off),
                            self.FETCH_BYTES)
        out: dict[int, memoryview] = {}
        lock = threading.Lock()
        src = archive.source
        into = getattr(src, "pread_into", None)

        def fetch(run):
            # Into a writable buffer, so the stored planes inside it can go to
            # the device without being copied again. `pread` returns immutable
            # bytes and would force exactly the copy this route cannot afford.
            if into is None:
                return bytearray(archive.fetch(run))
            buf = bytearray(run.span)
            if into(buf, run.off) != run.span:
                raise EOFError(f"{src.name}: short read at {run.off}")
            return buf

        def place(run, payload):
            view = memoryview(payload)
            with lock:
                for c in run.chunks:
                    a = c.off - run.off
                    out[c.off] = view[a:a + c.clen]
            return len(payload)

        fetch_threads, _ = default_threads()
        transport(runs, fetch, place, fetch_threads=fetch_threads,
                  place_threads=1, inflight=fetch_threads * 2)
        return out

    def _decode_batch_dev(self, streams, offsets, nstr, plane, out, stream):
        """The device decode, through whichever entry point this lmz offers."""
        handle = stream.handle if stream is not None else None
        if self.entry.startswith("lmz.gpu"):
            return self.gpu.decode_batch_dev(
                streams, offsets, nstr, plane, out, header=None,
                stream=(handle.value if handle is not None else None))
        return self.lib.lmz_gpu_decode_batch_dev(
            None, ctypes.c_void_p(streams), ctypes.c_void_p(offsets),
            nstr, plane, ctypes.c_void_p(out), handle, 0)

    def _host_chunk(self, archive, chunk, out, base, stream) -> None:
        buf = bytearray(chunk.rlen)
        archive.place_one(chunk, buf)
        out.copy_from(buf, offset=chunk.dst - base, stream=None)

    def _device_group(self, archive, chunks, nelem, out, base, stream,
                      stats: Stats) -> None:
        """One batch: every rANS plane of these chunks, decoded in one call."""
        # Coded planes go into the rANS batch; stored planes go over as
        # themselves, packed in the same job order so the merge can take the
        # two from separate bases. Nothing is scattered afterwards -- an
        # earlier version wove stored planes in with a byte-wise kernel, which
        # is wrong for CODEC_BF16 because its planes are bit fields rather than
        # bytes, and produced weights that were the right length and not the
        # right values.
        # Sized first, then filled through memoryviews. Growing a bytearray
        # by `+=` for every plane copies a gigabyte through the interpreter
        # one slice at a time and was, once the kernels reached 88 GB/s, the
        # largest single cost left in the route.
        payloads = self._fetch_all(archive, chunks)
        parsed, coded_total, stored_total, nstr = [], 0, 0, 0
        for chunk in chunks:
            got = planes_of(chunk, payloads[chunk.off])
            if got is None:
                stats.note_host("plane header did not parse", chunk.rlen)
                self._host_chunk(archive, chunk, out, base, stream)
                continue
            nplanes, _n, planes = got
            parsed.append((chunk, nplanes, planes))
            for method, _off, ln in planes:
                if method == METHOD_RANS:
                    coded_total += ln
                    nstr += 1
                else:
                    stored_total += ln

        blob = bytearray(coded_total)
        bview = memoryview(blob)
        stored_at: list[tuple] = []      # (device offset, source view)
        offs: list[int] = []
        layout: list[tuple] = []
        bpos = spos = 0
        slot = 0
        for chunk, nplanes, planes in parsed:
            layout.append((chunk, nplanes, slot))
            payload = payloads[chunk.off]
            for method, off, ln in planes:
                if method == METHOD_RANS:
                    bview[bpos:bpos + ln] = payload[off:off + ln]
                    offs += [bpos, ln]
                    bpos += ln
                    slot += 1
                else:
                    if ln != nelem:
                        raise RuntimeError(
                            f"stored plane is {ln} bytes, expected {nelem}")
                    # Uploaded straight from the read buffer. Concatenating
                    # these into one host array first cost 0.49 s on a 1.87 GB
                    # model, which was half the route.
                    stored_at.append((spos, payload[off:off + ln]))
                    spos += ln
        offsets = struct.pack(f"<{len(offs)}Q", *offs) if offs else b""
        if not layout:
            return

        d_streams = self.ctx.allocate(len(blob) + self.pad)
        d_off = self.ctx.allocate(len(offsets))
        d_planes = self.ctx.allocate(max(1, nstr * nelem))
        d_stored = self.ctx.allocate(max(1, stored_total))
        try:
            started = time.perf_counter()
            d_streams.copy_from(blob, stream=stream)
            d_off.copy_from(offsets, stream=stream)
            for at, view in stored_at:
                d_stored.copy_from(view, offset=at, stream=stream)
            if stream is not None:
                stream.synchronize()
            stats.upload_seconds += time.perf_counter() - started
            # Stored planes cross PCIe as plain bytes and are counted, or the
            # link saving this route is sold on would look better than it is.
            stats.coded_bytes_uploaded += len(blob) + stored_total

            started = time.perf_counter()
            rc = self._decode_batch_dev(d_streams.ptr, d_off.ptr, nstr,
                                        nelem, d_planes.ptr, stream)
            if rc != 0:
                raise RuntimeError(f"lmz_gpu_decode_batch_dev returned {rc}")

            # One launch for the whole batch. Per-chunk launches cost more
            # than the merge itself once an archive has thousands of blocks.
            dsts = b"".join(struct.pack("<Q", c.dst - base) for c, _n, _f in layout)
            d_dst = self.ctx.allocate(len(dsts))
            try:
                d_dst.copy_from(dsts)
                nplanes = layout[0][1]
                total = nelem * len(layout)
                args = [ctypes.c_ulonglong(d_planes.ptr),
                        ctypes.c_ulonglong(d_dst.ptr),
                        ctypes.c_ulonglong(out.ptr)]
                first_chunk = layout[0][0]
                coded_planes = self._coded_mask(first_chunk, nplanes)
                if _rec(first_chunk).codec == CODEC_BF16 and stored_total:
                    name = "lmz_merge_bf16_split"
                    args = [ctypes.c_ulonglong(d_planes.ptr),
                            ctypes.c_ulonglong(d_stored.ptr),
                            ctypes.c_ulonglong(d_dst.ptr),
                            ctypes.c_ulonglong(out.ptr),
                            ctypes.c_ulonglong(nelem),
                            ctypes.c_uint(len(layout))]
                    self.module.launch(name, (total + BLOCK - 1) // BLOCK,
                                       BLOCK, args, stream)
                elif _rec(first_chunk).codec == CODEC_BF16:
                    self.module.launch(
                        "lmz_merge_bf16_batch",
                        (total + BLOCK - 1) // BLOCK, BLOCK,
                        [ctypes.c_ulonglong(d_planes.ptr),
                         ctypes.c_ulonglong(d_dst.ptr),
                         ctypes.c_ulonglong(out.ptr),
                         ctypes.c_ulonglong(nelem),
                         ctypes.c_uint(len(layout))], stream)
                elif stored_total:
                    # Mixed: some planes coded, some stored. One descriptor
                    # per plane, shared by every job in the batch.
                    src, ci, si = [], 0, 0
                    for p in range(nplanes):
                        if p in coded_planes:
                            src.append(ci); ci += 1
                        else:
                            src.append(-(si + 1)); si += 1
                    d_src = self.ctx.allocate(4 * nplanes)
                    try:
                        d_src.copy_from(struct.pack(f"<{nplanes}i", *src))
                        self.module.launch(
                            "lmz_merge_masked", (total + BLOCK - 1) // BLOCK,
                            BLOCK,
                            [ctypes.c_ulonglong(d_planes.ptr),
                             ctypes.c_ulonglong(d_stored.ptr),
                             ctypes.c_ulonglong(d_dst.ptr),
                             ctypes.c_ulonglong(d_src.ptr),
                             ctypes.c_ulonglong(out.ptr),
                             ctypes.c_uint(nplanes), ctypes.c_uint(ci),
                             ctypes.c_uint(si), ctypes.c_ulonglong(nelem),
                             ctypes.c_uint(len(layout))], stream)
                        if stream is not None:
                            stream.synchronize()
                        else:
                            self.ctx.synchronize()
                    finally:
                        d_src.close()
                else:
                    self.module.launch(
                        "lmz_merge_batch", (total + BLOCK - 1) // BLOCK, BLOCK,
                        [ctypes.c_ulonglong(d_planes.ptr),
                         ctypes.c_ulonglong(d_dst.ptr),
                         ctypes.c_ulonglong(out.ptr),
                         ctypes.c_uint(nplanes), ctypes.c_ulonglong(nelem),
                         ctypes.c_uint(len(layout))], stream)
                if stream is not None:
                    stream.synchronize()
                else:
                    self.ctx.synchronize()
            finally:
                d_dst.close()
            for chunk, _npl, _first in layout:
                stats.chunks_on_device += 1
                stats.bytes_on_device += chunk.rlen
            if stream is not None:
                stream.synchronize()
            else:
                self.ctx.synchronize()
            stats.kernel_seconds += time.perf_counter() - started
        finally:
            d_streams.close()
            d_off.close()
            d_planes.close()
            d_stored.close()

    def _merge(self, d_planes, first: int, nplanes: int, nelem: int,
               out, dst_off: int, stream) -> None:
        src = d_planes.ptr + first * nelem
        dst = out.ptr + dst_off
        grid = (nelem + BLOCK - 1) // BLOCK
        name = {2: "lmz_merge_2", 4: "lmz_merge_4",
                8: "lmz_merge_8"}.get(nplanes)
        # The vectorised kernels assemble an element and store it in one
        # transaction, which requires the destination to be aligned to that
        # width. A chunk's destination is a byte offset into the member and is
        # under no obligation to be: safetensors puts its payload after a
        # JSON header of arbitrary length, so whether this holds is decided by
        # how long the header happens to be. Misaligned, the vector store is
        # undefined -- it faulted the whole context in testing -- so the byte
        # kernel takes those, which is correct at any address.
        if name and dst % nplanes:
            name = None
        if name:
            args = [ctypes.c_ulonglong(src), ctypes.c_ulonglong(dst),
                    ctypes.c_ulonglong(nelem)]
        else:
            name = "lmz_merge_n"
            args = [ctypes.c_ulonglong(src), ctypes.c_ulonglong(dst),
                    ctypes.c_uint(nplanes), ctypes.c_ulonglong(nelem)]
        self.module.launch(name, grid, BLOCK, args, stream)

    def _scatter(self, data: bytes, plane: int, nplanes: int, nelem: int,
                 out, dst_off: int, stream) -> None:
        if len(data) != nelem:
            raise RuntimeError(f"stored plane is {len(data)}, expected {nelem}")
        tmp = self.ctx.allocate(nelem)
        try:
            tmp.copy_from(data)
            # Counted: a stored plane crosses PCIe as plain bytes, and leaving
            # it out of the tally would make the link saving look better than
            # it is -- which is the one number this route is sold on.
            self._uploaded = getattr(self, "_uploaded", 0) + len(data)
            grid = (nelem + BLOCK - 1) // BLOCK
            self.module.launch(
                "lmz_scatter_1", grid, BLOCK,
                [ctypes.c_ulonglong(tmp.ptr), ctypes.c_ulonglong(out.ptr + dst_off),
                 ctypes.c_uint(nplanes), ctypes.c_uint(plane),
                 ctypes.c_ulonglong(nelem)], stream)
            if stream is not None:
                stream.synchronize()
            else:
                self.ctx.synchronize()
        finally:
            tmp.close()
