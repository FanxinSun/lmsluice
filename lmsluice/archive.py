"""An lmz archive, addressed as work a transport can move.

lmz owns the coded stream, its tables, its blocks and its index; this package
owns dispatch, queue depth, staging and overlap. That boundary is drawn in
`lmz/docs/gpu-residency-handover.md` and this module is where the two meet, so
every use of lmz's internals is here and nowhere else.

Two of them are internals rather than public API -- decoding one chunk, and
the table set a v7 archive shares across chunks -- because lmz exposes
decoding a *file* and reading a *byte range*, and neither is the shape a
pipeline wants. Reaching in is checked rather than assumed: `Archive` probes
for what it needs at construction and falls back to `lmz.MappedArchive`, which
is public and stable, if the internals have moved. The fallback is correct and
slower, and `Archive.route` says which one is in use so a benchmark can never
quietly measure the wrong one.

The unit of work is a *run*: consecutive chunks whose payloads sit together in
the file, fetched in one read and decoded one at a time. Runs rather than
chunks because a page-mapped archive has tens of thousands of 64 KiB blocks
and asking the kernel for each one separately spends more on system calls than
on bytes; runs rather than the whole file because memory is bounded and
because a caller may want four tensors out of four hundred.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field


class Unsupported(RuntimeError):
    """Raised when this build of lmz cannot serve the pipeline at all."""


@dataclass
class Run:
    """One read, and the chunks it satisfies."""

    off: int                       # first payload byte in the archive
    span: int                      # bytes to read, padding included
    chunks: list = field(default_factory=list)
    plain: int = 0                 # plain bytes the run produces

    def __len__(self) -> int:
        return self.span


@dataclass(frozen=True)
class Tensor:
    """Where one tensor lives in the restored output."""

    name: str
    dtype: str
    shape: tuple
    member: str
    start: int                     # absolute offset in the archive's output
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


# How far apart two payloads may sit and still be read together. An aligned
# archive pads between payloads, and reading the padding is cheaper than
# issuing a second request for what is on the other side of it -- but only up
# to a point, and the point is a fraction of the useful bytes rather than a
# fixed size, so it scales with the run instead of with any one device.
COALESCE_SLACK = 1.25
COALESCE_MAX = 32 << 20


class Archive:
    """An lmz archive opened for transport rather than for extraction."""

    def __init__(self, target, *, source=None):
        from ._lmz import require
        from .source import Source, open_source

        (fmt,) = require("format")
        ArchiveReader = fmt.ArchiveReader

        if source is None:
            source = target if isinstance(target, Source) else open_source(target)
        self.source = source
        self.path = source.name
        try:
            self._reader = ArchiveReader(source.reader())
        except BaseException:
            source.close()
            raise
        self.coded_bytes = source.size
        self.members = self._reader.members
        self.plain_bytes = sum(m.size for m in self.members)
        self._mapped = None
        self._decode, self.route = self._decoder()

        # Destination order, materialised once. The table is the index and a
        # loader walks all of it, so the per-access unpack that keeps opening
        # cheap for a mount is the wrong trade here -- lmz's own ChunkTable
        # docstring says as much.
        self._chunks = sorted(self._reader.chunks, key=lambda c: c.dst)
        self._starts = [c.dst for c in self._chunks]

    # -- what it holds ----------------------------------------------------
    @property
    def ratio(self) -> float:
        """Coded bytes per plain byte. Below 1 is a saving."""
        return self.coded_bytes / self.plain_bytes if self.plain_bytes else 1.0

    @property
    def block_bytes(self) -> int:
        """The largest block: what a one-byte read costs to satisfy."""
        return max((c.rlen for c in self._chunks), default=0)

    @property
    def manifest(self) -> dict:
        return self._reader.manifest

    def tensors(self) -> dict:
        """Every tensor the archive knows about, by name.

        Only populated for members lmz parsed -- safetensors and GGUF. A raw
        member has no tensor index and can only be addressed by byte range,
        which `span_of` still does.
        """
        out = {}
        for m in self.members:
            for name, meta in (m.tensors or {}).items():
                start, end = meta["offsets"]
                out[name] = Tensor(name, meta.get("dtype", ""),
                                   tuple(meta.get("shape", ())), m.path,
                                   m.dst + start, m.dst + end)
        return out

    def span_of(self, member: str | None = None) -> tuple[int, int]:
        """The plain byte range of one member, or of everything."""
        if member is None:
            return 0, self.plain_bytes
        for m in self.members:
            if m.path == member:
                return m.dst, m.dst + m.size
        raise KeyError(f"no member named {member!r}")

    # -- planning work ----------------------------------------------------
    def chunks_for(self, spans) -> list:
        """Every chunk touching any of `spans`, once, in file order.

        File order rather than destination order because this is what the
        fetch stage will read, and a device is fastest when asked for its
        bytes in the order it holds them. Destination order is what the
        decode writes and it needs no help: each chunk already knows where it
        lands.
        """
        wanted = {}
        for lo, hi in spans:
            lo, hi = max(0, lo), min(self.plain_bytes, hi)
            if lo >= hi:
                continue
            i = bisect_right(self._starts, lo) - 1
            if i < 0:
                i = 0
            while i < len(self._chunks) and self._chunks[i].dst < hi:
                c = self._chunks[i]
                if c.dst + c.rlen > lo:
                    wanted[c.off] = c
                i += 1
        return [wanted[k] for k in sorted(wanted)]

    def cover(self, spans, target: int):
        """(lo, hi, runs) -- the plain range the blocks holding `spans` fill.

        A block is the smallest thing that can be decoded, and it does not
        stop where a tensor does: asking for a 200-byte bias means decoding
        whatever block it sits in, which begins before it and ends after it.
        So a caller that allocates only the bytes it asked for has allocated
        too few, and the decode runs off both ends of the buffer.

        Returning the covered range rather than clipping to the requested one
        keeps that arithmetic in one place, and it is also the honest number:
        `hi - lo` is what this read really costs to decode, which is the
        amplification a page-mapped archive exists to keep small.
        """
        chunks = self.chunks_for(spans)
        if not chunks:
            return 0, 0, []
        lo = min(c.dst for c in chunks)
        hi = max(c.dst + c.rlen for c in chunks)
        return lo, hi, self.runs(chunks, target)

    def runs(self, chunks, target: int) -> list[Run]:
        """Group chunks into reads of roughly `target` coded bytes each.

        A gap is absorbed when it is small next to the run it would split, and
        starts a new run when it is not. The comparison is against the useful
        bytes rather than against a size, so the same rule holds for a
        64 KiB-block archive and an 8 MiB-block one without either being
        named here.
        """
        runs: list[Run] = []
        cur: Run | None = None
        for c in chunks:
            if cur is not None:
                end = cur.off + cur.span
                gap = c.off - end
                grown = (c.off + c.clen) - cur.off
                if (gap >= 0 and grown <= COALESCE_MAX
                        and grown <= max(target, int((cur.span + c.clen)
                                                     * COALESCE_SLACK))):
                    cur.span = grown
                    cur.chunks.append(c)
                    cur.plain += c.rlen
                    if cur.span >= target:
                        cur = None
                    continue
            cur = Run(off=c.off, span=c.clen, chunks=[c], plain=c.rlen)
            runs.append(cur)
            if cur.span >= target:
                cur = None
        return runs

    # -- doing it ---------------------------------------------------------
    def fetch(self, run: Run) -> bytes:
        """One read. Called on the fetch stage, so it must release the GIL."""
        return self.source.pread(run.off, run.span)

    def place(self, run: Run, payload: bytes, out, base: int, clip=None) -> int:
        """Decode a run into `out`, which starts at plain offset `base`.

        `clip` names the plain range `out` actually owns. A block that reaches
        past either end is decoded into scratch and only the part inside is
        copied -- the alternative, refusing such a block, would make every
        partial read fail at its edges, which is every partial read.
        """
        n = 0
        view = memoryview(payload)
        dst = memoryview(out)
        lo_ok, hi_ok = clip if clip is not None else (base, base + len(dst))
        for c in run.chunks:
            a = c.off - run.off
            src = view[a:a + c.clen]
            lo, hi = c.dst, c.dst + c.rlen
            if lo >= lo_ok and hi <= hi_ok:
                self._decode(c, src, dst, lo - base, c.rlen)
            else:
                keep_lo, keep_hi = max(lo, lo_ok), min(hi, hi_ok)
                if keep_lo >= keep_hi:
                    continue
                scratch = memoryview(bytearray(c.rlen))
                self._decode(c, src, scratch, 0, c.rlen)
                dst[keep_lo - base:keep_hi - base] = \
                    scratch[keep_lo - lo:keep_hi - lo]
            n += c.rlen
        return n

    def fetch_chunk(self, c) -> bytes:
        """One chunk's payload, for the ref and delta chunks that resolve an
        earlier output range by decoding whatever chunks cover it."""
        return self.source.pread(c.off, c.clen)

    # -- the coupling, in one place ---------------------------------------
    def _decoder(self):
        """A callable that decodes one chunk into a buffer at an offset.

        Preferred form goes through lmz's own per-chunk decoder, which is what
        `decompress` uses and which hands the whole chunk to native code in
        one call. If that has moved, `MappedArchive.read` is public, stable
        and does the same job with a copy and a cache in the way.
        """
        try:
            from ._lmz import require

            api, freqs = require("api", "freqs")
            _chunk_decoder, TableSet = api._chunk_decoder, freqs.TableSet
        except Exception:                 # noqa: BLE001 -- fall back, do not fail
            return self._decode_via_mapped, "mapped"

        try:
            tables = TableSet.from_json(self._reader.manifest.get("tables"))
            # `fetch_chunk`, not None: a ref or delta chunk names an earlier
            # output range and lmz materialises it by decoding whatever chunks
            # cover it, which means reading payloads this run never asked for.
            # A resolver with no way to read is a decoder that works on every
            # archive until it meets a deduplicated directory or a delta
            # checkpoint -- both of which lmz writes by default.
            inner = _chunk_decoder(self._reader.chunks, self.fetch_chunk,
                                   True, tables)
        except Exception:                 # noqa: BLE001
            return self._decode_via_mapped, "mapped"

        def decode(chunk, payload, out, lo, rlen):
            # `out=` asks lmz to decode straight into the staging buffer, so
            # the bytes are written once. Not every codec can honour it -- a
            # stored chunk returns the payload itself, a whole-chunk zstd one
            # returns what the decompressor allocated, and a ref returns a
            # freshly resolved range -- so the returned view is the authority
            # and a copy happens only when it points somewhere else.
            dst = out[lo:lo + rlen]
            got = inner(chunk, payload, out=dst)
            if got is not None and not _same(dst, got):
                dst[:] = got
            return rlen

        return decode, "direct"

    def _decode_via_mapped(self, chunk, payload, out, lo, rlen):
        if self._mapped is None:
            from ._lmz import lmz

            self._mapped = lmz().MappedArchive(self.path, cache_blocks=2)
        out[lo:lo + rlen] = self._mapped.read(chunk.dst, rlen)
        return rlen

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _same(dst, got) -> bool:
    """Whether lmz decoded in place into the buffer we gave it.

    It does when `out` was honoured, and then copying would be moving the
    chunk onto itself -- which at a gigabyte of weights is a second pass over
    all of it. Compared by address rather than by value, because comparing the
    contents would cost exactly what the copy costs.
    """
    if dst is got:
        return True
    try:
        import ctypes

        return (len(dst) == len(got)
                and ctypes.addressof(ctypes.c_char.from_buffer(dst))
                == ctypes.addressof(ctypes.c_char.from_buffer(got)))
    except (TypeError, ValueError, BufferError):
        return False
