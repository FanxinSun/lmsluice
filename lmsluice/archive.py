"""A coded container as work this package can move.

The transport half of the split. A `Codec` (see `codec.py`) says what the units
of coded work are, where their plaintext lands, and how to turn one into the
other; everything on this side is the decision the codec deliberately does not
take -- which units to read together, in what order, on how many threads, into
whose buffer.

That division is why this file no longer imports lmz. It composes a codec
adapter and never looks inside a coded stream, which is invariant I4 and is
checked by a test that walks the package source rather than the import graph,
so a new file that reaches for lmz is caught the day it is written.

The unit of I/O is a **run**: consecutive units whose payloads sit together in
the container, read in one call and decoded one at a time. Runs rather than
units because a page-mapped archive has tens of thousands of 64 KiB blocks and
asking the kernel for each separately spends more on system calls than on
bytes; runs rather than the whole file because memory is bounded and because a
caller may want four tensors out of four hundred. Coalescing is an I/O decision
and so it is here, not in the codec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .codec import Tensor, Unit  # noqa: F401 -- re-exported for callers

# How far apart two payloads may sit and still be read together. An aligned
# archive pads between payloads and reading the padding is cheaper than issuing
# a second request for what is on the other side of it -- but only up to a
# point, and the point is a fraction of the useful bytes rather than a fixed
# size, so it scales with the run instead of with any one device.
COALESCE_SLACK = 1.25
COALESCE_MAX = 32 << 20


class Unsupported(RuntimeError):
    """Raised when no adapter can serve this container."""


@dataclass
class Run:
    """One read, and the units it satisfies."""

    off: int                       # first payload byte in the container
    span: int                      # bytes to read, padding included
    chunks: list = field(default_factory=list)
    plain: int = 0                 # plain bytes the run produces

    def __len__(self) -> int:
        return self.span


class Archive:
    """A container opened for transport rather than for extraction."""

    def __init__(self, target, *, source=None, codec=None):
        from .source import Source, open_source

        if codec is None:
            if source is None:
                source = (target if isinstance(target, Source)
                          else open_source(target))
            # By magic, not by extension: a file's name is a hint and its first
            # eight bytes are a fact.
            from . import sealed, zstdcodec

            # An encrypted envelope is unwrapped here, before any codec is
            # asked what it is looking at. That ordering is the whole of the
            # read side: below this line every codec sees the plaintext
            # container it would have seen unencrypted, and none of them
            # contains the word. See `sealed.py` on why encryption is
            # transport and not format.
            if sealed.is_sealed(source):
                source = sealed.SealedSource(source)

            if zstdcodec.is_archive(source):
                codec = zstdcodec.open_codec(target, source)
            else:
                from .lmzcodec import open_codec

                codec = open_codec(target, source)
        self.codec = codec
        self.source = codec.source
        self.path = self.source.name
        self.plain_bytes = codec.plain_bytes
        self.coded_bytes = codec.coded_bytes
        self.route = getattr(codec, "route", "codec")

    # -- what it holds ----------------------------------------------------
    @property
    def ratio(self) -> float:
        """Coded bytes per plain byte. Below 1 is a saving."""
        return self.codec.ratio()

    @property
    def block_bytes(self) -> int:
        """The largest block: what a one-byte read costs to satisfy."""
        return self.codec.block_bytes

    @property
    def members(self):
        return self.codec.members()

    @property
    def manifest(self) -> dict:
        return getattr(self.codec, "manifest", {})

    def tensors(self) -> dict:
        return self.codec.tensor_index()

    def span_of(self, member: str | None = None) -> tuple[int, int]:
        return self.codec.span_of(member)

    def capabilities(self):
        return self.codec.capabilities()

    def cost_model(self):
        return self.codec.cost_model()

    # -- planning work ----------------------------------------------------
    def chunks_for(self, spans) -> list:
        return self.codec.units_for(spans)

    def cover(self, spans, target: int):
        """(lo, hi, runs) — the plain range the covering blocks fill.

        The range comes from the codec, which knows where a block stops; the
        runs come from here, because grouping them into reads is an I/O
        decision and the codec has no business making one.
        """
        units = self.codec.units_for(spans)
        if not units:
            return 0, 0, []
        lo, hi = self.codec.cover(spans)
        return lo, hi, self.runs(units, target)

    def runs(self, chunks, target: int) -> list[Run]:
        """Group units into reads of roughly `target` coded bytes each.

        A gap is absorbed when it is small next to the run it would split, and
        starts a new run when it is not. The comparison is against the useful
        bytes rather than against a size, so the same rule holds for a
        64 KiB-block archive and an 8 MiB one without either being named here.
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

    def fetch_chunk(self, c) -> bytes:
        """One unit's payload."""
        return self.source.pread(c.off, c.clen)

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
                self.codec.decode_chunk(c, src, dst, lo - base)
            else:
                keep_lo, keep_hi = max(lo, lo_ok), min(hi, hi_ok)
                if keep_lo >= keep_hi:
                    continue
                scratch = memoryview(bytearray(c.rlen))
                self.codec.decode_chunk(c, src, scratch, 0)
                dst[keep_lo - base:keep_hi - base] = \
                    scratch[keep_lo - lo:keep_hi - lo]
            n += c.rlen
        return n

    def place_one(self, c, out) -> int:
        """Decode one unit into a buffer that starts at its own `dst`."""
        return self.codec.decode_chunk(c, memoryview(self.fetch_chunk(c)),
                                       memoryview(out), 0)

    def close(self) -> None:
        self.codec.close()
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
