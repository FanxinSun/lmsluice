"""A codec that needs nothing installed.

Python 3.14 ships zstd in the standard library, which means this package can
compress and decompress without anyone installing anything at all. That is not
a convenience: it is what makes compress-on-first-fetch possible. A cache that
requires a dependency is a dependency; a cache that requires only the
interpreter already present is just a cache.

Measured on real BF16 weights on the box in `MEASURED.md`: **21.1% saved,
0.59-0.71 GB/s decode single-threaded**, against lmz's 32.9% and ~1.05. So this
captures a 1.27x ceiling where lmz reaches 1.49x -- about 85% of the available
win, for no dependency -- and it clears the gate comfortably everywhere the gate
matters, which is every link under about half a gigabyte a second.

lmz remains the better codec and is preferred wherever it is present. This is
the floor, not the ceiling.

## The container

Deliberately small, because owning a format is a cost and this one should carry
no more than it must:

    magic "LMSLUICE" + u32 version
    chunk payloads, in destination order
    index: JSON, zstd
    footer: u64 index offset, u64 index length, magic "LMSLTAIL"

Chunks are fixed-size spans of the *plaintext*, each compressed on its own, so
any byte range can be decoded by decoding the blocks that cover it and nothing
else -- which is what lets the transport read a subset, run stages in parallel,
and stream under a memory ceiling. The index carries a CRC per chunk, because a
silently wrong weight is worse than a slow one.

The index also carries the **measured** encode and decode rates of the data it
holds. An archive written on this machine therefore knows what it cost and what
it will cost to read back, so the routing decision for the second load needs no
separate probe -- which matters because the first load is exactly when nobody
has measured anything yet.
"""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from dataclasses import dataclass

from .codec import Capabilities, CodecCost, Cost, Option, Tensor, Unit, missing

MAGIC = b"LMSLUICE"
TAIL = b"LMSLTAIL"
VERSION = 1
FOOTER = struct.Struct("<QQ8s")

# Plaintext per chunk. Large enough that per-chunk overhead is noise and the
# compressor sees enough context to be worth using; small enough that a partial
# read does not decode the whole file. The same figure the transport uses for a
# pipeline job, for the same reasons.
DEFAULT_CHUNK = 4 << 20

# zstd level. 1 rather than 3 because on weight data the measured difference was
# 0.1 points of ratio (21.1% against 21.2%) for 23% more encode time, and this
# codec's job is to be cheap enough to run on the way past.
DEFAULT_LEVEL = 1


# The first four bytes of every zstd frame. Deflate cannot collide with it:
# a zlib header is two bytes whose big-endian value must be divisible by 31,
# and 0x28B5 leaves a remainder of 5. So the compressor a blob was written with
# is readable from the blob, which is what lets an archive written on 3.14 be
# read on 3.11 without the format carrying a flag it did not always carry.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class _Deflate:
    """`zlib` wearing the zstd module's two methods.

    The fallback, and not a hypothetical one: `compression.zstd` arrived in
    Python 3.14, and this package supports 3.10. Without this, `pip install
    lmsluice` on the Python most people are running produced a package whose
    own codec could not run -- which CI caught on the first push and no local
    test could, because the development box is on 3.14.

    Deflate compresses BF16 weights worse than zstd. That is a ratio, and the
    gate prices ratios; it is not a failure, and `lmsluice[zstd]` closes it for
    anyone who would rather install one small wheel. lmz makes the same bargain
    in `lmz/entropy.py`.
    """

    name = "deflate"

    @staticmethod
    def compress(data, level=1):
        return zlib.compress(bytes(data), min(max(int(level), 1), 9))

    @staticmethod
    def decompress(data):
        return zlib.decompress(bytes(data))


def _zstd(*, required: bool = True):
    """The zstd module if this interpreter has one, else `None` or an error."""
    try:
        from compression import zstd

        return zstd
    except ImportError:
        pass
    try:                                  # 3.14 moved it; older builds may vary
        import zstandard as zstd

        return zstd
    except ImportError:
        if not required:
            return None
        raise ImportError(
            "no zstd on this interpreter: `compression.zstd` is Python 3.14+ "
            "and the `zstandard` package is not installed. Install "
            "`lmsluice[zstd]` to read this archive") from None


def _compressor():
    """What to write with: zstd where it exists, deflate everywhere else."""
    return _zstd(required=False) or _Deflate


def _decompress(blob):
    """Decompress by what the bytes say they are, not by what this box has.

    Sniffing rather than trusting a header field, because archives written
    before the field existed must still open -- and because the bytes are the
    fact while a field is a claim about them.
    """
    data = bytes(blob)
    if data[:4] == ZSTD_MAGIC:
        return _zstd().decompress(data)
    return zlib.decompress(data)


def backend_name() -> str:
    """`zstd` or `deflate`: what this interpreter would write with."""
    return "deflate" if _compressor() is _Deflate else "zstd"


@dataclass(frozen=True)
class Member:
    path: str
    size: int
    dst: int


class ZstdEncoder:
    """The write side. Options are declared, as `codec.Encoder` requires."""

    OPTIONS = (
        Option("level", DEFAULT_LEVEL, "format", "zstd compression level"),
        Option("chunk_size", DEFAULT_CHUNK, "format",
               "plaintext bytes per independently-decodable block"),
    )

    def encode_options(self) -> dict:
        return {o.name: o for o in self.OPTIONS}

    def encode(self, src: str, dst: str, **opts):
        declared = self.encode_options()
        unknown = sorted(set(opts) - set(declared))
        if unknown:
            raise ValueError(f"{unknown} is not declared by this codec; "
                             f"encode_options() lists {sorted(declared)}")
        zstd = _compressor()
        level = opts.get("level", DEFAULT_LEVEL)
        chunk = opts.get("chunk_size", DEFAULT_CHUNK)
        name = os.path.basename(src)
        total = os.path.getsize(src)

        chunks = []
        started = time.perf_counter()
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            fo.write(MAGIC + struct.pack("<I", VERSION))
            off = fo.tell()
            dstpos = 0
            while True:
                raw = fi.read(chunk)
                if not raw:
                    break
                payload = zstd.compress(raw, level)
                fo.write(payload)
                chunks.append([off, len(payload), dstpos, len(raw),
                               zlib.crc32(raw) & 0xFFFFFFFF])
                off += len(payload)
                dstpos += len(raw)
            encode_seconds = time.perf_counter() - started

            # Decode one representative chunk to record what reading it back
            # will cost. Nearly free, and it means the archive knows its own
            # read rate before anybody has probed this machine.
            decode_rate = None
            if chunks:
                mid = chunks[len(chunks) // 2]
                fo.flush()
                with open(dst, "rb") as probe:
                    probe.seek(mid[0])
                    blob = probe.read(mid[1])
                t = time.perf_counter()
                got = _decompress(blob)
                dt = time.perf_counter() - t
                if dt > 0:
                    decode_rate = len(got) / dt

            index = {
                "version": VERSION, "member": name, "total": total,
                "chunk_size": chunk, "level": level, "chunks": chunks,
                # What wrote it, so `info` can say so and a reader on another
                # interpreter knows before it starts. The bytes are still the
                # authority -- `_decompress` sniffs and does not consult this.
                "compressor": backend_name(),
                "tensors": _safetensors_tensors(src),
                "measured": {
                    "encode_rate": total / encode_seconds if encode_seconds else None,
                    "decode_rate_1t": decode_rate,
                    "machine": _fingerprint(),
                },
            }
            blob = zstd.compress(json.dumps(index).encode(), 3)
            index_off = fo.tell()
            fo.write(blob)
            fo.write(FOOTER.pack(index_off, len(blob), TAIL))
        return index["measured"]


def _fingerprint() -> str:
    from .rates import fingerprint

    return fingerprint()


def _safetensors_tensors(path: str):
    """The tensor index of a safetensors file, or None if it is not one.

    Carried into the archive so `tensor_index()` works without the plain file,
    which is what lets a cached copy serve a partial read.
    """
    try:
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n <= 0 or n > (64 << 20):
                return None
            head = json.loads(fh.read(n))
    except Exception:                     # noqa: BLE001 -- not safetensors
        return None
    base = 8 + n
    out = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        out[name] = {"dtype": meta.get("dtype", ""),
                     "shape": meta.get("shape", []),
                     "offsets": [base + lo, base + hi]}
    return out


class ZstdCodec:
    """The read side: `codec.Codec` over the container above."""

    route = "zstd"

    def __init__(self, source):
        self.source = source
        self._path = source.name
        self.coded_bytes = source.size
        head = source.pread(0, len(MAGIC) + 4)
        if head[:len(MAGIC)] != MAGIC:
            raise ValueError(f"{self._path}: not an lmsluice archive")
        foot = source.pread(source.size - FOOTER.size, FOOTER.size)
        index_off, index_len, tail = FOOTER.unpack(foot)
        if tail != TAIL:
            raise ValueError(f"{self._path}: truncated or corrupt (bad footer)")
        self._index = json.loads(
            _decompress(source.pread(index_off, index_len)))
        self.plain_bytes = self._index["total"]
        self._units = [Unit(off, clen, dst, rlen, crc)
                       for off, clen, dst, rlen, crc in self._index["chunks"]]
        self._starts = [u.dst for u in self._units]

    # -- what it holds ----------------------------------------------------
    def ratio(self) -> float:
        return self.coded_bytes / self.plain_bytes if self.plain_bytes else 1.0

    @property
    def block_bytes(self) -> int:
        return max((u.rlen for u in self._units), default=0)

    @property
    def manifest(self) -> dict:
        return self._index

    def members(self):
        return [Member(self._index["member"], self.plain_bytes, 0)]

    def units(self):
        return self._units

    def tensor_index(self) -> dict:
        out = {}
        member = self._index["member"]
        for name, meta in (self._index.get("tensors") or {}).items():
            lo, hi = meta["offsets"]
            out[name] = Tensor(name, meta.get("dtype", ""),
                               tuple(meta.get("shape", ())), member, lo, hi)
        return out

    def span_of(self, member: str | None = None) -> tuple[int, int]:
        if member is None or member == self._index["member"]:
            return 0, self.plain_bytes
        raise KeyError(f"no member named {member!r}")

    def units_for(self, spans) -> list:
        from bisect import bisect_right

        wanted = {}
        for lo, hi in spans:
            lo, hi = max(0, lo), min(self.plain_bytes, hi)
            if lo >= hi:
                continue
            i = max(0, bisect_right(self._starts, lo) - 1)
            while i < len(self._units) and self._units[i].dst < hi:
                u = self._units[i]
                if u.dst + u.rlen > lo:
                    wanted[u.off] = u
                i += 1
        return [wanted[k] for k in sorted(wanted)]

    def cover(self, spans) -> tuple:
        units = self.units_for(spans)
        if not units:
            return 0, 0
        return (min(u.dst for u in units),
                max(u.dst + u.rlen for u in units))

    # -- decoding ---------------------------------------------------------
    def decode_chunk(self, unit: Unit, payload, out, off: int) -> int:
        """Pure: bytes in, bytes out. No I/O, no threads."""
        raw = _decompress(payload)
        if len(raw) != unit.rlen:
            raise ValueError(f"chunk decoded to {len(raw)}, expected {unit.rlen}")
        if unit.opaque is not None and \
                (zlib.crc32(raw) & 0xFFFFFFFF) != unit.opaque:
            raise ValueError("chunk failed its checksum")
        out[off:off + unit.rlen] = raw
        return unit.rlen

    # -- declarations -----------------------------------------------------
    def capabilities(self) -> Capabilities:
        """No device decoder exists for this codec, and saying so is the point.

        `derived=False` because this is a declaration rather than an inference:
        there is no zstd kernel to be uncertain about.
        """
        return Capabilities(
            host=True, device=False, device_share=0.0,
            why_not="this codec has no device decoder",
            advice="use lmz for a device-decodable archive", derived=False)

    def cost_model(self) -> CodecCost:
        """What this codec cost, measured when the archive was written.

        Unusual among cost models in that it is per-archive and per-machine
        rather than published by an implementation -- because zstd's rate
        depends on the data and this container measured it on the way past.
        `gate.py` will not use `k` (wrong unit, and there is no kernel), which
        is correct: this codec never runs on a device.
        """
        m = self._index.get("measured") or {}
        rate = m.get("decode_rate_1t")
        return CodecCost(
            k=missing("k"),
            lanes=Cost("lanes", low=1, high=1, unit="threads per stream",
                       provenance="zstd frames decode on one thread each; "
                                  "parallelism comes from chunk count"),
            bytes_per_symbol=missing("bytes_per_symbol"),
            source=(f"measured on this archive at write time: "
                    f"{rate / 1e9:.2f} GB/s single-threaded on "
                    f"{m.get('machine', 'an unrecorded machine')}"
                    if rate else "not measured"))

    @property
    def measured_decode_rate(self) -> float | None:
        """Single-threaded decode, measured when this archive was written."""
        return (self._index.get("measured") or {}).get("decode_rate_1t")

    def close(self) -> None:
        pass


def encoder() -> ZstdEncoder:
    return ZstdEncoder()


def is_archive(source) -> bool:
    try:
        return source.pread(0, len(MAGIC)) == MAGIC
    except Exception:                     # noqa: BLE001
        return False


def open_codec(target, source=None):
    from .source import Source, open_source

    if source is None:
        source = target if isinstance(target, Source) else open_source(target)
    return ZstdCodec(source)
