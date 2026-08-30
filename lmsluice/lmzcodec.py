"""lmz behind the codec interface — the only module that knows the format.

Every use of lmz in this package is here. That is the point: `docs/boundary.md`
says no lmsluice module outside an adapter may parse coded-stream structure, and
a rule that lives in one file can be enforced by a test that walks the source.

**The private-symbol loan is retired.** lmz shipped `ArchiveIndex` in `b426cc9`:
`chunks()` gives the addressing and `decode(ref, payload, out=)` turns one
chunk's coded bytes into plaintext with, in its own words, "No I/O, no
threads." That is invariant I1 met exactly, and it is what this adapter now
uses. The reach for `lmz.api._chunk_decoder` and `lmz.freqs.TableSet` remains
only as a fallback, because that API is on `shared-frequency-tables` and a
released 1.2.0 has none of it.

Three paths, probed in order, and `route` names the live one so a benchmark can
never quietly measure a fallback:

- `public` -- `ArchiveIndex`, the I1-satisfying entry point;
- `direct` -- the old private reach, for an lmz that predates it;
- `mapped` -- `MappedArchive.read`, public and stable everywhere, doing the
  same job with a copy and a cache in the way.

What is still on loan is `merge.cu` and its dispatch, retired by I2 -- a device
decode that returns plaintext rather than plane-major bytes. See the register in
`docs/boundary.md`.
"""

from __future__ import annotations

import os

from .codec import (Capabilities, CodecCost, Cost, Option, Tensor, Unit,
                    missing)

# lmz codec ids this adapter reasons about. Held here and nowhere else in the
# package -- a codec id is coded-stream structure, and I4 keeps it out of the
# transport modules.
CODEC_ZSTD = 1
CODEC_SPLIT = 2
CODEC_BF16 = 3
CODEC_BF16C = 5
CODEC_SPLIT_ST = 9

# Per-plane coding method, two bits per plane in a split chunk's flags.
METHOD_STORED = 0
METHOD_RANS = 3

# lmz only attempts its conditioned BF16 codec at or above this many elements
# per chunk. Read from lmz at run time where possible; the literal is the
# fallback and is a loan -- see `gpu_chunk_size` in `devdecode.py`.
COND_MIN_ELEMS = 1 << 20


class Unsupported(RuntimeError):
    """Raised when this build of lmz cannot serve the pipeline at all."""


class LmzCodec:
    """The lmz implementation of `codec.Codec`."""

    def __init__(self, source):
        from ._lmz import require

        (fmt,) = require("format")
        self.source = source
        self._reader = fmt.ArchiveReader(source.reader())
        self.coded_bytes = source.size
        self.members_ = self._reader.members
        self.plain_bytes = sum(m.size for m in self.members_)
        self._mapped = None
        self._path = source.name
        self._decode, self.route = self._decoder()
        self._chunks = sorted(self._reader.chunks, key=lambda c: c.dst)
        self._starts = [c.dst for c in self._chunks]
        self._units = [Unit(c.off, c.clen, c.dst, c.rlen, c) for c in self._chunks]

    # -- what it holds ----------------------------------------------------
    def ratio(self) -> float:
        return self.coded_bytes / self.plain_bytes if self.plain_bytes else 1.0

    @property
    def block_bytes(self) -> int:
        return max((u.rlen for u in self._units), default=0)

    @property
    def manifest(self) -> dict:
        return self._reader.manifest

    def members(self):
        return self.members_

    def units(self):
        return self._units

    def tensor_index(self) -> dict:
        out = {}
        for m in self.members_:
            for name, meta in (m.tensors or {}).items():
                start, end = meta["offsets"]
                out[name] = Tensor(name, meta.get("dtype", ""),
                                   tuple(meta.get("shape", ())), m.path,
                                   m.dst + start, m.dst + end)
        return out

    def span_of(self, member: str | None = None) -> tuple[int, int]:
        if member is None:
            return 0, self.plain_bytes
        for m in self.members_:
            if m.path == member:
                return m.dst, m.dst + m.size
        raise KeyError(f"no member named {member!r}")

    # -- addressing -------------------------------------------------------
    def units_for(self, spans) -> list:
        """Every unit touching any of `spans`, once, in container order.

        Container order rather than destination order because it is what the
        fetch stage will read, and a device is fastest asked for its bytes in
        the order it holds them. Destination order needs no help: each unit
        already knows where it lands.
        """
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
        return self._decode(unit.opaque, payload, out, off, unit.rlen)

    # -- declarations -----------------------------------------------------
    def capabilities(self) -> Capabilities:
        """Which decoders can read this archive.

        **Declared where lmz declares it.** `lmz.capabilities()` shipped in
        b426cc9 and reports `batch_decodable`, the fraction of bytes a device
        decoder can take, and the blockers by name -- with its own `derived`
        flag set False, because it reads the archive rather than reproducing
        an encoder threshold. That is I3 met, so this asks rather than infers,
        and `Capabilities.derived` carries lmz's answer instead of a standing
        confession.

        The inference below remains for an lmz that predates it. It agrees
        with the declaration on both fixtures here, which is reassuring and is
        not a reason to keep trusting it: an inference that reproduces an
        encoder's threshold goes stale the day the encoder changes it, which
        is the whole argument for I3.
        """
        try:
            from ._lmz import lmz

            got = lmz().capabilities(self._path)
            share = float(got.get("batch_decodable_bytes", 0.0))
            blockers = got.get("blockers") or {}
            why = ("; ".join(f"{name} ({count} chunks)"
                             for name, count in blockers.items())
                   if blockers else "")
            advice = ""
            if "bf16-cond" in blockers:
                from .devdecode import gpu_chunk_size

                advice = (f"recompress with chunk_size="
                          f"{gpu_chunk_size(2) >> 20} MiB or less, which makes "
                          f"lmz emit the plain field split instead; measured "
                          f"cost 0.2 points of ratio on a 1.87 GB BF16 "
                          f"checkpoint")
            return Capabilities(host=True, device=share > 0.0,
                                device_share=share, why_not=why, advice=advice,
                                derived=bool(got.get("derived", False)))
        except Exception:                 # noqa: BLE001 -- older lmz: infer
            pass
        from .devdecode import DEVICE_CODECS, gpu_chunk_size

        total = sum(u.rlen for u in self._units) or 1
        ok = sum(u.rlen for u in self._units
                 if u.opaque.codec in DEVICE_CODECS)
        cond = [u for u in self._units if u.opaque.codec == CODEC_BF16C]
        why = advice = ""
        if cond:
            esize = cond[0].opaque.esize or 2
            why = ("lmz's conditioned BF16 codec partitions sign and mantissa "
                   "into per-bucket streams of unequal length, and the batch "
                   "decoder needs every stream in a batch to decode to the "
                   "same size")
            advice = (f"recompress with chunk_size="
                      f"{gpu_chunk_size(esize) >> 20} MiB or less, which makes "
                      f"lmz emit the plain field split instead; measured cost "
                      f"0.2 points of ratio on a 1.87 GB BF16 checkpoint")
        return Capabilities(host=True, device=ok > 0, device_share=ok / total,
                            why_not=why, advice=advice, derived=True)

    def cost_model(self) -> CodecCost:
        """lmz's decoder cost constants, read from the codec where it publishes
        them and derived only where it does not.

        lmz publishes `lmz.gpu.cost_model()`, and `k` is **measured** rather
        than inferred. As of 1.3.0 it is also no longer a single-point fit:
        `k_is_single_point_fit` is False for the shared-table kernel, because
        the value comes from a grid sweep at fixed byte traffic rather than
        from one launch. That distinction is carried through here because it is
        what lets a consumer pick between a compute-bound and a bandwidth-bound
        reading instead of reporting both.

        **No value is restated here.** Everything below is read out of the
        published dict, including the kernel's shared-memory shape, which lmz
        used to describe only in prose and now publishes as fields. A constant
        copied across this boundary is correct until it is not, and goes stale
        without saying so -- which is the whole reason this exchange exists.

        Note the unit. lmz publishes **lane-cycles**, and says in the same
        breath that the inner loop is a dependent shared-memory chain which
        "does not scale with a device's FP32 rate and must not be estimated
        from it". `gate.py` was built on FP32-FLOP equivalents, so the two are
        not interchangeable and the unit is carried through here rather than
        converted -- a silent conversion would be a units error dressed as an
        improvement.
        """
        try:
            from ._lmz import require

            (gpu,) = require("gpu")
            published = gpu.cost_model()
        except Exception:                 # noqa: BLE001 -- older lmz, or none
            published = None

        if not published:
            return CodecCost(
                k=Cost("k", unit="unknown",
                       provenance="this codec publishes no cost model"),
                lanes=Cost("lanes", low=8, high=8,
                           unit="rANS states per stream",
                           provenance="lmz format: 8 interleaved states"),
                bytes_per_symbol=missing("bytes_per_symbol"),
                source="derived; lmz publishes no cost model")

        prov = published.get("provenance", {})
        where = "; ".join(str(prov.get(key, "")) for key in
                          ("kernel", "device", "method") if prov.get(key))
        bound = published.get("bound") or {}

        def scalar(name, value, unit, note="published by the codec"):
            return Cost(name, low=value, high=value, unit=unit,
                        provenance=note if value is not None
                        else "not published by this codec")

        def interval(name, value, unit):
            lo, hi = (value if isinstance(value, (list, tuple)) and len(value) == 2
                      else (value, value))
            return Cost(name, low=lo, high=hi, unit=unit,
                        provenance="published by the codec" if lo is not None
                        else "not published by this codec")

        lo, hi = published.get("k_cycles_per_byte", (None, None))
        single = published.get("k_is_single_point_fit")
        return CodecCost(
            k=Cost("k", low=lo, high=hi, unit="lane-cycles per decoded byte",
                   provenance=f"measured and published by the codec: {where}"),
            lanes=scalar("lanes", published.get("lanes"),
                         "rANS states per stream"),
            bytes_per_symbol=scalar("bytes_per_symbol",
                                    published.get("bytes_per_symbol"),
                                    "coded bytes per symbol"),
            expansion=scalar("expansion", published.get("expansion"),
                             "plain bytes per coded byte"),
            achieved_fraction=scalar(
                "achieved_fraction",
                bound.get("saturates_at_fraction_of_peak_dram"),
                "fraction of peak DRAM the kernel sustains"),
            shmem_lut_bytes=scalar("shmem_lut_bytes",
                                   published.get("shmem_lut_bytes"),
                                   "bytes per block"),
            shmem_per_group_bytes=scalar(
                "shmem_per_group_bytes",
                published.get("shmem_per_group_bytes"),
                "bytes per group of `lanes` threads"),
            block_threads=scalar("block_threads",
                                 bound.get("compute_below_threads"),
                                 "threads per block"),
            blocks_per_unit_at_measurement=interval(
                "blocks_per_unit_at_measurement",
                published.get("blocks_per_unit_at_measurement"),
                "blocks resident per unit"),
            k_single_point=single,
            source="lmz.gpu.cost_model(), published")

    # -- the coupling, in one place ---------------------------------------
    def _decoder(self):
        # Public first. `ArchiveIndex.decode` is pure by contract, which is the
        # whole of I1, so preferring it is not a style choice -- it is the
        # boundary being satisfied rather than worked around.
        try:
            from ._lmz import lmz

            m = lmz()
            index = m.ArchiveIndex(self._path)
            needs_source = m.ArchiveIndex.NeedsSource
            refs = {r.off: r for r in index.chunks()}

            def decode_public(chunk, payload, out, lo, rlen):
                ref = refs.get(chunk.off)
                if ref is None:
                    return self._decode_via_mapped(chunk, payload, out, lo, rlen)
                dst = out[lo:lo + rlen]
                try:
                    got = index.decode(ref, payload, out=dst)
                except needs_source:
                    # A ref or delta chunk is defined against an earlier output
                    # range, which cannot be decoded from its own payload. lmz
                    # says to use `decompress`; the mapped reader is the same
                    # answer without giving up the rest of the pipeline.
                    return self._decode_via_mapped(chunk, payload, out, lo, rlen)
                if got is not None and not _same(dst, got):
                    dst[:] = got
                return rlen

            self._index = index
            return decode_public, "public"
        except Exception:                 # noqa: BLE001 -- older lmz, fall through
            pass

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
            inner = _chunk_decoder(self._reader.chunks, self.fetch_chunk,
                                   True, tables)
        except Exception:                 # noqa: BLE001
            return self._decode_via_mapped, "mapped"

        def decode(chunk, payload, out, lo, rlen):
            # `out=` asks lmz to decode straight into the staging buffer, so
            # the bytes are written once. Not every codec can honour it, so
            # the returned view is the authority and a copy happens only when
            # it points somewhere else.
            dst = out[lo:lo + rlen]
            got = inner(chunk, payload, out=dst)
            if got is not None and not _same(dst, got):
                dst[:] = got
            return rlen

        return decode, "direct"

    def _decode_via_mapped(self, chunk, payload, out, lo, rlen):
        if self._mapped is None:
            from ._lmz import lmz

            self._mapped = lmz().MappedArchive(self._path, cache_blocks=2)
        out[lo:lo + rlen] = self._mapped.read(chunk.dst, rlen)
        return rlen

    def fetch_chunk(self, c) -> bytes:
        """One chunk's payload, for the ref and delta chunks that resolve an
        earlier output range by decoding whatever chunks cover it.

        The one place the adapter does I/O, and it is lmz's resolver reaching
        back through it rather than a scheduling decision of ours.
        """
        return self.source.pread(c.off, c.clen)

    def close(self) -> None:
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None


def _same(dst, got) -> bool:
    """Whether lmz decoded in place into the buffer we gave it.

    Compared by address rather than by value, because comparing the contents
    would cost exactly what the copy costs.
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


def backend_info() -> dict:
    """What codec backend is active, for diagnostics.

    Here rather than in `rates.py` because asking lmz its version is still
    asking lmz, and `docs/boundary.md` puts every such question behind the
    adapter so one test can enforce it.
    """
    try:
        from ._lmz import lmz

        m = lmz()
        return {"name": "lmz", "version": m.__version__,
                "backends": m.backends()}
    except Exception as exc:              # noqa: BLE001 -- a codec is optional
        return {"name": "lmz", "version": None, "error": str(exc)}


class LmzEncoder:
    """lmz's write side, with its options declared rather than assumed.

    Eleven options, three kinds. Eight are **format**: they change what the
    bytes become. One is **schedule** -- `workers` is a thread count, which is
    machine-shaped and by the ownership table is lmsluice's to choose even
    though the codec is what accepts it. One is **observe** -- `progress` is a
    callback that touches neither the bytes nor the schedule, and without a
    third kind it would be homeless, forcing `encode()` to carry an unwritten
    exception list for callbacks. An unwritten exception list is the thing the
    declaration exists to abolish.

    On the schedule kind: the rule is that **interrogation is not
    measurement**. A static capability query is fine -- `compress()` already
    defaults `workers` from `os.process_cpu_count()`, which is asking the
    machine what it is, not timing it. What a codec may not do is time, fit,
    adapt or cache a measured profile, because that is a policy measurement
    and policy measurement is this package's. A schedule option's default is a
    fallback, never a decision.
    """

    OPTIONS = (
        Option("level", 1, "format",
               "entropy coding effort; the codec's own scale"),
        Option("chunk_size", 8 << 20, "format",
               "bytes of plaintext per block; decides random-access "
               "granularity and, on this codec, which sub-codecs are reachable"),
        Option("mapped", False, "format",
               "small blocks so any byte range can be read without expanding "
               "the archive"),
        Option("align", False, "format",
               "start payloads on page boundaries"),
        Option("checksum", True, "format", "per-chunk checksums"),
        Option("dedup", True, "format",
               "store a tensor once when a directory ships it twice"),
        Option("delta", True, "format",
               "code a checkpoint as the difference from an earlier one"),
        Option("shared_tables", False, "format",
               "one frequency table per plane kind instead of one per stream"),
        Option("workers", None, "schedule",
               "threads to encode with -- a property of the machine, not of "
               "the format, and therefore the caller's to choose"),
        Option("progress", None, "observe",
               "a callback invoked as the encode proceeds; touches neither "
               "the bytes nor the schedule"),
    )

    def encode_options(self) -> dict:
        return {o.name: o for o in self.OPTIONS}

    def encode(self, src: str, dst: str, **opts):
        declared = self.encode_options()
        unknown = sorted(set(opts) - set(declared))
        if unknown:
            raise ValueError(
                f"{unknown} is not declared by this codec; "
                f"encode_options() lists {sorted(declared)}")
        from ._lmz import lmz

        return lmz().compress(src, dst, **opts)


def encoder() -> LmzEncoder:
    """The write side of this codec."""
    return LmzEncoder()


def compress(src: str, dst: str, **kw):
    """Deprecated passthrough; use `encoder().encode(...)`."""
    return LmzEncoder().encode(src, dst, **kw)


def open_codec(target, source=None):
    """The lmz adapter for a target, opening a source if one is not given."""
    from .source import Source, open_source

    if source is None:
        source = target if isinstance(target, Source) else open_source(target)
    return LmzCodec(source)
