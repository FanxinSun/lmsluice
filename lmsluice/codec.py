"""What lmsluice needs from a codec — written as an interface, not as lmz.

The boundary between the two projects is stated in `docs/boundary.md`: lmz
decides what the bytes *are*, lmsluice decides *when, from where, over which
link, and into whose memory*. A charter in a document is a convention, and
conventions drift — four of them did. This module is the same statement in a
form a test can check, because anything lmsluice does that is not one of the
operations below is a **loan** and has to be registered as one.

Writing it against *a* codec rather than against lmz is not tidiness. The
founding arithmetic of this package, `speedup = min(1/f, decode/link)`,
mentions no codec at all; the layer has to be able to route a zstd or GGUF
source on a machine where lmz is not installed, and it is cheap to keep that
possible now, while one adapter is the only thing that touches lmz.

Two properties of this interface carry most of its weight:

**Nothing here does I/O.** `decode_chunk` takes bytes and returns bytes.
Reading them is a scheduling decision -- how many at a time, at what queue
depth, coalesced how -- and scheduling is this package's whole job. A codec
entry point that opens a file has taken that decision away, which is why
lmz's public `decompress()` and `MappedArchive.read()` are both the wrong
shape here despite doing the right arithmetic.

**A decoder returns plaintext.** Not plane-major bytes, not "almost". A
decoder that stops short leaves its last step to be written in the transport
repo, where the format's own tests and its future backends cannot see it.
`merge.cu` in this package is exactly that, and it is registered as a loan
with the interface change that retires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Unit:
    """One unit of coded work, addressed but not read.

    The codec's side of the split: where a block's payload sits, where its
    plaintext lands, and how big each is. Everything about *reading* it --
    which units to coalesce, how deep to queue them, which thread -- is this
    package's and is deliberately absent.

    `opaque` carries whatever the adapter needs to decode this unit later
    (for lmz, the chunk record). Nothing outside the adapter may look inside
    it; that is the rule that keeps coded-stream structure out of here.
    """

    off: int                       # coded payload offset in the container
    clen: int                      # coded payload length
    dst: int                       # where its plaintext begins in the output
    rlen: int                      # plaintext length
    opaque: object = None


@dataclass(frozen=True)
class Tensor:
    """Where one tensor lives in the restored output."""

    name: str
    dtype: str
    shape: tuple
    member: str
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Capabilities:
    """Which decoders can read this archive, and what stops the others.

    Invariant I3: the archive declares this, rather than the transport layer
    holding a copy of an encoder-side threshold. Until a format publishes it,
    an adapter derives it and says so through `derived`, which is what makes
    `gpu_chunk_size()` visibly a loan rather than a feature.
    """

    host: bool = True              # a CPU decoder can read every unit
    device: bool = False           # a device decoder can read every unit
    device_share: float = 0.0      # fraction of plain bytes a device can take
    why_not: str = ""              # what stops the rest, in one sentence
    # LOAN ARTEFACT — expires with I3. `advice` exists only to tell a writer
    # which encoder flag would make this archive device-readable, which is
    # necessary only while the format does not declare its own capabilities.
    # When it does, this field and `derived` both go. Not permanent interface.
    advice: str = ""               # what the writer could do differently
    derived: bool = True           # inferred here, not declared by the format


@dataclass(frozen=True)
class Cost:
    """One cost constant, with where it came from and how sure it is.

    An interval rather than a number, because the decoder's compute cost per
    byte is known from a single point on a single card that fits a
    compute-bound and a bandwidth-bound reading equally well -- and the two
    differ by 8x on a 2-CU integrated GPU, which is the class this project
    exists for. `gate.py` consumes these and must keep refusing to pick.
    """

    name: str
    low: float | None = None
    high: float | None = None
    unit: str = ""
    provenance: str = ""

    @property
    def known(self) -> bool:
        return self.low is not None and self.high is not None

    @property
    def settled(self) -> bool:
        """Whether the interval has collapsed to a measured value."""
        return self.known and abs(self.high - self.low) <= 1e-9 * max(1.0, self.high)


@dataclass(frozen=True)
class CodecCost:
    """The decoder's cost model, as the codec publishes it.

    Owned by the codec, not by this package: `k` is a property of a decoder
    implementation and moves when that implementation does. `gate.py` used to
    hold it as a literal, which put a constant nobody owned in the wrong
    repository and orphaned the measurement that would settle it.
    """

    k: Cost = field(default_factory=lambda: Cost("k"))
    lanes: Cost = field(default_factory=lambda: Cost("lanes"))
    bytes_per_symbol: Cost = field(default_factory=lambda: Cost("bytes_per_symbol"))

    # The rest of the curve's inputs. These were held as literals in `gate.py`
    # for as long as no codec published them, which made them copies of
    # somebody else's constants sitting in the wrong repository -- the exact
    # defect this record exists to remove, and one that survives being correct
    # today because it goes stale silently.
    expansion: Cost = field(default_factory=lambda: Cost("expansion"))
    achieved_fraction: Cost = field(
        default_factory=lambda: Cost("achieved_fraction"))

    # The kernel's shared-memory request, which is what decides how many
    # blocks a unit holds and therefore how many lanes are resident. Published
    # as structure rather than as prose in a provenance string, so a consumer
    # no longer has to restate it to use it.
    shmem_lut_bytes: Cost = field(
        default_factory=lambda: Cost("shmem_lut_bytes"))
    shmem_per_group_bytes: Cost = field(
        default_factory=lambda: Cost("shmem_per_group_bytes"))
    block_threads: Cost = field(default_factory=lambda: Cost("block_threads"))

    # The occupancy `k` was measured under. Not a footnote: it is what the
    # interval means, and a device below it is predicted optimistically.
    blocks_per_unit_at_measurement: Cost = field(
        default_factory=lambda: Cost("blocks_per_unit_at_measurement"))

    # Whether `k` came from one point or from a sweep. A single-point fit
    # cannot distinguish a compute-bound from a bandwidth-bound reading, so a
    # consumer that picks between them needs to know which it has.
    k_single_point: bool | None = None

    source: str = ""

    @property
    def published(self) -> bool:
        """Whether the codec published this rather than the adapter deriving it."""
        return bool(self.source) and "derived" not in self.source


@dataclass(frozen=True)
class Option:
    """One encoder option the codec declares.

    `kind` is the load-bearing field and the reason this is a record rather
    than a bare default. **format** options describe what the bytes become and
    are the codec's to define -- level, block size, which sub-codecs to try.
    **schedule** options describe how the work is spread over this machine and
    are lmsluice's to set, even though the codec is the one that accepts them.

    Keeping the two apart in the declaration is what stops the write side
    leaking the way `gpu_chunk_size()` did: a caller can pass a format option
    through without understanding it, and must never invent one.
    """

    name: str
    default: object = None
    kind: str = "format"           # "format" | "schedule"
    describes: str = ""


@runtime_checkable
class Encoder(Protocol):
    """Writing an archive.

    Separate from `Codec` because encoding does not need an opened container,
    and because the two halves have different owners in the same sentence:
    **lmsluice decides whether to write compressed on this machine; the codec
    decides how.** So the options are declared by the codec and merely chosen
    by the caller -- the caller never names one the codec did not publish.
    """

    def encode_options(self) -> dict:
        """name -> Option, everything this codec accepts and what it means."""

    def encode(self, src: str, dst: str, **opts):
        """Write an archive. Options must be keys of `encode_options()`."""


@runtime_checkable
class Codec(Protocol):
    """A coded container, addressed and decoded but never read by itself.

    lmz is the first implementation. The operations are the ones
    `archive.py`, `model.py`, `probe.py` and `devdecode.py` actually use, and
    the list is simultaneously the request to lmz: every entry that an
    adapter has to fake today is a loan in `docs/boundary.md`.
    """

    plain_bytes: int
    coded_bytes: int

    def ratio(self) -> float:
        """Coded bytes per plain byte. Below 1 is a saving."""

    def units(self):
        """Every unit of coded work, in container order."""

    def units_for(self, spans):
        """The units touching any of `spans`, once, in container order."""

    def cover(self, spans) -> tuple:
        """(lo, hi) — the plain range the units covering `spans` actually fill.

        A block is the smallest decodable thing and does not stop where a
        tensor does, so a caller that allocates only what it asked for has
        allocated too little. Format knowledge, hence the codec's.
        """

    def decode_chunk(self, unit: Unit, payload, out, off: int) -> int:
        """Decode one unit into `out` at `off`. Pure: no I/O, no threads."""

    def tensor_index(self) -> dict:
        """name -> Tensor, from the container's own metadata."""

    def members(self):
        """The files inside the container."""

    def span_of(self, member: str | None) -> tuple:
        """The plain byte range of one member, or of everything."""

    def capabilities(self) -> Capabilities:
        """Which decoders can read this archive."""

    def cost_model(self) -> CodecCost:
        """The decoder's published cost constants, with provenance."""


def missing(name: str) -> Cost:
    """A constant the codec has not published, recorded as absent."""
    return Cost(name, provenance="not published by this codec")
