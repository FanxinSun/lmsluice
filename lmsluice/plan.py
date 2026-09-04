"""Which route moves the bytes faster, and by how much.

A route is a chain of stages, and the stages overlap. That is the whole model,
and everything else here follows from it. Reading a compressed model is two
stages -- pull the coded bytes off the source, turn them into plain ones --
and if they run at the same time the route costs the slower of the two rather
than their sum. Writing one is the same two stages in the other order.

Which gives an answer that is short enough to check by hand. For N plain bytes
and an archive whose coded size is f times its plain size:

    plain read   : N / link
    coded read   : max(fN / link, N / decode)
    speedup      : min(1/f, decode / link)

The gate is the second term, and it does not mention f at all: **compression
pays on a path exactly when the codec is faster than the link it is feeding
from**, and the ratio then decides how much it pays, up to a ceiling of 1/f.
Writing is symmetric with `encode` in place of `decode`. Both statements are
arithmetic rather than a finding, so they hold on any machine; what changes
from machine to machine is the two rates.

**This is the same identity `../gate.py` holds, and the two are deliberately
separate.** `gate.py` answers the question about a machine that is not here:
it *predicts* a decode rate from a device's compute and bandwidth, carries the
unmeasured constant `k` as an interval rather than picking a reading, and has
no imports so it can be copied onto someone else's box and run. This module
answers the question about the machine that is here, and takes decode as a
number somebody measured. Predicting where you can and measuring where you
can are two jobs, and collapsing them would mean either importing a package
into a file whose value is that it needs none, or letting a measurement be
replaced by a model.

What they must not do is disagree, so `speedup` is asserted equal to
`gate.speedup` across a sweep in the test suite. The duplication is pinned
rather than removed.

Three consequences worth stating, because each has been got wrong somewhere:

**A fast disk is a reason not to compress.** Above the gate the coded route
loses, and it loses by more the faster the disk is. A loader that always takes
the compressed path is choosing to be slower on exactly the machines that
would otherwise be fastest.

**The ratio cannot rescue a slow codec.** `min` is a floor, not a sum. Saving
40% of the bytes buys nothing on a path where the decoder is already the
bottleneck, which is the shape of every "but it's smaller" argument.

**Expanding first is a third route and it is much worse than both.** A user
without this tool decompresses to a file and then opens it, paying the source
twice and a write in between. It is named here so the comparison a tool
usually makes -- coded against plain -- does not hide the one users actually
face.
"""

from __future__ import annotations

from dataclasses import dataclass

# How much faster a route must be before it is worth switching to.
#
# Not a safety factor: a hedge against the measurement. Decode timings on a
# desktop with other work on it vary by tens of percent run to run -- lmz's
# own documents say 40% -- so a decision made on a 5% margin is a decision
# made on noise. Below this the plain route wins ties, because it is the route
# with no second failure mode and no CPU spent.
MARGIN = 1.15


@dataclass(frozen=True)
class Stage:
    """One link: how many bytes cross it, and how fast they cross."""

    name: str
    nbytes: int
    rate: float | None                 # bytes/s, or None if never measured
    pool: str = ""                     # stages sharing a pool run serially
    phase: str = ""                    # phases run one after another

    @property
    def seconds(self) -> float | None:
        if not self.rate:
            return None
        return self.nbytes / self.rate


@dataclass(frozen=True)
class Route:
    """A way of getting the bytes there."""

    name: str
    stages: tuple[Stage, ...]
    overlapped: bool = True
    note: str = ""
    peak: int = 0                      # disk high-water, when it is not traffic

    @property
    def known(self) -> bool:
        return all(s.seconds is not None for s in self.stages)

    @property
    def seconds(self) -> float | None:
        """Pipelined, a route costs its slowest stage; serial, their sum.

        `overlapped` is a claim about the implementation, not about the world.
        It is true here because `transport` runs the stages on separate pools
        with a queue between them; a caller that decompresses and then reads
        must set it False and gets the sum, which is the honest cost of doing
        it that way.
        """
        if not self.known:
            return None
        if not self.overlapped:
            return sum(s.seconds for s in self.stages)
        return sum(max(t for t, _ in group) for group in self._phases())

    def _phases(self):
        """`_pools()`, cut into phases that run one after another.

        Two levels, because a route can need both. Sealing a checkpoint is the
        case that forced it: the archive is written in one pass whose encode
        and write overlap, and then read back and encrypted in a second pass
        that cannot start until the first has finished. One level could say
        "all parallel" or "all serial" and both are wrong here -- the first by
        making the second pass free, the second by charging for an overlap that
        does happen.

        A route that names no phase has one, so this changes nothing for any
        route written before it existed.
        """
        order, groups = [], {}
        for s in self.stages:
            if s.phase not in groups:
                order.append(s.phase)
                groups[s.phase] = []
        for t, slowest in self._pools():
            groups[slowest.phase].append((t, slowest))
        return [groups[name] for name in order if groups[name]]

    def _pools(self):
        """(seconds, slowest stage) per pool, summing the stages inside one.

        Two stages share a pool when they run on the same threads, and then
        they are serial with each other however parallel the route is: a
        sealed archive decrypts on the fetch pool, so its bytes are read and
        then decrypted before the decode pool sees them, while that decode
        still overlaps both. Modelling it as one more parallel stage would
        have priced decryption at zero until it was slower than everything
        else, which is exactly the case where it stops being free.

        A stage with no pool is its own, so a route that names none behaves as
        it did before this existed: the slowest stage, and no sums.
        """
        groups: dict[str, list] = {}
        for i, s in enumerate(self.stages):
            groups.setdefault(s.pool or f"\x00{i}", []).append(s)
        return [(sum(s.seconds for s in g), max(g, key=lambda s: s.seconds))
                for g in groups.values()]

    @property
    def limited_by(self) -> str:
        """The stage to attack. The binding pool, then its slowest stage --
        so a fetch pool that is 90% decryption says `decrypt`, not `read`."""
        if not self.known:
            return "unknown"
        if not self.overlapped:
            return max(self.stages, key=lambda s: s.seconds).name
        return max(self._pools(), key=lambda p: p[0])[1].name

    @property
    def peak_bytes(self) -> int:
        """The disk high-water mark: what must exist at once, not in total.

        `device_bytes` is traffic and answers "what does this cost to move".
        This answers "will it fit", which is a different question with a
        different answer -- a route that writes a file and then rewrites it
        moves each byte twice and needs both copies on disk at the same moment.
        """
        return self.peak or self.device_bytes

    @property
    def device_bytes(self) -> int:
        """Bytes that actually cross a link -- storage, network or PCIe.

        The other number a route is judged on, and the one that does not
        depend on any measurement: it is why a slower route is sometimes still
        the right one, on a metered link or a full disk.
        """
        return sum(s.nbytes for s in self.stages
                   if s.name in ("read", "write", "pcie"))


@dataclass(frozen=True)
class Decision:
    """What to do, why, and how sure."""

    chosen: Route
    routes: tuple[Route, ...]
    speedup: float | None              # of the winner over the plain route
    confident: bool
    why: str
    margin: float = MARGIN

    @property
    def coded(self) -> bool:
        return self.chosen.name == "coded"

    def explain(self) -> str:
        peaks = any(r.peak_bytes != r.device_bytes for r in self.routes)
        head = (f"  {'route':<10} {'seconds':>9} {'GB moved':>9}"
                + (f" {'GB on disk':>11}" if peaks else "")
                + "  limited by")
        lines = [head]
        for r in self.routes:
            secs = "unknown" if r.seconds is None else f"{r.seconds:.3f}"
            lines.append(f"  {r.name:<10} {secs:>9} "
                         f"{r.device_bytes / 1e9:>9.2f}"
                         + (f" {r.peak_bytes / 1e9:>11.2f}" if peaks else "")
                         + f"  {r.limited_by}"
                         + (f"   ({r.note})" if r.note else ""))
        lines.append(f"  -> {self.chosen.name}: {self.why}")
        return "\n".join(lines)


def read_plan(plain_bytes: int, coded_bytes: int | None, *,
              source: float | None, decode: float | None,
              margin: float = MARGIN,
              expand_write: float | None = None,
              decrypt: float | None = None) -> Decision:
    """Choose between reading a model plain and reading it coded.

    `source` is the rate of the link the bytes come over; `decode` the rate
    the machine turns coded bytes into plain ones, counted in plain bytes.
    Either may be None, and then no claim is made about the route it belongs
    to. `expand_write` enables the third route -- decompress to disk, then
    read it back -- which needs a write rate to price.

    `coded_bytes` is None when the target is a plain file and nothing is known
    about what a coded copy of it would weigh. There is then one route and no
    decision, which is a different thing from a decision that came out in
    plain's favour, and is reported as such.
    """
    only = Route("plain", (Stage("read", plain_bytes, source),))
    if coded_bytes is None:
        return Decision(only, (only,), None, False,
                        "the target is not compressed and no ratio is known "
                        "for it, so there is nothing to compare", margin)
    # Decryption is priced on the fetch pool, where it runs: coded bytes in,
    # coded bytes out, serial with the read that produced them. `None` when
    # the archive is not sealed, and then the route is what it always was.
    fetch = [Stage("read", coded_bytes, source, pool="fetch")]
    if decrypt is not None:
        fetch.append(Stage("decrypt", coded_bytes, decrypt, pool="fetch"))
    routes = [only,
              Route("coded", (*fetch,
                              Stage("decode", plain_bytes, decode,
                                    pool="decode")))]
    if expand_write:
        # Not overlapped, and not by an oversight: the file has to exist
        # before it can be opened, so nothing here runs at the same time as
        # anything else. This is what taking a compressed checkpoint and
        # running `lmz decompress` on it costs.
        routes.append(Route(
            "expand", (Stage("read", coded_bytes, source),
                       Stage("decode", plain_bytes, decode),
                       Stage("write", plain_bytes, expand_write),
                       Stage("reread", plain_bytes, source)),
            overlapped=False, note="decompress to disk first"))
    return _decide(tuple(routes), source, decode, margin, "decode")


def seal_rate(*, source: float | None, sink: float | None,
              encrypt: float | None) -> float | None:
    """How fast the sealing pass moves coded bytes, from the rates around it.

    Derived rather than measured, because it is a serial loop over three things
    this machine has already been measured on: read a unit, encrypt it, write
    it. `sealed.seal_file` does exactly that and nothing concurrently, so the
    reciprocals add. Stated this way it re-evaluates on another machine without
    re-measuring -- a slow disk and a fast cipher give a different answer from
    a fast disk and a phone-class one, and neither needs a new constant.

    Checked against the development box: source 6.34, encrypt ~18 single
    threaded, sink ~2 GB/s predicts 1.40 GB/s; sealing a 202 MB lmz archive
    measured 1.24. Close enough to price a decision, and the gap is the
    per-unit Python overhead the model does not carry.
    """
    if not source or not sink or not encrypt:
        return None
    return 1.0 / (1.0 / source + 1.0 / sink + 1.0 / encrypt)


def write_plan(plain_bytes: int, ratio: float, *, sink: float | None,
               encode: float | None, margin: float = MARGIN,
               seal: float | None = None) -> Decision:
    """Choose between writing a model plain and writing it coded.

    The mirror of `read_plan`, and worth having separately because the answer
    is often the other one: a sustained write is usually slower than the read
    on the same device, so a codec can clear the gate on the way out and miss
    it on the way in. That is the case for compressing training checkpoints on
    machines where compressing the weights you load would be a mistake.

    `seal` enables the encrypted row, at the rate `seal_rate` gives. Encryption
    is an envelope around a finished archive, so it is a **second pass**: the
    archive is written, then read back and rewritten sealed. Both costs are
    stated rather than absorbed -- the wall clock as its own phase, and the
    disk high-water as `peak_bytes`, because a 70 GB checkpoint needs to know
    it will hold two compressed copies at once before it starts, not after.
    """
    coded_bytes = int(plain_bytes * ratio)
    routes = [Route("plain", (Stage("write", plain_bytes, sink),)),
              Route("coded", (Stage("encode", plain_bytes, encode),
                              Stage("write", coded_bytes, sink)))]
    if seal:
        routes.append(Route(
            "sealed",
            (Stage("encode", plain_bytes, encode, phase="code"),
             Stage("write", coded_bytes, sink, phase="code"),
             Stage("seal", coded_bytes, seal, phase="seal")),
            note="a second pass; holds both copies",
            peak=coded_bytes * 2))
    return _decide(tuple(routes), sink, encode, margin, "encode")


def _decide(routes, device: float | None, codec: float | None,
            margin: float, codec_name: str) -> Decision:
    plain, coded = routes[0], routes[1]
    if plain.seconds is None or coded.seconds is None:
        missing = "the link" if device is None else f"the {codec_name}"
        return Decision(plain, routes, None, False,
                        f"{missing} has not been measured, so the plain route "
                        f"is taken as the one that cannot be wrong")

    speedup = plain.seconds / coded.seconds
    gate = codec / device if device else float("inf")
    if speedup >= margin:
        why = (f"{speedup:.2f}x faster: {codec_name} runs at {gate:.2f}x the "
               f"link, and the ratio caps the win at {plain.device_bytes / max(1, coded.device_bytes):.2f}x")
        return Decision(coded, routes, speedup, True, why, margin)
    if speedup <= 1 / margin:
        why = (f"{1 / speedup:.2f}x slower: {codec_name} runs at {gate:.2f}x "
               f"the link, and below 1.00x the coded route cannot win")
        return Decision(plain, routes, speedup, True, why, margin)
    why = (f"too close to call at {speedup:.2f}x, inside the {margin:.2f}x "
           f"margin these rates are measured to; plain wins ties")
    return Decision(plain, routes, speedup, False, why, margin)


def vram_plan(plain_bytes: int, coded_bytes: int | None, *,
              source: float | None, pcie: float | None,
              host_decode: float | None = None,
              device_decode: float | None = None,
              margin: float = MARGIN) -> Decision:
    """Choose how to get a model into device memory.

    Two links now, not one -- storage and PCIe -- and the route decides how many
    bytes cross each. That is what makes this different from `read_plan` rather
    than a longer version of it:

        plain          max(N/L,  N/P)              N over both
        host-decode    max(fN/L, N/Dh, N/P)        fewer off disk, all of it over PCIe
        device-decode  max(fN/L, fN/P, N/Dd)       fewer over BOTH, decoded where it lands

    `device-decode` is the only route that shortens both links, and it is also
    the only one whose decode is not competing with the host for memory
    bandwidth. That is the whole argument for a GPU decoder on the load path,
    and it is why the interesting number is not the decoder's headline rate but
    the ratio: a decoder 14x faster than PCIe converts every point of the
    archive's ratio into load speed and nothing beyond that helps.

    `host-decode` is priced even where it loses, because on a machine with no
    device decoder it is the only coded route there is -- and it loses here for
    the same reason the CPU route loses on the storage path, which is one fact
    rather than two.
    """
    routes = [Route("plain", (Stage("read", plain_bytes, source),
                              Stage("pcie", plain_bytes, pcie)))]
    if coded_bytes is not None:
        routes.append(Route("host-decode",
                            (Stage("read", coded_bytes, source),
                             Stage("decode", plain_bytes, host_decode),
                             Stage("pcie", plain_bytes, pcie))))
        routes.append(Route("device-decode",
                            (Stage("read", coded_bytes, source),
                             Stage("pcie", coded_bytes, pcie),
                             Stage("decode", plain_bytes, device_decode)),
                            note="decoded in VRAM"))
    known = [r for r in routes if r.seconds is not None]
    if len(known) < 2:
        why = ("not enough of the path has been measured to compare routes; "
               "the plain one is taken as the one that cannot be wrong")
        return Decision(routes[0], tuple(routes), None, False, why, margin)
    # Ties broken by bytes moved. Two routes that finish together are not
    # equivalent: the one that puts less on the wire leaves more of a shared
    # disk, a metered link and a contended PCIe bus for everything else.
    best = min(known, key=lambda r: (r.seconds, r.device_bytes))
    speedup = routes[0].seconds / best.seconds
    if best is routes[0] or speedup < margin:
        loser = min((r for r in known if r is not routes[0]),
                    key=lambda r: r.seconds)
        return Decision(routes[0], tuple(routes), speedup,
                        speedup <= 1 / margin,
                        f"no coded route clears the {margin:.2f}x margin; the "
                        f"best is {loser.name} at {routes[0].seconds / loser.seconds:.2f}x",
                        margin)
    return Decision(best, tuple(routes), speedup, True,
                    f"{speedup:.2f}x faster than plain, limited by "
                    f"{best.limited_by}", margin)


def gate_ratio(codec_rate: float | None, device_rate: float | None) -> float | None:
    """The one ratio. Above 1, compression is free on this path.

    Named `gate_ratio` rather than `gate` because `lmsluice.gate` is a module --
    the curve for devices that are not present -- and a function shadowing a
    submodule of the same name is a collision that resolves differently
    depending on import order.
    """
    if not codec_rate or not device_rate:
        return None
    return codec_rate / device_rate
