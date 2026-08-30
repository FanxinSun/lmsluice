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
        times = [s.seconds for s in self.stages]
        return max(times) if self.overlapped else sum(times)

    @property
    def limited_by(self) -> str:
        if not self.known:
            return "unknown"
        return max(self.stages, key=lambda s: s.seconds).name

    @property
    def device_bytes(self) -> int:
        """Bytes that actually cross the storage or the network.

        The other number a route is judged on, and the one that does not
        depend on any measurement: it is why a slower route is sometimes still
        the right one, on a metered link or a full disk.
        """
        return sum(s.nbytes for s in self.stages if s.name in ("read", "write"))


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
        lines = [f"  {'route':<10} {'seconds':>9} {'GB moved':>9}  limited by"]
        for r in self.routes:
            secs = "unknown" if r.seconds is None else f"{r.seconds:.3f}"
            lines.append(f"  {r.name:<10} {secs:>9} "
                         f"{r.device_bytes / 1e9:>9.2f}  {r.limited_by}"
                         + (f"   ({r.note})" if r.note else ""))
        lines.append(f"  -> {self.chosen.name}: {self.why}")
        return "\n".join(lines)


def read_plan(plain_bytes: int, coded_bytes: int | None, *,
              source: float | None, decode: float | None,
              margin: float = MARGIN,
              expand_write: float | None = None) -> Decision:
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
    routes = [only,
              Route("coded", (Stage("read", coded_bytes, source),
                              Stage("decode", plain_bytes, decode)))]
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


def write_plan(plain_bytes: int, ratio: float, *, sink: float | None,
               encode: float | None, margin: float = MARGIN) -> Decision:
    """Choose between writing a model plain and writing it coded.

    The mirror of `read_plan`, and worth having separately because the answer
    is often the other one: a sustained write is usually slower than the read
    on the same device, so a codec can clear the gate on the way out and miss
    it on the way in. That is the case for compressing training checkpoints on
    machines where compressing the weights you load would be a mistake.
    """
    coded_bytes = int(plain_bytes * ratio)
    routes = (Route("plain", (Stage("write", plain_bytes, sink),)),
              Route("coded", (Stage("encode", plain_bytes, encode),
                              Stage("write", coded_bytes, sink))))
    return _decide(routes, sink, encode, margin, "encode")


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


def gate(codec_rate: float | None, device_rate: float | None) -> float | None:
    """The one ratio. Above 1, compression is free on this path."""
    if not codec_rate or not device_rate:
        return None
    return codec_rate / device_rate
