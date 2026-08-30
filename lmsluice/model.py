"""One way to open a model, whichever route turns out to be faster.

A caller asks for tensors. It should not have to know whether they arrived
from a plain safetensors file, from an lmz archive on the same disk, or from
an archive over a network link -- and it certainly should not have to know
which of those is quickest on the machine it woke up on, because that is a
measurement and it changes from box to box.

So `open_model` takes a target and returns the same object either way: a
tensor index, and buffers on request. What differs underneath is the route,
and `Model.plan` says which was taken and why, so a caller that wants to
disagree can, and a caller that wants to check the tool was right can.

Three ways to get the bytes, for three shapes of consumer:

`map()` is the fastest and the laziest. On a plain file it is an mmap and no
bytes move until they are touched, which is what makes partial access -- one
adapter, one expert, one rank's shard -- nearly free. It has no coded form:
a compressed archive cannot be mapped, and the honest answer is to say so
rather than to decode the whole thing and call it a map.

`load()` reads for real, in parallel, into one buffer. This is the whole-model
case, and the one the plan prices.

`stream()` is the same work under a memory ceiling, yielding tensors as they
land. It exists because the machines this is for are the ones where the model
does not fit twice: a consumer that copies each tensor to a device as it
arrives needs a window, not the whole checkpoint, and gets the pipeline's
overlap for free while it does.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass

from . import plan as _plan
from .archive import Archive, Tensor
from .probe import default_threads
from .rates import Profile, storage_key
from .source import FileSource, open_source
from .transport import split, transport

# Plain bytes one pipeline job carries. Large enough that dispatching it costs
# nothing next to the work, small enough that a queue of them is a memory
# bound rather than a memory plan. lmz uses the same figure for the same
# reason on its own decode path.
JOB_BYTES = 4 << 20

# Default ceiling for `stream`, when the caller does not set one. A window
# rather than a fraction of RAM: what the consumer does with each tensor is
# unknown, and a ceiling that grows with the machine would defeat the purpose
# on the machine that needs it most.
STREAM_BUDGET = 256 << 20


class NotMappable(RuntimeError):
    """Raised when a coded route is asked for a zero-copy mapping."""


@dataclass(frozen=True)
class Plan:
    """What was decided, and what it was decided on."""

    decision: "_plan.Decision"
    source_rate: float | None
    codec_rate: float | None
    measured: bool
    note: str = ""

    def explain(self) -> str:
        head = (f"  source {_gb(self.source_rate)}   "
                f"decode {_gb(self.codec_rate)}"
                + ("" if self.measured else "   (unmeasured: run `lmsluice probe`)"))
        return head + ("\n" + self.note if self.note else "") + "\n" + \
            self.decision.explain()


def _gb(rate) -> str:
    return "unknown" if not rate else f"{rate / 1e9:.2f} GB/s"


class Model:
    """A safetensors model, opened over whichever route is faster."""

    from_cache = False

    def __init__(self, target: str, *, profile: Profile | None = None,
                 prefer: str | None = None, member: str | None = None,
                 fetch_threads: int | None = None,
                 place_threads: int | None = None, cache: str = "auto"):
        self.target = target
        self.route = "plain"
        self._arc: Archive | None = None
        self._src = None
        self._mm = None
        f, p = default_threads()
        self._fetch_threads = fetch_threads or f
        self._place_threads = place_threads or p

        src = open_source(target)
        coded = _is_coded(src)
        if not coded and cache != "off" and isinstance(src, FileSource):
            # An existing cache entry is used without being asked; building one
            # is never automatic. See `cache.py` for why that asymmetry.
            from . import cache as _cache

            hit = _cache.find(target)
            if hit:
                src.close()
                target, self.target = hit, hit
                src = open_source(hit)
                coded = _is_coded(src)
                self.from_cache = True
        if coded:
            src.close()
            self._arc = Archive(target)
            self.route = "coded"
            self._member = member or (self._arc.members[0].path
                                      if self._arc.members else None)
            lo, hi = self._arc.span_of(self._member)
            self._base, self._end = lo, hi
            self.tensors = {n: t for n, t in self._arc.tensors().items()
                            if self._member is None or t.member == self._member}
            self.plain_bytes = hi - lo
            self.coded_bytes = self._arc.coded_bytes
        else:
            self._src = src
            self._member = os.path.basename(target)
            self._base, self._end = 0, src.size
            self.tensors = _safetensors_index(src, self._member)
            self.plain_bytes = src.size
            self.coded_bytes = src.size

        self.profile = profile if profile is not None else Profile.load()
        self.plan = self._make_plan()
        if prefer in ("plain", "coded") and prefer != self.route:
            # A caller may override, and is told rather than silently obeyed:
            # asking for the coded route on a plain file is not something this
            # can do, and asking for the plain route on an archive means
            # expanding it, which `to_file` does explicitly.
            self.plan = Plan(self.plan.decision, self.plan.source_rate,
                             self.plan.codec_rate, self.plan.measured,
                             note=f"  (asked for {prefer}; the target is "
                                  f"{self.route}, which is what it is)")

    # -- what is in it ----------------------------------------------------
    @property
    def ratio(self) -> float:
        return self.coded_bytes / self.plain_bytes if self.plain_bytes else 1.0

    def span_of(self, names) -> list[tuple[int, int]]:
        """The byte ranges holding a set of tensors, merged where adjacent.

        Merged because two tensors that sit next to each other are one read,
        and because a safetensors file lays tensors out in the order its
        header names them, so a contiguous group is the common case rather
        than the lucky one.
        """
        spans = sorted((self.tensors[n].start, self.tensors[n].end)
                       for n in names)
        out: list[list[int]] = []
        for lo, hi in spans:
            if out and lo <= out[-1][1]:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        return [(a, b) for a, b in out]

    # -- getting the bytes ------------------------------------------------
    def map(self) -> memoryview:
        """The whole member, mapped, with no bytes moved until touched."""
        if self.route == "coded":
            raise NotMappable(
                f"{self.target} is an lmz archive: a coded file cannot be "
                f"mapped. Use load() or stream(), or to_file() to expand it.")
        if not isinstance(self._src, FileSource):
            raise NotMappable("only a local file can be mapped")
        if self._mm is None:
            self._mm = mmap.mmap(self._src.fileno(), 0,
                                 access=mmap.ACCESS_READ)
        return memoryview(self._mm)

    def load(self, names=None, into=None) -> memoryview:
        """Read tensors for real, in parallel, into one buffer.

        With `names`, only the ranges holding those tensors are moved -- and
        on the coded route only the blocks covering them are read and decoded,
        which is the property that makes a compressed archive usable for a
        partial load at all.

        The returned view spans the member, so a tensor's own view is taken at
        its file offset; ranges that were not asked for are untouched rather
        than absent, which keeps `tensor()` working on one buffer whatever
        subset built it.
        """
        spans = ([(self._base, self._end)] if names is None
                 else self.span_of(names))
        buf = into if into is not None else bytearray(self.plain_bytes)
        if len(buf) < self.plain_bytes:
            raise ValueError(f"buffer holds {len(buf)}, member needs "
                             f"{self.plain_bytes}")
        # Clipped to the member: blocks may overhang it at either end in a
        # multi-file archive, and the overhang belongs to a different member,
        # not in this buffer. `_gather` handles the ends; here the member
        # buffer is the whole of what the caller sees.
        self._gather(spans, into=buf, base=self._base,
                     limit=(self._base, self._end))
        return memoryview(buf)[:self.plain_bytes]

    def stream(self, names=None, *, budget: int = STREAM_BUDGET):
        """Yield (name, view) as tensors land, holding at most `budget` bytes.

        Each window is dropped before the next is read, so a consumer that
        copies onward -- to a device, to a socket, to a quantiser -- never
        holds more than one window plus whatever it keeps. A tensor larger
        than the budget gets a window of its own rather than being refused:
        the ceiling is a target, and a single weight matrix that exceeds it is
        the caller's arithmetic to fix, not this function's to fail on.
        """
        order = sorted(names or self.tensors, key=lambda n: self.tensors[n].start)
        for group in _windows(order, self.tensors, budget):
            lo = self.tensors[group[0]].start
            hi = max(self.tensors[n].end for n in group)
            base, buf = self._gather([(lo, hi)])
            view = memoryview(buf)
            for name in group:
                t = self.tensors[name]
                yield name, view[t.start - base:t.end - base]
            del view, buf

    # Plain bytes staged through one pinned window. Two of these exist at a
    # time so a copy can overlap the next fill, and the size is a parameter
    # because the right value is a property of the machine: enough to keep the
    # link busy, small enough that page-locking it is not itself the problem.
    WINDOW = 128 << 20

    def to_device(self, names=None, *, context=None, window: int | None = None,
                  route: str | None = None):
        """Fill a CUDA allocation with the model, and hand it back.

        Returns a `cuda.DeviceBuffer`, which carries `__cuda_array_interface__`
        -- so `torch.as_tensor(buf, device="cuda")` and `cupy.asarray(buf)`
        adopt the memory with no copy, and lmsluice imports neither.

        Windows are cut at block boundaries on the coded route rather than at
        round numbers. A block straddling a window edge would otherwise be
        decoded twice, once for each side, and paying an eighth of the decode
        again to keep the arithmetic tidy is the wrong trade.
        """
        from . import cuda

        ok, why = cuda.available()
        if not ok:
            raise RuntimeError(f"no CUDA device: {why}")
        ctx = context or cuda.Context()
        own_ctx = context is None
        window = window or self.WINDOW
        chosen = route or ("device" if self.route == "coded" else "plain")

        if chosen == "device" and self.route == "coded":
            # Coded bytes cross PCIe and are turned into plaintext where they
            # land. Falls back rather than fails: a machine with no nvcc, or an
            # archive whose codec the kernel cannot read, still gets its model.
            from . import devdecode

            dd = devdecode.for_context(ctx)
            chunks = (self._arc.chunks_for(spans_for_device(self, names))
                      if dd.available else [])
            takeable = [c for c in chunks if not dd.takes(c)]
            if dd.available and takeable:
                dev = ctx.allocate(self.plain_bytes)
                try:
                    with ctx.stream() as stream:
                        self.device_stats = dd.decode(self._arc, chunks, dev,
                                                      self._base, stream)
                    self.device_route = "device"
                    return dev
                except BaseException:
                    dev.close()
                    if own_ctx:
                        ctx.close()
                    raise
            # Nothing the device can take. Falling through to the host route
            # rather than limping through the device path's per-chunk host
            # fallback, which decodes one chunk at a time with a synchronous
            # copy after each and measured slower than the route it was
            # standing in for.
            if not dd.available:
                self.device_why = dd.why
            elif chunks:
                self.device_why = dd.takes(chunks[len(chunks) // 2])
            chosen = "coded-host"

        spans = ([(self._base, self._end)] if names is None
                 else self.span_of(names))
        dev = ctx.allocate(self.plain_bytes)
        try:
            windows = self._device_windows(spans, window)
            size = max((hi - lo) for lo, hi in windows) if windows else window
            pins = [ctx.pinned(size), ctx.pinned(size)]
            stream = ctx.stream()
            try:
                for i, (lo, hi) in enumerate(windows):
                    pin = pins[i % len(pins)]
                    # Before reusing a staging buffer, wait for the copy that
                    # last read it. The stream is FIFO, so synchronising it
                    # covers that copy and every earlier one.
                    #
                    # This was wrong and the bug was silent: the guard used to
                    # compare against the *most recent* buffer, which with two
                    # alternating buffers is never the one about to be reused,
                    # so it never fired. Each window was written into a buffer
                    # whose DMA might still be in flight, and whether that
                    # corrupted the load depended on whether reading and
                    # decoding the next window took longer than the copy --
                    # so it passed almost always and failed under load.
                    if i >= len(pins):
                        stream.synchronize()
                    view = pin.view()[:hi - lo]
                    # The window's own spans, not the whole request: `_gather`
                    # fills everything it is given, and handing it the full
                    # span list would try to write the entire model into one
                    # staging buffer.
                    self._gather(_clip(spans, lo, hi), into=view, base=lo,
                                 limit=(lo, hi))
                    dev.copy_from(pin, offset=lo - self._base, stream=stream,
                                  nbytes=hi - lo)
                stream.synchronize()
            finally:
                stream.close()
                for p in pins:
                    p.close()
            self.device_route = chosen
            return dev
        except BaseException:
            dev.close()
            if own_ctx:
                ctx.close()
            raise

    device_stats = None
    device_why = ""

    def device_plan(self):
        """Price the three routes into VRAM, from the measured profile.

        A comparison rather than a switch. `to_device` moves whichever form the
        caller opened, because the plain file and the archive are two files and
        choosing between them is the caller's -- what this does is say which
        one they should have opened, with the arithmetic attached.
        """
        from .plan import vram_plan

        st = (self.profile.storage.get(storage_key(self.target))
              if self.profile and self._is_local() else None)
        link = self.profile.storage.get("pcie") if self.profile else None
        codec = self.profile.codecs.get("lmz-cpu") if self.profile else None
        ratio = codec.ratio if codec else None
        coded = (self.coded_bytes if self.route == "coded"
                 else (int(self.plain_bytes * ratio) if ratio else None))
        return vram_plan(self.plain_bytes, coded,
                         source=st.source if st else None,
                         pcie=link.cold if link else None,
                         host_decode=codec.decode if codec else None,
                         device_decode=None)

    def _device_windows(self, spans, window: int):
        """Cut `spans` into staging windows, on block boundaries where they
        exist, so no block is decoded twice."""
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        if self.route != "coded":
            return [(a, min(a + window, hi))
                    for a in range(lo, hi, window)] or [(lo, hi)]
        edges = [c.dst + c.rlen for c in self._arc.chunks_for([(lo, hi)])]
        edges = sorted({e for e in edges if lo < e < hi}) + [hi]
        out, start = [], lo
        for e in edges:
            if e - start >= window or e == hi:
                out.append((start, e))
                start = e
        if start < hi:
            out.append((start, hi))
        return out

    def tensor(self, name: str, buffer=None) -> memoryview:
        """One tensor. Reads only the blocks that hold it."""
        t = self.tensors[name]
        if buffer is not None:
            return memoryview(buffer)[t.start - self._base:t.end - self._base]
        if self.route == "plain" and isinstance(self._src, FileSource):
            return self.map()[t.start:t.end]
        base, buf = self._gather([(t.start, t.end)])
        return memoryview(buf)[t.start - base:t.end - base]

    def to_file(self, path: str, *, overwrite: bool = False) -> int:
        """Write the member out plain. Named `expand`, and priced, in the plan."""
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        buf = bytearray(self.plain_bytes)
        self._gather([(self._base, self._end)], into=buf, base=self._base,
                     limit=(self._base, self._end))
        with open(path, "wb") as fh:
            fh.write(buf)
        return len(buf)

    # -- the pipeline -----------------------------------------------------
    def _gather(self, spans, *, into=None, base=None, limit=None):
        """Fill a buffer with `spans`, and say where the buffer starts.

        On the coded route the buffer has to hold whole blocks, which reach
        past the requested bytes at both ends, so the caller cannot know its
        size or its origin in advance -- that is what comes back. `limit`
        clips the blocks to a range the caller owns, for the case where the
        overhang belongs to a neighbouring member.
        """
        if self.route == "coded":
            arc = self._arc
            lo, hi, runs = arc.cover(spans, JOB_BYTES)
            if limit is not None:
                lo, hi = max(lo, limit[0]), min(hi, limit[1])
            if into is None:
                into, base = bytearray(max(0, hi - lo)), lo
            elif base is None:
                base = lo
            self.last = transport(
                runs, arc.fetch,
                lambda r, p: arc.place(r, p, into, base, clip=(lo, hi)),
                fetch_threads=self._fetch_threads,
                place_threads=self._place_threads,
                inflight=max(4, self._fetch_threads * 2))
            return base, into

        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        if into is None:
            into, base = bytearray(hi - lo), lo
        elif base is None:
            base = lo
        self._fill(spans, into, base)
        return base, into

    def _fill(self, spans, buf, base: int):
        """Move every span into `buf`, which begins at plain offset `base`."""
        src, jobs = self._src, []
        for lo, hi in spans:
            jobs += split(hi - lo, JOB_BYTES, base=lo)
        view = memoryview(buf)

        def place(job, payload):
            off, _ = job
            view[off - base:off - base + len(payload)] = payload
            return len(payload)

        self.last = transport(
            jobs, lambda j: src.pread(j[0], j[1]), place,
            fetch_threads=self._fetch_threads,
            place_threads=max(1, self._place_threads // 2),
            inflight=max(4, self._fetch_threads * 2))
        return self.last

    # -- deciding ---------------------------------------------------------
    def _make_plan(self) -> Plan:
        key = (storage_key(self.target) if self._is_local()
               else str(self.target))
        st = (self.profile.storage.get(key) if self.profile else None)
        codec = (self.profile.codecs.get("lmz-cpu") if self.profile else None)
        source = st.source if st else None
        decode = codec.decode if codec else None
        note = ""
        if st is not None and st.source is None and st.warm_bound:
            # The coded route is unevaluated here, not beaten. Saying which
            # matters: a reader who sees "plain wins" on a machine that could
            # not measure its own disk will believe a verdict nobody reached.
            note = (f"  note: this platform could not measure a cold read "
                    f"({st.method}). The warm rate was "
                    f"{st.warm_bound / 1e9:.2f} GB/s, which is a cache rate "
                    f"and only an upper bound on the link, so the coded route "
                    f"is unevaluated rather than beaten")
        if st and st.layered:
            note = ("  note: a cache below this process serves repeat reads, "
                    "so the cold rate holds on first touch only")
        if codec and codec.archive and self.route == "coded" \
                and os.path.basename(codec.archive) != os.path.basename(self.target):
            note += (("\n" if note else "")
                     + f"  note: decode rate was measured on "
                       f"{os.path.basename(codec.archive)}, not this archive")
        coded_bytes = self.coded_bytes
        if self.route == "plain":
            # There is no coded copy of a plain file. Pricing one anyway is
            # still the useful question -- would compressing this pay? -- but
            # only with a ratio, and the only ratio available is one measured
            # on some other archive, so it is hypothetical and says so.
            ratio = codec.ratio if codec else None
            coded_bytes = int(self.plain_bytes * ratio) if ratio else None
            if ratio:
                note += (("\n" if note else "")
                         + f"  note: the coded row is hypothetical, at the "
                           f"r={ratio:.3f} measured on "
                           f"{os.path.basename(codec.archive) or 'another archive'}")
        decision = _plan.read_plan(
            self.plain_bytes, coded_bytes, source=source, decode=decode,
            expand_write=(st.write if st else None))
        return Plan(decision, source, decode,
                    measured=bool(source and (decode or self.route == "plain")),
                    note=note)

    def _is_local(self) -> bool:
        return isinstance(self._src, FileSource) or (
            self._arc is not None and isinstance(self._arc.source, FileSource))

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._arc is not None:
            self._arc.close()
            self._arc = None
        if self._src is not None:
            self._src.close()
            self._src = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def open_model(target: str, **kw) -> Model:
    """Open a model from a path, a URL, or an lmz archive of either."""
    return Model(target, **kw)


def spans_for_device(model, names):
    """The plain ranges a device load must cover."""
    return ([(model._base, model._end)] if names is None
            else model.span_of(names))


def _clip(spans, lo: int, hi: int):
    """The parts of `spans` inside [lo, hi), dropping what falls outside."""
    out = []
    for a, b in spans:
        a, b = max(a, lo), min(b, hi)
        if a < b:
            out.append((a, b))
    return out


def _windows(order, tensors, budget: int):
    """Group tensors into runs whose byte span fits the budget."""
    group, lo, hi = [], None, None
    for name in order:
        t = tensors[name]
        if group and (max(hi, t.end) - lo) > budget:
            yield group
            group, lo, hi = [], None, None
        if not group:
            lo, hi = t.start, t.end
        group.append(name)
        hi = max(hi, t.end)
    if group:
        yield group


def _is_coded(src) -> bool:
    """Whether this source is a coded archive of any codec we know."""
    if src.size < 8:
        return False
    head = src.pread(0, 8)
    return head[:4] == b"LMZ\x01" or head == b"LMSLUICE"


def _safetensors_index(src, member: str) -> dict:
    """The tensor index of a plain safetensors file.

    Offsets are made absolute here, so that a tensor's `start` means the same
    thing whichever route it came from: lmz records absolute offsets in its
    manifest, and having the two disagree by the length of a header is exactly
    the kind of difference that shows up as one wrong tensor much later.
    """
    head = src.pread(0, 8)
    (n,) = struct.unpack("<Q", head)
    if n <= 0 or n > src.size:
        raise ValueError(f"{src.name}: not a safetensors file")
    header = json.loads(src.pread(8, n))
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        out[name] = Tensor(name, meta.get("dtype", ""),
                           tuple(meta.get("shape", ())), member,
                           base + lo, base + hi)
    return out
