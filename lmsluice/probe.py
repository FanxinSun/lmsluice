"""Measuring the two rates the decision turns on.

Everything here answers one of two questions about the machine it is running
on: how fast do bytes cross this link, and how fast does this codec run here.
Neither has a defensible default. A rate copied from a spec sheet, from
another machine, or from the last time anyone looked is how a loader ends up
confidently taking the slower route, so nothing in this package ships a number
-- it ships the means to take one.

Three rules the measurements here follow, each of which has a way of being got
wrong that is easy and quiet:

**Measure the file that will actually be read.** Decode rate is not a property
of the build. It moves with chunk size, with which codecs the encoder chose,
and with how much interpreter runs between native calls -- all properties of
the archive. So the codec probe takes a sample of the real archive rather than
a synthetic one, which also means it needs no fixture and costs a second.

**Say when a number is warm.** A checkpoint is read once and is usually bigger
than the cache that would hold it, so the number that matters is the cold one.
Linux can drop a range from the page cache and macOS can ask an fd to bypass
it; Windows offers a Python process neither. Where the cache cannot be
dropped, the rate is recorded as warm and the plan is told it is warm, rather
than a cache-speed number being passed off as a disk-speed one.

**Look for a cache underneath.** Dropping the page cache does not drop the
hypervisor's, the network filesystem client's, or the drive's. The tell is
that a second cold read is much faster than the first, which is measured here
rather than assumed, and reported as `layered` -- on such a machine the honest
cold number can only be taken once, on first touch, and every later
measurement is an upper bound. The test only fires when that lower cache was
itself cold, so it detects layering and never rules it out, and the cold rate
is an upper bound wherever it did not fire.
"""

from __future__ import annotations

import os
import shutil
import statistics
import tempfile
import threading
import time

from .rates import Codec, Profile, Storage, host_info, storage_key
from .source import FileSource, Source, open_source
from .transport import transport

# What a probe reads or writes when it is not told otherwise. Big enough to be
# past a drive's write-combining buffer and any small cache, small enough that
# probing is not itself a load. A caller with a slow link should pass less; the
# probe is a measurement, not a benchmark.
SAMPLE = 256 << 20

# Queue depths tried against a source. Sequential, a shallow parallel, and a
# deep one: an NVMe device needs the last, a spinning disk is fastest on the
# first, and a network source is often still climbing at the last. Which wins
# is the measurement -- that is the point of sweeping rather than picking.
DEPTHS = (1, 4, 16)

# A second cold read this much faster than the first means something below
# this process is caching too. Well clear of run-to-run noise, which on a busy
# desktop reaches tens of percent.
LAYERED_AT = 1.5


def default_threads() -> tuple[int, int]:
    """(fetch, place) to start from, before anything has been measured.

    Fetch wants depth and place wants cores, and neither wants the machine's
    whole thread count: the fetch stage blocks on the device, so more of it
    costs nothing but does nothing either past the device's own depth, and the
    place stage is capped by whatever serialises between its native calls.
    These are starting points that the codec probe then replaces with a
    measurement.
    """
    cpus = os.cpu_count() or 4
    return max(4, min(16, cpus)), max(1, min(8, cpus // 2))


# -- storage ---------------------------------------------------------------
def read_rate(src: Source, *, sample: int = SAMPLE,
              depths=DEPTHS, key: str = "") -> Storage:
    """How fast this source delivers, cold if the platform allows it."""
    sample = max(1 << 20, min(sample, src.size))
    regions = _regions(src.size, sample, len(depths) + 2)
    uncache = getattr(src, "uncache", None)
    method = uncache() if uncache else ""

    def timed(region, depth):
        if uncache:
            uncache()
        off, length = region
        return _read_at(src, off, length, depth)

    first = timed(regions[0], 1)
    best, best_depth = first, 1
    for depth, region in zip(depths, regions[1:]):
        if depth == 1:
            continue
        r = timed(region, depth)
        if r > best:
            best, best_depth = r, depth

    again = timed(regions[0], 1)
    layered = None
    if method:
        layered = True if again > first * LAYERED_AT else None
        if layered:
            # The sweep above was measured against whatever is underneath, so
            # it is an upper bound on the cold rate rather than the cold rate.
            # The first read is the only one that was not, so it is what is
            # reported -- and the depth that won is still worth keeping,
            # because the right queue depth does not change with the cache.
            best = max(first, best if best_depth == 1 else first)
    warm = _read_at(src, *regions[0], 1)

    return Storage(key=key or storage_key(src.name), cold=best if method else None,
                   warm=warm, layered=layered, depth=best_depth,
                   method=method or "cannot drop the cache on this platform",
                   sample_bytes=sample)


def write_rate(directory: str, *, sample: int = SAMPLE) -> float | None:
    """Sustained write, fsync included, on the filesystem holding `directory`.

    fsync is in the timing because a checkpoint that is not on the disk is not
    a checkpoint. Leaving it out measures the page cache, which on this class
    of machine is several times faster than the device and would make every
    write look like a win.
    """
    free = shutil.disk_usage(directory).free
    sample = min(sample, max(0, free - (1 << 30)))
    if sample < (16 << 20):
        return None
    block = bytearray(os.urandom(min(sample, 8 << 20)))
    fd, path = tempfile.mkstemp(dir=directory, prefix=".lmsluice-probe-")
    try:
        written, started = 0, time.perf_counter()
        while written < sample:
            written += os.write(fd, block[:min(len(block), sample - written)])
        os.fsync(fd)
        return written / (time.perf_counter() - started)
    finally:
        os.close(fd)
        os.unlink(path)


def _regions(size: int, sample: int, n: int) -> list[tuple[int, int]]:
    """`n` disjoint regions of `sample` bytes, spread over the file.

    Disjoint so that one measurement does not warm the next, and spread so
    that a device with a fast outer zone or a partly cached file does not have
    every sample land in the same place.
    """
    sample = min(sample, size)
    if size <= sample:
        return [(0, size)] * n
    step = max(1, (size - sample) // max(1, n - 1))
    return [(min(i * step, size - sample), sample) for i in range(n)]


def _read_at(src: Source, off: int, length: int, depth: int,
             block: int = 1 << 20) -> float:
    """Bytes per second reading [off, off+length) at `depth` outstanding."""
    offsets = list(range(off, off + length, block))
    lock, it = threading.Lock(), iter(offsets)
    end = off + length

    def worker():
        while True:
            with lock:
                o = next(it, None)
            if o is None:
                return
            src.pread(o, min(block, end - o))

    started = time.perf_counter()
    if depth <= 1:
        worker()
    else:
        ts = [threading.Thread(target=worker) for _ in range(depth)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    return length / (time.perf_counter() - started)


# -- the codec -------------------------------------------------------------
def decode_rate(archive, *, sample: int = SAMPLE, threads=None,
                reps: int = 3) -> Codec:
    """How fast this machine decodes *this* archive, in plain bytes per second.

    Sweeps the number of decode threads rather than assuming one. The optimum
    is not a constant: it moves with block size, with whether the interpreter
    holds a lock between native calls, and with how much memory bandwidth is
    left over, and a build or a platform that changes any of those changes the
    answer. Payloads are fetched once and decoded from memory, so what is
    timed is the codec and not the disk.

    The median of `reps` rather than the best, because the best of several runs
    on a busy machine is the one where nothing else happened, and a loader has
    to live on a machine where things happen.
    """
    from .archive import Archive

    own = not isinstance(archive, Archive)
    arc = Archive(archive) if own else archive
    try:
        chunks = arc.chunks_for([(0, min(sample, arc.plain_bytes))])
        runs = arc.runs(chunks, 4 << 20)
        if not runs:
            return Codec(name="lmz-cpu", archive=arc.path, ratio=arc.ratio)
        payloads = {id(r): arc.fetch(r) for r in runs}
        plain = sum(r.plain for r in runs)
        out = bytearray(max(r.chunks[-1].dst + r.chunks[-1].rlen
                            for r in runs) - runs[0].chunks[0].dst)
        base = runs[0].chunks[0].dst
        cpus = os.cpu_count() or 4
        ladder = threads or sorted({1, 2, 4, min(8, cpus), min(16, cpus)})

        best, best_n = 0.0, 1
        for n in ladder:
            times = []
            for _ in range(reps):
                started = time.perf_counter()
                transport(runs, lambda r: payloads[id(r)],
                          lambda r, p: arc.place(r, p, out, base),
                          fetch_threads=1, place_threads=n,
                          inflight=max(4, n * 2))
                times.append(time.perf_counter() - started)
            rate = plain / statistics.median(times)
            if rate > best:
                best, best_n = rate, n
        return Codec(name="lmz-cpu", decode=best, threads=best_n,
                     archive=arc.path, ratio=arc.ratio, sample_bytes=plain)
    finally:
        if own:
            arc.close()


def encode_rate(plain_path: str, *, sample: int = SAMPLE,
                directory: str | None = None) -> tuple[float | None, float | None]:
    """(bytes per second, ratio) for compressing a sample of a plain file.

    Writing the sample out and compressing it is the only honest way to time
    what a checkpoint save will cost, because that is what a checkpoint save
    does. The coded output is written too, so this is encode plus a smaller
    write rather than encode alone -- which is the number the write plan wants
    anyway, and is noted here so it is not later quoted as the codec's own.
    """
    from ._lmz import lmz

    directory = directory or os.path.dirname(os.path.abspath(plain_path))
    size = os.path.getsize(plain_path)
    sample = min(sample, size)
    if sample < (16 << 20):
        return None, None
    tmp = tempfile.mkdtemp(dir=directory, prefix=".lmsluice-probe-")
    try:
        src = os.path.join(tmp, "sample.bin")
        with open(plain_path, "rb") as fh, open(src, "wb") as out:
            shutil.copyfileobj(fh, out, 8 << 20) if sample == size else None
            if sample != size:
                left = sample
                while left:
                    b = fh.read(min(8 << 20, left))
                    if not b:
                        break
                    out.write(b)
                    left -= len(b)
        dst = os.path.join(tmp, "sample.lmz")
        started = time.perf_counter()
        lmz().compress(src, dst)
        elapsed = time.perf_counter() - started
        coded = os.path.getsize(dst)
        return os.path.getsize(src) / elapsed, coded / os.path.getsize(src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- putting it together ---------------------------------------------------
def run(target: str, *, sample: int = SAMPLE, write: bool = True,
        profile: Profile | None = None) -> Profile:
    """Measure everything that a decision about `target` depends on."""
    profile = profile or Profile()
    profile.host = host_info()
    src = open_source(target)
    try:
        key = storage_key(target) if isinstance(src, FileSource) else src.name
        st = read_rate(src, sample=sample, key=key)
        if write and isinstance(src, FileSource):
            directory = os.path.dirname(os.path.abspath(target)) or "."
            st = Storage(**{**st.__dict__,
                            "write": write_rate(directory, sample=sample)})
        profile.storage[key] = st
    finally:
        src.close()

    if _is_archive(target):
        codec = decode_rate(target, sample=sample)
        profile.codecs[codec.name] = codec
    profile.measured_at = time.time()
    return profile


def _is_archive(target: str) -> bool:
    try:
        with open_source(target) as src:
            return src.pread(0, 4) == b"LMZ\x01"
    except Exception:                     # noqa: BLE001 -- not readable, not an archive
        return False
