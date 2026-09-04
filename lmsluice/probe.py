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


# Fetch threads for a sealed archive, where the pool does more than block.
#
# Normally the fetch stage waits on the device, so depth is free. Decryption
# puts CPU work on those same threads, reached through ctypes -- and a short
# foreign call that releases and reacquires the GIL scales the wrong way: the
# threads spend their time handing the interpreter to each other. Measured
# here (9800X3D, 16 logical cores, OpenSSL 3.5.5 via libcrypto.so.3, 403 MB
# BF16 checkpoint, lmz-coded, warm page cache, best of 5): a sealed load runs
# at 3.61 GB/s on 1 fetch thread, 3.47 on 2, 3.25 on 4 and 2.90 on 16, against
# 3.69-3.87 unsealed at every depth. So the default of 16 was the worst
# setting available and cost 21%.
#
# It is a property of the mechanism and not of this CPU: the same sweep run
# with processes instead of threads scales 15 -> 99 GB/s across 8 workers, so
# the ceiling is the interpreter, not AES. Any machine reaching libcrypto this
# way has it. What the number does depend on is the work per foreign call, so
# it is expressed as a small depth rather than a rate, and a caller that has
# measured its own machine passes `fetch_threads` and overrides it.
SEALED_FETCH = 2


def default_threads(*, sealed: bool = False) -> tuple[int, int]:
    """(fetch, place) to start from, before anything has been measured.

    Fetch wants depth and place wants cores, and neither wants the machine's
    whole thread count: the fetch stage blocks on the device, so more of it
    costs nothing but does nothing either past the device's own depth, and the
    place stage is capped by whatever serialises between its native calls.
    These are starting points that the codec probe then replaces with a
    measurement.

    `sealed` says the fetch pool will also be decrypting, which turns it from a
    waiting stage into a computing one. See `SEALED_FETCH`.
    """
    cpus = os.cpu_count() or 4
    fetch = SEALED_FETCH if sealed else max(4, min(16, cpus))
    return fetch, max(1, min(8, cpus // 2))


def decrypt_rate(*, chunk: int = 4 << 20, seconds: float = 0.35,
                 threads: int | None = None) -> float | None:
    """Ciphertext bytes per second, at the depth a sealed archive will read at.

    Measured rather than assumed for the same reason every other rate here is:
    AES-NI is on every x86 worth shipping to and absent from plenty of the
    machines this project exists for, and the ceiling turns out not to be AES
    anyway -- reaching a library through ctypes puts the interpreter in the
    loop, so the number depends on the thread count and on the work per call.
    Both are therefore parameters, and the default thread count is the one a
    sealed load actually uses.

    Returns None when no backend is reachable, which prices the stage as
    unknown rather than as free.
    """
    from concurrent.futures import ThreadPoolExecutor

    from . import crypt

    if not crypt.available():
        return None
    threads = threads or SEALED_FETCH
    key = crypt.subkey(crypt.new_key(), crypt.new_salt())
    # One set per thread: sharing one measures a shared working set instead.
    sets = [[(j * 4 + i, crypt.seal(key, j * 4 + i, os.urandom(chunk)))
             for i in range(4)] for j in range(threads)]

    def run(blobs):
        done, i, deadline = 0, 0, time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            for _ in range(4):
                idx, blob = blobs[i % len(blobs)]
                crypt.open_chunk(key, idx, blob)
                done += len(blob)
                i += 1
        return done

    run(sets[0])                       # bind the symbols before timing
    started = time.perf_counter()
    # Positional: `max_workers` collides with lmz's word for a coded-stream
    # lane, and the boundary test that keeps that vocabulary out of this
    # tree cannot tell the two apart -- nor should it have to.
    with ThreadPoolExecutor(threads) as pool:
        total = sum(f.result() for f in [pool.submit(run, s) for s in sets])
    elapsed = time.perf_counter() - started
    return total / elapsed if elapsed > 0 else None


# -- storage ---------------------------------------------------------------
def read_rate(src: Source, *, sample: int = SAMPLE,
              depths=DEPTHS, key: str = "") -> Storage:
    """How fast this source delivers, cold if the platform allows it."""
    sample = max(1 << 20, min(sample, src.size))
    regions = _regions(src.size, sample, len(depths) + 2)

    # Two ways to see past a cache, and they are not interchangeable. Dropping
    # it (Linux, macOS) leaves ordinary reads to do the timing; bypassing it
    # (Windows, unbuffered handle) does the reading itself. Either gives a cold
    # number. Neither gives one when it fails, and a failure here must produce
    # `cold=None` rather than a warm read wearing a cold label.
    direct = None
    reader = getattr(src, "direct_reader", None)
    if reader is not None:
        direct = reader()
    if direct is not None:
        cold_ok, method = True, "unbuffered read (FILE_FLAG_NO_BUFFERING)"
    else:
        uncache = getattr(src, "uncache", None)
        cold_ok, method = uncache() if uncache else (False, "no method on this source")

    def timed(region, depth):
        off, length = region
        if direct is not None:
            started = time.perf_counter()
            got = direct.read_at(off, length)
            return got / (time.perf_counter() - started)
        if cold_ok:
            src.uncache()
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
    if cold_ok:
        layered = True if again > first * LAYERED_AT else None
        if layered:
            # The sweep above was measured against whatever is underneath, so
            # it is an upper bound on the cold rate rather than the cold rate.
            # The first read is the only one that was not, so it is what is
            # reported -- and the depth that won is still worth keeping,
            # because the right queue depth does not change with the cache.
            best = max(first, best if best_depth == 1 else first)
    # Warm, and it takes two steps rather than one. `uncache` on macOS is a
    # mode on the descriptor, not a one-shot eviction: it has to be cleared, and
    # then the region has to be read once *untimed* to actually populate the
    # cache, because every read up to here bypassed it and left it empty. On
    # Linux the clear is a no-op and the priming read is one extra read of an
    # already-cached region, so doing it unconditionally costs little and
    # removes a platform difference that would otherwise report a cold number
    # as warm on exactly one platform.
    restore = getattr(src, "recache", None)
    if restore is not None:
        restore()
    _read_at(src, *regions[0], 1)          # prime, untimed
    warm = _read_at(src, *regions[0], 1)
    if direct is not None:
        direct.close()

    return Storage(key=key or storage_key(src.name),
                   cold=best if cold_ok else None,
                   warm=warm, layered=layered, depth=best_depth,
                   method=method, sample_bytes=sample)


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


def pcie_rate(sample: int = 128 << 20, reps: int = 3) -> dict:
    """Host-to-device rate, pinned and pageable, in bytes per second.

    Both, because the difference is not a detail: a copy out of pageable memory
    is staged by the driver through its own pinned buffer, so it runs at
    roughly half the link and cannot overlap anything. A loader that reports
    the pageable number as "PCIe" has mismeasured the link by 2x; one that
    reports the pinned number while copying from pageable memory has
    mispredicted its own route by the same factor. The profile carries both and
    the plan is told which the transport actually uses.
    """
    from . import cuda

    ok, why = cuda.available()
    if not ok:
        return {"available": False, "why": why}
    ctx = cuda.Context()
    try:
        info = cuda.device_info()
        dev = ctx.allocate(sample)
        out = {"available": True, "device": info["name"],
               "sample_bytes": sample}
        with ctx.pinned(sample) as pin, ctx.stream() as stream:
            times = []
            for _ in range(reps):
                started = time.perf_counter()
                dev.copy_from(pin, stream=stream)
                stream.synchronize()
                times.append(time.perf_counter() - started)
            out["pinned"] = sample / statistics.median(times)
        pageable = bytearray(sample)
        times = []
        for _ in range(reps):
            started = time.perf_counter()
            dev.copy_from(pageable)
            times.append(time.perf_counter() - started)
        out["pageable"] = sample / statistics.median(times)
        dev.close()
        return out
    finally:
        ctx.close()


def encode_rate(plain_path: str, *, sample: int = SAMPLE,
                directory: str | None = None) -> tuple[float | None, float | None]:
    """(bytes per second, ratio) for compressing a sample of a plain file.

    Writing the sample out and compressing it is the only honest way to time
    what a checkpoint save will cost, because that is what a checkpoint save
    does. The coded output is written too, so this is encode plus a smaller
    write rather than encode alone -- which is the number the write plan wants
    anyway, and is noted here so it is not later quoted as the codec's own.
    """
    from .lmzcodec import encoder

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
        # No option is named here. Whether to compress at all is this
        # package's decision; every knob of *how* belongs to the codec, and
        # naming one would put an encoder-side constant in the transport
        # layer for the second time.
        encoder().encode(src, dst)
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

    # The second link. Measured unconditionally where a device exists, because
    # it is a property of the machine rather than of the file, and a plan for a
    # load into VRAM cannot be priced without it.
    pcie = pcie_rate()
    if pcie.get("available"):
        profile.storage["pcie"] = Storage(
            key="pcie", cold=pcie["pinned"], warm=pcie["pageable"],
            depth=1, method=f"cuMemcpyHtoDAsync to {pcie['device']}, pinned; "
                            f"the warm column is the pageable rate",
            sample_bytes=pcie["sample_bytes"])
    profile.measured_at = time.time()
    return profile


def _is_archive(target: str) -> bool:
    try:
        with open_source(target) as src:
            return src.pread(0, 4) == b"LMZ\x01"
    except Exception:                     # noqa: BLE001 -- not readable, not an archive
        return False
