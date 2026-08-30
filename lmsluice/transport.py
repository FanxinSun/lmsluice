"""Two stages, overlapped, with memory bounded.

Moving a model off storage is two jobs with different appetites. Fetching
wants queue depth: an NVMe device reaches its rated speed only with a dozen
requests outstanding, and a network source needs more than that. Decoding
wants the opposite -- lmz's own measurement is that past two threads the
interpreter between native calls turns into GIL contention and a third core
subtracts.

A single pool cannot serve both. `lmz.decompress` reads and decodes inside
one worker, so the number of threads that decides decode throughput also
decides how many reads are outstanding, and the setting that is right for one
is wrong for the other. Splitting them is the whole of this module: F fetch
threads feeding P place threads across a bounded queue, sized independently.

The queue is what makes the stages overlap and what stops them running away.
Its depth is the memory bound -- at most `inflight` payloads exist at once --
and its fullness is the diagnosis: a queue that is always full means placing
is the bottleneck, always empty means fetching is, and `Report` records which.

Placement is unordered. Every job here writes to an offset it already knows,
so nothing downstream depends on arrival order, and imposing one would make
a slow job block the jobs behind it. A consumer that needs order -- a socket,
a pipe -- needs a reorder buffer this does not have.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


# How long a blocked thread waits before looking at the abort flag again.
# Short enough that a failure is noticed promptly, long enough that idling
# threads are not a measurable share of a load.
POLL = 0.05


@dataclass
class Report:
    """What the run cost, and which stage decided it."""

    jobs: int = 0
    fetched_bytes: int = 0
    placed_bytes: int = 0
    seconds: float = 0.0
    fetch_seconds: float = 0.0   # summed across fetch threads
    place_seconds: float = 0.0   # summed across place threads
    stalls_full: int = 0         # fetch waited: placing is behind
    stalls_empty: int = 0        # place waited: fetching is behind
    fetch_threads: int = 0
    place_threads: int = 0

    @property
    def rate(self) -> float:
        """Placed bytes per second -- the number a caller actually gets."""
        return self.placed_bytes / self.seconds if self.seconds else 0.0

    @property
    def limited_by(self) -> str:
        """Which stage the run waited on.

        Read off the stalls rather than the summed thread times, because the
        summed times are per-thread and the two stages have different thread
        counts, so they are not comparable without dividing -- and dividing
        assumes threads that were busy the whole run, which is the thing being
        measured.
        """
        if self.stalls_full > self.stalls_empty * 2:
            return "place"
        if self.stalls_empty > self.stalls_full * 2:
            return "fetch"
        return "balanced"


class _Abort:
    """First exception wins; the rest of the run unwinds quietly.

    A bare Event plus a list would race on which error is reported. This is
    small enough to be obviously right, and the reported error being the first
    one raised matters: the later ones are usually consequences.
    """

    __slots__ = ("_event", "_error", "_lock")

    def __init__(self):
        self._event = threading.Event()
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return self._event.is_set()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc
        self._event.set()

    def stop(self) -> None:
        self._event.set()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


def transport(jobs, fetch, place, *, fetch_threads: int = 8,
              place_threads: int = 2, inflight: int = 16,
              progress=None) -> Report:
    """Run `fetch(job)` on one pool and `place(job, payload)` on another.

    `fetch` should be I/O and `place` should be the work: both must release
    the GIL for the split to buy anything, which for lmz means a native decode
    and an `os.pread`, not a Python loop.

    Returns a `Report`. Both callables may return a byte count, or None if
    they would rather not be counted.
    """
    jobs = list(jobs)
    if not jobs:
        return Report(fetch_threads=fetch_threads, place_threads=place_threads)

    fetch_threads = max(1, min(fetch_threads, len(jobs)))
    place_threads = max(1, min(place_threads, len(jobs)))
    inflight = max(1, inflight)

    q: "queue.Queue" = queue.Queue(maxsize=inflight)
    abort = _Abort()
    rep = Report(jobs=len(jobs), fetch_threads=fetch_threads,
                 place_threads=place_threads)
    lock = threading.Lock()
    cursor = iter(range(len(jobs)))
    cursor_lock = threading.Lock()

    def next_job():
        with cursor_lock:
            return next(cursor, None)

    def fetcher():
        spent, stalls, got = 0.0, 0, 0
        try:
            while not abort:
                i = next_job()
                if i is None:
                    break
                job = jobs[i]
                t = time.perf_counter()
                payload = fetch(job)
                spent += time.perf_counter() - t
                got += _sizeof(payload)
                # Push against the bound rather than around it: if placing is
                # behind, the fetch stage must wait, or the memory ceiling is
                # a suggestion.
                while not abort:
                    try:
                        q.put((job, payload), timeout=POLL)
                        break
                    except queue.Full:
                        stalls += 1
        except BaseException as exc:      # noqa: BLE001 -- re-raised by the caller
            abort.fail(exc)
        finally:
            with lock:
                rep.fetch_seconds += spent
                rep.fetched_bytes += got
                rep.stalls_full += stalls

    def placer():
        spent, stalls, put = 0.0, 0, 0
        try:
            while True:
                try:
                    item = q.get(timeout=POLL)
                except queue.Empty:
                    if abort or done.is_set():
                        break
                    stalls += 1
                    continue
                if item is None:          # sentinel: no more work is coming
                    break
                job, payload = item
                if abort:
                    continue              # drain, so no fetcher blocks on put
                t = time.perf_counter()
                n = place(job, payload)
                spent += time.perf_counter() - t
                n = _sizeof(payload) if n is None else n
                put += n
                # Counted under the lock every job rather than once at the
                # end, so `progress` and a caller reading `rep` mid-run see a
                # total that is behind but never wrong.
                with lock:
                    rep.placed_bytes += n
                    total = rep.placed_bytes
                if progress is not None:
                    progress(total, None)
        except BaseException as exc:      # noqa: BLE001
            abort.fail(exc)
        finally:
            with lock:
                rep.place_seconds += spent
                rep.stalls_empty += stalls

    done = threading.Event()

    started = time.perf_counter()
    fetchers = [threading.Thread(target=fetcher, name=f"lmsluice-fetch-{i}",
                                 daemon=True) for i in range(fetch_threads)]
    placers = [threading.Thread(target=placer, name=f"lmsluice-place-{i}",
                                daemon=True) for i in range(place_threads)]
    for t in fetchers + placers:
        t.start()
    try:
        for t in fetchers:
            t.join()
        done.set()                        # every payload that will exist, exists
        for _ in placers:
            _offer(q, None, abort)        # one sentinel each, best effort
        for t in placers:
            t.join()
    finally:
        abort.stop()
        for t in fetchers + placers:
            t.join(timeout=5.0)
    rep.seconds = time.perf_counter() - started
    abort.raise_if_failed()
    return rep


def _offer(q, item, abort) -> None:
    """Put without blocking forever. The placers also stop on `done`, so a
    dropped sentinel costs one POLL of shutdown latency rather than a hang."""
    for _ in range(20):
        if abort:
            return
        try:
            q.put(item, timeout=POLL)
            return
        except queue.Full:
            continue


def _sizeof(obj) -> int:
    if obj is None:
        return 0
    try:
        return len(obj)
    except TypeError:
        return 0


def split(total: int, target: int, base: int = 0) -> list[tuple[int, int]]:
    """`total` bytes as (offset, length) spans of about `target` each.

    The last span absorbs the remainder rather than being left short, so a
    1.01x span is possible and a 0.01x one is not: a runt span costs a whole
    round of dispatch for almost no work.
    """
    if total <= 0:
        return []
    target = max(1, target)
    n = max(1, total // target)
    step = total // n
    out = [(base + i * step, step) for i in range(n)]
    off, length = out[-1]
    out[-1] = (off, total - (off - base))
    return out
