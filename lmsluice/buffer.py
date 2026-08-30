"""Where decoded bytes land, and why that is a measurement and not a `bytearray`.

Every route through this package ends by writing plain bytes into a destination
the caller usually does not supply. For a 1.87 GB checkpoint that destination
cost **more than the transport that filled it**: `bytearray(n)` zero-fills, which
faults all 457,519 pages up front and writes a full pass of zeroes the decoder
then overwrites. Measured on this box, fresh process, median of seven: 0.55 s of
allocation against 0.31 s of transport. `MEASURED.md` has the fault counts.

A private anonymous mapping advised for huge pages does the same job in 0.11 s,
because 2 MiB pages take 1,407 faults where 4 KiB pages take 457,730. The bytes
are still zero -- the kernel may not hand a process another's data, so anonymous
pages arrive zeroed whether or not anyone memsets them. That is the point: the
zeroing was always free, and `bytearray` was paying for it twice.

**Three details are load-bearing, and two of them are silent when wrong.**

- `MAP_PRIVATE`. Python's `mmap.mmap(-1, n)` defaults to `MAP_SHARED`, and
  transparent huge pages do not back shared anonymous mappings. `MADV_HUGEPAGE`
  on one is accepted, returns success, and does nothing.
- The advice itself. With THP in `madvise` mode -- the common default -- an
  unadvised private mapping gets 4 KiB pages and lands *slower* than `bytearray`
  (0.55 s against 0.66 s), because deferring the faults does not remove them and
  taking them one at a time during the write costs more than taking them in a
  batch.
- Alignment is **not** among them. Over-mapping to a 2 MiB boundary was tried and
  changes nothing; the kernel aligns what it backs.

So the mapping is used only where it will actually be backed by huge pages, which
is a property of the machine and is read from the machine rather than assumed.
Where it will not be -- THP off, no `MADV_HUGEPAGE`, not Linux -- `bytearray` is
the better of the two and is what you get. On macOS the problem is smaller anyway
(16 KiB base pages take a quarter of the faults) and the remedy is absent; on
Windows large pages need `SeLockMemoryPrivilege`, which a library has no business
demanding.
"""

from __future__ import annotations

import mmap
import os

# Below this the two are indistinguishable and the mapping is not worth a
# syscall: measured on this box, parity to ~8 MB and 3.5-8x above 16 MB. Stated
# in huge pages rather than megabytes so it re-derives on a machine whose huge
# page is not 2 MiB -- arm64 alone ships 64 KiB, 2 MiB, 32 MiB and 512 MiB.
THRESHOLD_PAGES = 8


def huge_page_bytes() -> int:
    """The machine's huge page size, or 0 if it has none."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("Hugepagesize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _thp_mode() -> str | None:
    """Which THP mode the kernel is in: 'always', 'madvise', 'never', or None."""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as fh:
            text = fh.read()
    except OSError:
        return None
    for mode in ("always", "madvise", "never"):
        if f"[{mode}]" in text:
            return mode
    return None


_strategy: tuple[str, str, int] | None = None


def strategy() -> tuple[str, str, int]:
    """(how, why, threshold) -- decided once, from the machine, not from a flag.

    `how` is 'mapped' or 'bytes'. `why` is short enough to print in a report,
    because a user asking why a load is slow deserves to see this answer.
    """
    global _strategy
    if _strategy is not None:
        return _strategy
    need = ("MAP_PRIVATE", "MAP_ANONYMOUS", "MADV_HUGEPAGE")
    missing = [n for n in need if not hasattr(mmap, n)]
    if missing:
        _strategy = ("bytes", f"{os.name}/{','.join(missing)} absent: no huge "
                              f"pages to ask for", 0)
        return _strategy
    huge = huge_page_bytes()
    mode = _thp_mode()
    if not huge or mode is None:
        _strategy = ("bytes", "kernel reports no transparent huge pages", 0)
    elif mode == "never":
        _strategy = ("bytes", "transparent huge pages disabled "
                              "(/sys/.../transparent_hugepage/enabled = never)", 0)
    else:
        _strategy = ("mapped", f"{huge >> 20} MiB huge pages, THP={mode}",
                     THRESHOLD_PAGES * huge)
    return _strategy


def destination(n: int):
    """A writable buffer of `n` zero bytes, allocated the cheap way where there
    is one.

    Reads as zeroes either way, so this is a drop-in for `bytearray(n)` and not
    a change of contract -- a partial load still leaves the ranges nobody asked
    for reading as zero rather than as whatever was in memory before. That is
    why this is a mapping and not `numpy.empty`, which really would hand back
    the previous tenant's bytes.
    """
    if n <= 0:
        return bytearray(n)
    how, _why, floor = strategy()
    if how == "mapped" and n >= floor:
        try:
            buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            buf.madvise(mmap.MADV_HUGEPAGE)
            return buf
        except (OSError, ValueError):
            pass          # out of address space, or a kernel that refuses: fall back
    return bytearray(n)
