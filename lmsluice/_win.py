"""Reading a file on Windows without the cache getting in the way.

Linux and macOS can be told to forget what they cached about a file, so a cold
measurement there is: drop, then read normally. Windows offers a Python process
no equivalent -- and the first version of this package concluded from that that
cold reads were simply unavailable there, fell back to the warm number, and let
the router consume a page-cache rate as though it were a disk rate. On a machine
whose cache reads at 30 GB/s that puts the gate out by a factor of thirty, in
the direction that always recommends the plain file.

There is a way, and it is the other half of the same idea: rather than dropping
the cache after the fact, open a handle that never uses it. `FILE_FLAG_NO_BUFFERING`
does exactly that, and it is reachable from `ctypes` with nothing installed --
which is the constraint the whole probe is built under.

The price is that unbuffered I/O is strict. Offsets, lengths **and buffer
addresses** must all be multiples of the volume's sector size, which is a
property of the volume and not a constant to assume: 512 on older disks, 4096 on
most modern ones, and larger on some arrays. So the sector size is queried, the
buffer comes from `VirtualAlloc` (page-granular, which satisfies any sector size
in practice), and the sample is rounded down to whole sectors.

**This path is for measurement only.** Routing ordinary reads through it would
defeat the page cache for real loads, which is the opposite of what a loader
wants. `FileSource.pread` never touches it.

**Run on Windows**, not merely written there: Python 3.12 on NTFS, 512-byte
sectors, verified to return the file's real bytes at offset 0 and at 64 MiB and
to hold its rate across repeats rather than climbing the way a cached read
does. Every entry point still fails soft and returns None rather than raising,
so a volume or a build that refuses it degrades to "cold unavailable" instead
of crashing.
"""

from __future__ import annotations

import os

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
INVALID_HANDLE_VALUE = -1

MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


def available() -> bool:
    return os.name == "nt"


def sector_size(path: str) -> int | None:
    """Bytes per sector for the volume holding `path`.

    Queried rather than assumed. A 512-byte guess on a 4Kn volume makes every
    unbuffered read fail with ERROR_INVALID_PARAMETER, which is an unhelpful
    way to discover the number.
    """
    if not available():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        drive = os.path.splitdrive(os.path.abspath(path))[0]
        root = f"{drive}\\" if drive else None
        spc = wintypes.DWORD()
        bps = wintypes.DWORD()
        free = wintypes.DWORD()
        total = wintypes.DWORD()
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceW(
            ctypes.c_wchar_p(root), ctypes.byref(spc), ctypes.byref(bps),
            ctypes.byref(free), ctypes.byref(total))
        if not ok or not bps.value:
            return None
        return int(bps.value)
    except Exception:                     # noqa: BLE001 -- fail soft, see module docstring
        return None


class DirectReader:
    """An unbuffered handle on one file. Context manager; None-safe to skip."""

    def __init__(self, path: str):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._k32 = ctypes.windll.kernel32
        self._k32.CreateFileW.restype = wintypes.HANDLE
        self._k32.VirtualAlloc.restype = ctypes.c_void_p
        self.sector = sector_size(path) or 4096
        self._handle = self._k32.CreateFileW(
            ctypes.c_wchar_p(os.path.abspath(path)), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
        # INVALID_HANDLE_VALUE is -1, but the call comes back through a
        # `HANDLE` restype -- an unsigned pointer -- so the failure arrives as
        # 0xFFFFFFFFFFFFFFFF and `== -1` never matched. A missing file
        # therefore produced a reader whose every read failed with a confusing
        # error instead of the clean `None` the caller checks for. Compared as
        # signed, which is what the constant means.
        signed = ctypes.c_ssize_t(self._handle or 0).value
        if not self._handle or signed == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateFileW(NO_BUFFERING) failed on {path}: "
                          f"error {ctypes.GetLastError()}")
        self._buf = None
        self._cap = 0

    def _ensure(self, nbytes: int):
        """A sector-aligned buffer of at least `nbytes`.

        From VirtualAlloc rather than a bytearray: CPython gives no alignment
        guarantee for its buffers, and an unbuffered read into a misaligned
        address fails rather than being fixed up.
        """
        if self._cap >= nbytes and self._buf:
            return self._buf
        if self._buf:
            self._k32.VirtualFree(self._ct.c_void_p(self._buf), 0, MEM_RELEASE)
        self._buf = self._k32.VirtualAlloc(
            None, self._ct.c_size_t(nbytes), MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE)
        if not self._buf:
            raise OSError("VirtualAlloc failed")
        self._cap = nbytes
        return self._buf

    def read_at(self, offset: int, length: int, copy: bool = True) -> int:
        """Read `length` bytes at `offset`, returning how many arrived.

        `copy` controls whether the bytes are handed back to Python, and it
        defaults to True because leaving it out made the measurement
        incomparable. Every other rate in the profile -- the POSIX cold read,
        the warm read, the decode -- is measured through a call that allocates
        and fills a Python buffer, because that is what the fetch stage really
        does. An unbuffered read that discards into a reused aligned buffer
        skips that work and measured 5.69 GB/s against a 3.16 GB/s warm read on
        the same file, i.e. it reported the cache as slower than the disk.

        The number the router needs is what the pipeline can actually be fed,
        so the copy is part of it. `copy=False` remains available for measuring
        the device alone, which is a different question.
        """
        from ctypes import byref, c_void_p, wintypes

        s = self.sector
        offset -= offset % s
        length -= length % s
        if length <= 0:
            return 0
        buf = self._ensure(length)
        pos = wintypes.LARGE_INTEGER(offset)
        if not self._k32.SetFilePointerEx(
                wintypes.HANDLE(self._handle), pos, None, 0):
            raise OSError("SetFilePointerEx failed")
        got = wintypes.DWORD()
        if not self._k32.ReadFile(wintypes.HANDLE(self._handle),
                                  c_void_p(buf), wintypes.DWORD(length),
                                  byref(got), None):
            raise OSError(f"ReadFile failed: {self._ct.get_last_error()}")
        n = int(got.value)
        if copy and n:
            self.last = self._ct.string_at(buf, n)
        return n

    def close(self) -> None:
        if getattr(self, "_buf", None):
            self._k32.VirtualFree(self._ct.c_void_p(self._buf), 0, MEM_RELEASE)
            self._buf = None
        h = getattr(self, "_handle", None)
        if h and self._ct.c_ssize_t(h).value != INVALID_HANDLE_VALUE:
            from ctypes import wintypes

            self._k32.CloseHandle(wintypes.HANDLE(h))
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def open_direct(path: str) -> "DirectReader | None":
    """A `DirectReader`, or None where unbuffered reads are not available."""
    if not available():
        return None
    try:
        return DirectReader(path)
    except Exception:                     # noqa: BLE001 -- fail soft
        return None
