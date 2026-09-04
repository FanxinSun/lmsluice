"""Where coded bytes come from, behind one interface.

The routing arithmetic never asks what the source *is*. It asks how fast the
source delivers, and compares that against how fast the machine decodes. A
local NVMe and a web server differ by two orders of magnitude in that number
and by nothing else that matters here, so they are two implementations of the
same three methods and the pipeline above them does not change.

That is not tidiness for its own sake. The case where compression is most
obviously worth it is the one where the source is slow -- a download, a
network mount, a phone's flash, a shared disk with something else on it --
and a loader that can only read local files cannot reach it. A model that
arrives over a link slower than the decoder arrives faster compressed, every
time, with no measurement needed to know the sign.

`pread` is the whole interface because the pipeline is built on it: many
independent reads at known offsets, issued concurrently, is what fills a queue
depth on a device and what fills a connection pool on a network. Sources that
cannot do that -- a pipe, a socket with no range support -- say so through
`random_access`, and the caller falls back to reading in order.
"""

from __future__ import annotations

import http.client
import os
import threading
import urllib.parse
import urllib.request


class Source:
    """A readable byte range, addressed absolutely."""

    size: int = 0
    random_access: bool = True
    name: str = ""

    def pread(self, offset: int, length: int) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def reader(self) -> "_SeekAdapter":
        """A seek/read/tell file object, for readers that want one."""
        return _SeekAdapter(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class FileSource(Source):
    """A local file, read concurrently at arbitrary offsets.

    `os.pread` is the whole implementation on POSIX: it takes the offset as an
    argument, so any number of threads can read any number of places at once
    with no lock and no shared cursor. That property is what lets the fetch
    stage run at the device's queue depth.

    **Windows has no `os.pread`**, and this package assumed it did -- so every
    read failed there with `AttributeError`, which is a more thorough kind of
    broken than the cold-measurement bug this was found next to. The fix has to
    keep the concurrency, so it is not a lock around seek-then-read: that would
    serialise the fetch stage and quietly turn queue depth into one. Instead
    each thread gets its own descriptor and therefore its own file position,
    which is the same independence `pread` gives, bought with a handle per
    thread instead of none.
    """

    _HAS_PREAD = hasattr(os, "pread")

    def __init__(self, path: str):
        self.name = path
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        self._flags = flags
        self._fd = os.open(path, flags)
        self.size = os.fstat(self._fd).st_size
        self._local = threading.local()
        self._extra: list[int] = []
        self._extra_lock = threading.Lock()
        self._primary_taken = False

    def _thread_fd(self) -> int:
        """This thread's own descriptor, opened on first use.

        The claim on the constructor's descriptor is made under the lock and
        exactly once. An earlier version tested `not self._extra` outside it,
        so several threads starting together all saw an empty list and all
        took the same descriptor -- which reintroduces the shared file position
        this exists to avoid, and only under concurrency, which is the only
        time it matters.
        """
        fd = getattr(self._local, "fd", None)
        if fd is not None:
            return fd
        with self._extra_lock:
            if not self._primary_taken:
                self._primary_taken = True
                fd = self._fd
            else:
                fd = os.open(self.name, self._flags)
                self._extra.append(fd)
        self._local.fd = fd
        return fd

    def pread(self, offset: int, length: int) -> bytes:
        if self._HAS_PREAD:
            data = os.pread(self._fd, length, offset)
        else:
            fd = self._thread_fd()
            os.lseek(fd, offset, os.SEEK_SET)
            chunks, left = [], length
            while left:
                b = os.read(fd, left)
                if not b:
                    break
                chunks.append(b)
                left -= len(b)
            data = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        if len(data) != length:
            raise EOFError(f"{self.name}: wanted {length} at {offset}, "
                           f"got {len(data)}")
        return data

    def fileno(self) -> int:
        return self._fd

    def pread_into(self, buf, offset: int) -> int:
        """Read len(buf) bytes at `offset` straight into a writable buffer.

        `pread` has to allocate and return `bytes`, which is read-only -- and a
        read-only buffer cannot have its address taken, so anything that wants
        to hand those bytes to a device driver has to copy them first. On a
        route whose whole argument is that it moves fewer bytes, an extra
        gigabyte through the interpreter is not a detail: it was half the cost
        of the device path.

        `preadv` writes into memory the caller owns and keeps the offset
        argument, so it stays lock-free like `pread`. Where it is absent the
        per-thread descriptor of the no-`pread` path serves the same purpose.
        """
        view = memoryview(buf).cast("B")
        preadv = getattr(os, "preadv", None)
        if preadv is not None:
            got, pos = 0, offset
            while got < len(view):
                n = preadv(self._fd, [view[got:]], pos)
                if not n:
                    break
                got += n
                pos += n
            return got
        fd = self._thread_fd()
        os.lseek(fd, offset, os.SEEK_SET)
        got = 0
        while got < len(view):
            n = os.readinto(fd, view[got:]) if hasattr(os, "readinto") else 0
            if not n:
                chunk = os.read(fd, len(view) - got)
                if not chunk:
                    break
                view[got:got + len(chunk)] = chunk
                n = len(chunk)
            got += n
        return got

    def uncache(self, offset: int = 0, length: int = 0) -> tuple[bool, str]:
        """Best effort: forget what the OS cached about this file.

        Returns `(succeeded, how)`. Both halves matter and the first one is
        why this is a tuple: the previous version returned a single string and
        put the failure reason in it, so a failure -- `"posix_fadvise failed:
        ..."` -- was a truthy value that every caller read as success. A cold
        rate was then recorded from a warm read, silently, on any filesystem
        where the call is refused. tmpfs is one, so this was never a
        Windows-only problem.

        Not available everywhere: Linux drops a range from the page cache,
        macOS asks that an fd bypass it, and Windows offers a Python process
        neither -- there, `direct_pread` reads past the cache instead. Where
        none of them works the caller must report the number as warm rather
        than pass it off as cold.
        """
        fadvise = getattr(os, "posix_fadvise", None)
        if fadvise is not None:
            try:
                fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
                return True, "posix_fadvise(DONTNEED)"
            except OSError as exc:
                return False, f"posix_fadvise refused: {exc}"
        try:                              # macOS: read past the cache instead
            import fcntl

            F_NOCACHE = 48
            fcntl.fcntl(self._fd, F_NOCACHE, 1)
            return True, "fcntl(F_NOCACHE)"
        except (ImportError, OSError, ValueError) as exc:
            return False, f"F_NOCACHE unavailable: {exc}"

    def recache(self) -> tuple[bool, str]:
        """Undo what `uncache` did, where it is a mode rather than an event.

        The two platforms differ in a way that matters and is easy to miss.
        Linux's `posix_fadvise(DONTNEED)` is an **event**: it drops what is
        cached and changes nothing about the descriptor, so there is nothing to
        restore. macOS's `F_NOCACHE` is a **mode**: it is set on the descriptor
        and every subsequent read on it bypasses the cache -- and, just as
        importantly, does not populate it.

        Two consequences follow, and both were live before this existed. A
        "warm" rate measured after `uncache` on macOS is not warm, because the
        reads that were supposed to warm the cache bypassed it. And a `Source`
        that was probed and then used to load would keep bypassing the cache
        for the rest of its life, which is the opposite of what a loader wants
        -- `pread` is meant to stay buffered, and only the probe needs to see
        past it.
        """
        if getattr(os, "posix_fadvise", None) is not None:
            return True, "nothing to restore: DONTNEED is an event, not a mode"
        try:
            import fcntl

            F_NOCACHE = 48
            fcntl.fcntl(self._fd, F_NOCACHE, 0)
            return True, "fcntl(F_NOCACHE, 0)"
        except (ImportError, OSError, ValueError) as exc:
            return False, f"could not clear F_NOCACHE: {exc}"

    def direct_reader(self):
        """An unbuffered reader for cold measurement, or None.

        Windows only, and measurement only. Everywhere else the cache is
        dropped instead, and on every platform ordinary `pread` stays
        buffered -- a loader wants the page cache, it is only the probe that
        has to see past it.
        """
        from . import _win

        return _win.open_direct(self.name)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        with self._extra_lock:
            for fd in self._extra:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._extra = []


class HttpSource(Source):
    """An archive on a web server, read with range requests.

    Concurrency here is what a download is missing: one connection to a
    typical server is far slower than the link, and eight are not. The fetch
    stage already issues independent reads, so the only thing this has to add
    is a connection per thread and the discipline not to share one, which HTTP
    keep-alive does not allow.

    Whether ranges work is settled by asking for one byte, never by reading
    `Accept-Ranges`. Servers that serve ranges perfectly well omit the header
    -- Python's own `SimpleHTTPRequestHandler` is one, and CDNs and proxies
    are careless about it in both directions -- so the header is evidence and
    a 206 is proof. It costs one small request at open time and it is the
    difference between using range requests and falling back to fetching the
    whole body for no reason.
    """

    def __init__(self, url: str, *, timeout: float = 30.0, headers=None,
                 signer=None, range_header: str = "Range"):
        self.name = url
        self.url = url
        self.timeout = timeout
        self.headers = dict(headers or {})
        # `signer(method, url, headers) -> headers` authorises one request.
        # A callable and not a credential, so this class never holds a secret
        # and an object store is the same code as a public URL plus a
        # signature. Azure names its range header differently, which is the
        # only other thing that varies across the three.
        self._signer = signer
        self.range_header = range_header
        self._local = threading.local()
        self.size, self.random_access = self._head()

    def _sign(self, method: str, extra: dict | None = None) -> dict:
        """Headers for one request, signed where a signer is in play.

        Signed per request rather than once at open: SigV4 and Shared Key both
        cover the range header, so every read has a different signature, and
        both carry a timestamp that a long-lived load would age out of.
        """
        headers = {**self.headers, **(extra or {})}
        if self._signer is None:
            return headers
        return self._signer(method, self.url, headers)

    def _head(self) -> tuple[int, bool]:
        """(size, whether ranges work), settled by asking for one byte.

        `head_error` records why the one-byte GET failed, when it did. A server
        that refuses HEAD is ordinary and says nothing; a server that cannot be
        reached at all, or that refuses the credentials, is not -- and without
        this it produced a source of size zero that failed much later with a
        message about the archive rather than about the connection.
        """
        size = 0
        self.head_error = None
        try:
            req = urllib.request.Request(self.url, method="HEAD",
                                         headers=self._sign("HEAD"))
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except Exception:                 # noqa: BLE001 -- HEAD is often refused
            pass
        try:
            req = urllib.request.Request(
                self.url,
                headers=self._sign("GET", {self.range_header: "bytes=0-0"}))
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read(2)
                if r.status == 206 and len(body) == 1:
                    total = (r.headers.get("Content-Range") or "").rsplit("/", 1)
                    if total[-1].isdigit():
                        size = int(total[-1])
                    return size, True
                # A zero-length object cannot return a byte because it has
                # none, and stores disagree about how to say so: some answer
                # 206 with an empty body, Azure answers 416. Both mean the
                # object exists and is empty, which is a size and not a
                # failure -- Spark leaves a zero-byte `_SUCCESS` marker in
                # every output directory it writes, so this is common.
                if size == 0 or (r.status == 206 and not body):
                    return 0, True
                # 200 with the whole body: the server ignored the range.
                return (size or int(r.headers.get("Content-Length") or 0)), False
        except Exception as exc:          # noqa: BLE001
            if getattr(exc, "code", None) == 416:
                return 0, True            # the other half of the empty case
            self.head_error = exc
            return size, False

    def _conn(self) -> http.client.HTTPConnection:
        """One kept-alive connection per thread.

        A new connection per request is a TCP handshake and, over TLS, a
        second round trip on top -- which on a real link costs more than the
        megabyte it was opened to carry. Reused, it costs nothing, and the
        fetch stage's threads become what a parallel download actually is.

        Per thread rather than pooled because an HTTP/1.1 connection carries
        one request at a time, so a shared one would serialise exactly the
        concurrency the fetch stage exists to provide.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            p = urllib.parse.urlparse(self.url)
            cls = (http.client.HTTPSConnection if p.scheme == "https"
                   else http.client.HTTPConnection)
            conn = cls(p.netloc, timeout=self.timeout)
            self._local.conn = conn
            self._local.path = urllib.parse.urlunparse(
                ("", "", p.path or "/", p.params, p.query, ""))
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def pread(self, offset: int, length: int) -> bytes:
        if not self.random_access:
            raise OSError(f"{self.url}: server does not serve byte ranges")
        headers = self._sign("GET", {
            self.range_header: f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity", "Connection": "keep-alive"})
        # One retry, because a kept-alive connection can be closed by the far
        # end between requests and that is not an error -- it is the normal
        # end of an idle keep-alive, and the fix is to dial again rather than
        # to fail a load over it.
        for attempt in (0, 1):
            try:
                conn = self._conn()
                conn.request("GET", self._local.path, headers=headers)
                r = conn.getresponse()
                status, data = r.status, r.read()
                break
            except (http.client.HTTPException, OSError):
                self._drop()
                if attempt:
                    raise
        else:                             # pragma: no cover -- loop always breaks
            raise OSError(f"{self.url}: could not read")
        if status != 206:
            self.random_access = False
            raise OSError(f"{self.url}: range ignored (HTTP {status})")
        if len(data) != length:
            raise EOFError(f"{self.url}: wanted {length} at {offset}, "
                           f"got {len(data)}")
        return data

    def close(self) -> None:
        self._drop()


class CachedSource(Source):
    """A source with no random access, made random by fetching it once.

    Plenty of servers ignore `Range` -- some CDNs, most dynamically generated
    responses, anything behind a transforming proxy -- and an archive cannot
    be opened without seeking, because its index is at the tail. The choice is
    to refuse those servers or to pay for the whole body once, and paying is
    strictly better than refusing: the file still arrives, and it arrives in
    the compressed form, which over a link slower than the decoder is still
    faster than fetching the plain one.

    What it costs is the overlap. Nothing decodes until the download finishes,
    so this is the serial route rather than the pipelined one, and a plan that
    prices it should say `overlapped=False`.
    """

    def __init__(self, inner: Source, *, directory: str | None = None):
        self._inner = inner
        self.name = inner.name
        self.size = inner.size
        self.random_access = True
        self._path = None
        self._file = None
        self._dir = directory
        self._lock = threading.Lock()

    def _materialise(self):
        if self._file is not None:
            return self._file
        with self._lock:
            if self._file is not None:
                return self._file
            import shutil
            import tempfile
            import urllib.request as _r

            fd, path = tempfile.mkstemp(dir=self._dir, prefix="lmsluice-cache-")
            with os.fdopen(fd, "wb") as out:
                req = _r.Request(self._inner.name,
                                 headers=getattr(self._inner, "headers", {}))
                with _r.urlopen(req, timeout=getattr(self._inner, "timeout", 30)) as r:
                    shutil.copyfileobj(r, out, 8 << 20)
            self._path = path
            self._file = FileSource(path)
            self.size = self._file.size
            return self._file

    def pread(self, offset: int, length: int) -> bytes:
        return self._materialise().pread(offset, length)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None
        self._inner.close()


def open_source(target: str, **kw) -> Source:
    """A `Source` for a path or a URL.

    A URL whose server will not serve ranges comes back wrapped, so that every
    caller above this line can assume random access and none of them has to
    carry a second code path for the servers that are awkward.
    """
    parsed = urllib.parse.urlparse(target)
    # Object stores first, and by scheme rather than by sniffing: `s3://` is
    # unambiguous where a hostname is not. They land back on `HttpSource` with
    # a signature, so everything below this line is the same code path a
    # public URL takes -- which is the whole claim `cloud.py` makes.
    if parsed.scheme in ("s3", "gs", "gcs", "az", "azure", "abfs", "abfss"):
        from .cloud import open_cloud

        return open_cloud(target, **kw)
    if parsed.scheme in ("http", "https"):
        src = HttpSource(target, **kw)
        return src if src.random_access else CachedSource(src)
    if parsed.scheme == "file":
        return FileSource(urllib.request.url2pathname(parsed.path))
    return FileSource(target)


class _SeekAdapter:
    """Enough of a file object for a reader that seeks, over any `Source`.

    lmz's `ArchiveReader` opens an archive with seek, read and tell -- the
    header at the front, the footer at the back, then the table and manifest
    it points at. That is four small reads, so it works over a network source
    unchanged and there is no reason to have a second reader for the remote
    case.
    """

    __slots__ = ("_src", "_pos")

    def __init__(self, src: Source):
        self._src = src
        self._pos = 0

    def seek(self, off: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = off
        elif whence == 1:
            self._pos += off
        elif whence == 2:
            self._pos = self._src.size + off
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self._src.size - self._pos
        n = max(0, min(n, self._src.size - self._pos))
        if not n:
            return b""
        data = self._src.pread(self._pos, n)
        self._pos += len(data)
        return data

    def fileno(self) -> int:
        fd = getattr(self._src, "_fd", None)
        if fd is None:
            raise OSError("source has no file descriptor")
        return fd

    def close(self) -> None:
        pass
