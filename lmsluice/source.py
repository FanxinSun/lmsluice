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
    """A local file. `pread` is thread-safe and needs no lock, which is why
    the fetch stage can run as many threads as the device has depth for."""

    def __init__(self, path: str):
        self.name = path
        self._fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self._fd).st_size

    def pread(self, offset: int, length: int) -> bytes:
        data = os.pread(self._fd, length, offset)
        if len(data) != length:
            raise EOFError(f"{self.name}: wanted {length} at {offset}, "
                           f"got {len(data)}")
        return data

    def fileno(self) -> int:
        return self._fd

    def uncache(self, offset: int = 0, length: int = 0) -> str:
        """Best effort: forget what the OS cached about this file.

        Returns how it was done, or why it was not. This is the only honest
        way to measure a cold read, and it is not available everywhere: Linux
        can drop a range from the page cache, macOS can ask that an fd bypass
        it, and Windows offers neither to a Python process. Where it is not
        available the caller must say the number is warm rather than quietly
        report it as cold.
        """
        fadvise = getattr(os, "posix_fadvise", None)
        if fadvise is not None:
            try:
                fadvise(self._fd, offset, length, os.POSIX_FADV_DONTNEED)
                return "posix_fadvise(DONTNEED)"
            except OSError as exc:
                return f"posix_fadvise failed: {exc}"
        try:                              # macOS: read past the cache instead
            import fcntl

            F_NOCACHE = 48
            fcntl.fcntl(self._fd, F_NOCACHE, 1)
            return "fcntl(F_NOCACHE)"
        except (ImportError, OSError, ValueError):
            pass
        return ""

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


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

    def __init__(self, url: str, *, timeout: float = 30.0, headers=None):
        self.name = url
        self.url = url
        self.timeout = timeout
        self.headers = dict(headers or {})
        self._local = threading.local()
        self.size, self.random_access = self._head()

    def _head(self) -> tuple[int, bool]:
        """(size, whether ranges work), settled by asking for one byte."""
        size = 0
        try:
            req = urllib.request.Request(self.url, method="HEAD",
                                         headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except Exception:                 # noqa: BLE001 -- HEAD is often refused
            pass
        try:
            req = urllib.request.Request(
                self.url, headers={**self.headers, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read(2)
                if r.status == 206 and len(body) == 1:
                    total = (r.headers.get("Content-Range") or "").rsplit("/", 1)
                    if total[-1].isdigit():
                        size = int(total[-1])
                    return size, True
                # 200 with the whole body: the server ignored the range.
                return (size or int(r.headers.get("Content-Length") or 0)), False
        except Exception:                 # noqa: BLE001
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
        headers = {**self.headers,
                   "Range": f"bytes={offset}-{offset + length - 1}",
                   "Accept-Encoding": "identity", "Connection": "keep-alive"}
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
