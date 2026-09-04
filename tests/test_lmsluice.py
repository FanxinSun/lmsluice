"""Tests for lmsluice. Self-contained: builds its own fixtures, needs no model.

Run with `python3 tests/test_lmsluice.py`. Anything that needs lmz is skipped
rather than failed when lmz is not importable, so the transport and the plan
-- which are the parts with no dependency -- are still covered on a bare
machine.

The HTTP tests run a real server on localhost. Localhost is not a network and
nothing here claims a rate from it; what it establishes is that an archive can
be opened and decoded over range requests at all, which is the property the
remote route rests on and the one that would break silently.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import random
import re as RE
import struct
import urllib.parse
import urllib.request
import datetime as DT
import zlib

from lmsluice import sign as SIGN
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmsluice as _lmsluice_pkg                                     # noqa: E402
from lmsluice import plan as P                                        # noqa: E402
from lmsluice.model import _windows, open_model                       # noqa: E402
from lmsluice.source import FileSource, HttpSource, open_source       # noqa: E402
from lmsluice.transport import Report, split, transport               # noqa: E402

try:
    from lmsluice._lmz import lmz

    LMZ = lmz()
except Exception:                         # noqa: BLE001
    LMZ = None

needs_lmz = unittest.skipIf(LMZ is None, "lmz is not importable")

# The stdlib codec needs Python 3.14's `compression.zstd`, or the `zstandard`
# package on an older one. That is a property of the interpreter rather than a
# fault, and `zstdcodec` says so itself -- so the tests that need it skip and
# the rest of the suite still reports on 3.10 through 3.13.
try:
    from lmsluice.zstdcodec import _zstd as _probe_zstd

    _probe_zstd()
    _ZSTD_WHY = ""
except Exception as _exc:                     # noqa: BLE001 -- absence is the point
    _ZSTD_WHY = str(_exc).split(".")[0]
needs_zstd = unittest.skipIf(_ZSTD_WHY, f"no zstd: {_ZSTD_WHY}")

# Some invariants are properties of the repository rather than of the package
# -- that `docs/boundary.md` exists and is linked, for one. They cannot hold
# when the suite runs against an installed wheel, which ships code and not
# documentation, so they say so instead of failing.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_repo = unittest.skipUnless(
    os.path.isdir(os.path.join(REPO, "docs")),
    "running against an installed package, not the repository")


# -- fixtures --------------------------------------------------------------
def bf16_bytes(rng, n: int) -> bytes:
    """Words that look like weights: a shared exponent band, random mantissa.

    Real BF16 weights have a narrow exponent distribution and a nearly random
    mantissa, which is the whole reason lmz beats a general compressor on
    them. A fixture of uniform noise would compress to nothing and would make
    every ratio in these tests meaningless.
    """
    out = bytearray()
    for i in range(n):
        exp = 0x3D + (i * 7 // max(1, n // 6)) % 4          # a few exponents
        hi = (exp << 1) | (rng.getrandbits(1))
        out += struct.pack("<H", ((hi & 0xFF) << 8) | rng.getrandbits(8))
    return bytes(out)


def write_safetensors(path: str, tensors: dict) -> str:
    header, blobs, off = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for b in blobs:
            fh.write(b)
    return path


def make_model(directory: str, seed: int = 7) -> str:
    rng = random.Random(seed)
    tensors = {}
    for i in range(12):
        rows = 64 + i * 16
        raw = bf16_bytes(rng, rows * 128)
        tensors[f"layer.{i}.weight"] = ("BF16", (rows, 128), raw)
    tensors["tiny.bias"] = ("BF16", (13,), bf16_bytes(rng, 13))
    return write_safetensors(os.path.join(directory, "model.safetensors"),
                             tensors)


# -- the transport ---------------------------------------------------------
class TestTransport(unittest.TestCase):
    def test_every_job_placed_exactly_once(self):
        seen = []
        lock = threading.Lock()

        def place(job, payload):
            with lock:
                seen.append(job)
            return len(payload)

        rep = transport(range(200), lambda j: bytes(64), place,
                        fetch_threads=6, place_threads=3, inflight=8)
        self.assertEqual(sorted(seen), list(range(200)))
        self.assertEqual(rep.placed_bytes, 200 * 64)
        self.assertEqual(rep.jobs, 200)

    def test_memory_stays_bounded(self):
        """The queue is the ceiling, so a slow placer must stall the fetcher.

        Without backpressure a fast source and a slow codec is exactly how a
        loader ends up holding the whole archive in RAM, which is the failure
        this bound exists to prevent and the one that would only show up on a
        model too big to fit.
        """
        live, peak, lock = [0], [0], threading.Lock()

        def fetch(j):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            return bytes(1024)

        def place(j, p):
            time.sleep(0.002)
            with lock:
                live[0] -= 1
            return len(p)

        transport(range(120), fetch, place, fetch_threads=8,
                  place_threads=1, inflight=4)
        self.assertLessEqual(peak[0], 4 + 8 + 1)   # queue, fetchers, placer

    def test_first_error_propagates_and_run_stops(self):
        def fetch(j):
            if j == 17:
                raise OSError("device gone")
            return bytes(8)

        with self.assertRaises(OSError):
            transport(range(400), fetch, lambda j, p: len(p),
                      fetch_threads=4, place_threads=2)

    def test_error_in_place_propagates(self):
        def place(j, p):
            if j == 5:
                raise ValueError("bad chunk")
            return len(p)

        with self.assertRaises(ValueError):
            transport(range(50), lambda j: bytes(8), place,
                      fetch_threads=2, place_threads=2)

    def test_empty_and_single(self):
        self.assertEqual(transport([], lambda j: b"", lambda j, p: 0).jobs, 0)
        rep = transport([1], lambda j: b"xy", lambda j, p: len(p),
                        fetch_threads=8, place_threads=8)
        self.assertEqual(rep.placed_bytes, 2)

    def test_split_leaves_no_runt(self):
        for total, target in ((100, 30), (10, 30), (1 << 20, 4096), (7, 1)):
            spans = split(total, target)
            self.assertEqual(sum(n for _, n in spans), total)
            self.assertEqual(spans[0][0], 0)
            for (a, n), (b, _) in zip(spans, spans[1:]):
                self.assertEqual(a + n, b)

    def test_report_names_the_bottleneck(self):
        rep = Report(stalls_full=100, stalls_empty=1)
        self.assertEqual(rep.limited_by, "place")
        self.assertEqual(Report(stalls_full=1, stalls_empty=100).limited_by,
                         "fetch")
        self.assertEqual(Report(stalls_full=10, stalls_empty=10).limited_by,
                         "balanced")


# -- the arithmetic --------------------------------------------------------
class TestPlan(unittest.TestCase):
    N, C = 1_000_000_000, 670_000_000

    def test_gate_is_the_only_thing_that_decides(self):
        """Which side of 1.0 the speedup lands on is set by codec vs link.

        Swept across the range of ratios to make the point that the ratio does
        not enter the gate: a 5%-saving archive and a 95%-saving one cross at
        the same place. What the ratio sets is how far past 1.0 it gets, which
        is why the 0.95 case is faster and still not worth switching to -- the
        margin, tested separately below, holds it back.
        """
        for ratio in (0.05, 0.5, 0.95):
            coded = int(self.N * ratio)
            fast = P.read_plan(self.N, coded, source=1e9, decode=10e9)
            slow = P.read_plan(self.N, coded, source=10e9, decode=1e9)
            self.assertGreater(fast.speedup, 1.0, ratio)
            self.assertLess(slow.speedup, 1.0, ratio)
            self.assertFalse(slow.coded, ratio)
        # Far enough past the gate, the decision follows the arithmetic.
        self.assertTrue(P.read_plan(self.N, self.C, source=1e9,
                                    decode=10e9).coded)

    def test_speedup_is_capped_by_the_ratio(self):
        d = P.read_plan(self.N, self.C, source=1e6, decode=1e12)
        self.assertAlmostEqual(d.speedup, self.N / self.C, places=3)

    def test_ratio_cannot_rescue_a_slow_codec(self):
        d = P.read_plan(self.N, self.N // 100, source=10e9, decode=1e9)
        self.assertFalse(d.coded)
        self.assertLess(d.speedup, 1.0)

    def test_ties_go_to_plain_and_say_so(self):
        d = P.read_plan(self.N, self.C, source=1e9, decode=1.02e9)
        self.assertFalse(d.coded)
        self.assertFalse(d.confident)
        self.assertIn("too close", d.why)

    def test_unmeasured_rate_makes_no_claim(self):
        d = P.read_plan(self.N, self.C, source=1e9, decode=None)
        self.assertFalse(d.confident)
        self.assertIsNone(d.speedup)
        self.assertEqual(d.chosen.name, "plain")

    def test_plain_target_has_no_coded_route(self):
        d = P.read_plan(self.N, None, source=1e9, decode=1e9)
        self.assertEqual(len(d.routes), 1)
        self.assertFalse(d.confident)

    def test_expand_is_worse_than_both(self):
        d = P.read_plan(self.N, self.C, source=1e9, decode=2e9,
                        expand_write=1e9)
        expand = [r for r in d.routes if r.name == "expand"][0]
        self.assertFalse(expand.overlapped)
        for other in d.routes:
            if other.name != "expand":
                self.assertGreater(expand.seconds, other.seconds)

    def test_write_gate_is_the_encoder(self):
        win = P.write_plan(self.N, 0.67, sink=0.5e9, encode=2e9)
        lose = P.write_plan(self.N, 0.67, sink=5e9, encode=2e9)
        self.assertTrue(win.coded)
        self.assertFalse(lose.coded)

    def test_gate_helper(self):
        self.assertAlmostEqual(P.gate_ratio(2e9, 1e9), 2.0)
        self.assertIsNone(P.gate_ratio(None, 1e9))
        self.assertIsNone(P.gate_ratio(1e9, 0))

    def test_agrees_with_the_standalone_gate(self):
        """`plan.py` and `gate.py` state one identity and must not drift.

        They are separate modules on purpose -- `gate.py` predicts a decode
        rate for a machine that is not here and must stay runnable with no
        dependencies so it can be copied onto one, while `plan.py` predicts
        nothing and prices what was measured. The agreement is pinned here
        rather than removed by merging them.
        """
        from lmsluice import gate

        N = 1 << 30
        for f in (0.05, 0.3, 0.653, 0.9, 0.99):
            for link in (0.3e9, 2e9, 6.34e9, 28.8e9):
                for decode in (0.5e9, 2e9, 111e9, 418e9):
                    d = P.read_plan(N, int(N * f), source=link, decode=decode)
                    self.assertAlmostEqual(
                        d.speedup, gate.speedup(decode / 1e9, link / 1e9, f),
                        places=6, msg=f"f={f} link={link} decode={decode}")


# -- sources ---------------------------------------------------------------
class _Quiet(http.server.SimpleHTTPRequestHandler):
    """A static server that honours Range, which the stdlib one does not.

    Written out rather than worked around because the remote route rests on
    range requests, and a test that only ever exercises whole-body GETs would
    pass with the range path broken.
    """

    ranges = True

    def log_message(self, *_a):
        pass

    def copyfile(self, source, outputfile):
        # On the no-Range path the whole body is written while the client,
        # having taken the bytes it asked for, is already hanging up. Only
        # that hang-up is swallowed: every other write error still reaches
        # handle_error, so a genuine server-side failure is still visible.
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        rng = self.headers.get("Range") if self.ranges else None
        if not rng:
            return super().do_GET()
        path = self.translate_path(self.path)
        try:
            size = os.path.getsize(path)
        except OSError:
            return super().do_GET()
        lo, _, hi = rng.partition("=")[2].partition("-")
        lo = int(lo)
        hi = int(hi) if hi else size - 1
        hi = min(hi, size - 1)
        with open(path, "rb") as fh:
            fh.seek(lo)
            body = fh.read(hi - lo + 1)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)


class TestSources(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "blob.bin")
        self.data = os.urandom(300_000)
        with open(self.path, "wb") as fh:
            fh.write(self.data)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_file_source_reads_and_refuses_short(self):
        with open_source(self.path) as s:
            self.assertEqual(s.size, len(self.data))
            self.assertEqual(s.pread(1000, 64), self.data[1000:1064])
            with self.assertRaises(EOFError):
                s.pread(s.size - 4, 64)

    def test_the_no_pread_path_reads_the_same_bytes(self):
        """Windows has no `os.pread`, so `FileSource` falls back to a
        per-thread descriptor. Exercised here, on a platform that does have
        `pread`, because otherwise the fallback is only ever run on the
        machine nobody tests on -- which is how it came to be missing
        entirely.
        """
        from lmsluice.source import FileSource

        class NoPread(FileSource):
            _HAS_PREAD = False

        with NoPread(self.path) as s:
            self.assertEqual(s.pread(1000, 64), self.data[1000:1064])
            self.assertEqual(s.pread(0, len(self.data)), self.data)
            with self.assertRaises(EOFError):
                s.pread(s.size - 4, 64)

    def test_the_no_pread_path_stays_concurrent(self):
        """Threads must not share a file position. A lock around seek-then-read
        would pass the test above and silently reduce the fetch stage's queue
        depth to one, so what is checked is that concurrent reads at different
        offsets each get their own bytes."""
        import concurrent.futures as cf

        from lmsluice.source import FileSource

        class NoPread(FileSource):
            _HAS_PREAD = False

        with NoPread(self.path) as s:
            offsets = list(range(0, 200_000, 1000))  # noqa: E501
            with cf.ThreadPoolExecutor(8) as pool:
                got = list(pool.map(lambda o: s.pread(o, 512), offsets))
            for o, b in zip(offsets, got):
                self.assertEqual(b, self.data[o:o + 512], f"at {o}")

    def test_seek_adapter_matches_a_real_file(self):
        with open_source(self.path) as s, open(self.path, "rb") as fh:
            a, b = s.reader(), fh
            for whence, off in ((0, 10), (1, 20), (2, -100), (0, 0)):
                self.assertEqual(a.seek(off, whence), b.seek(off, whence))
                self.assertEqual(a.read(50), b.read(50))
            self.assertEqual(a.tell(), b.tell())
            self.assertEqual(a.read(-1), b.read(-1))

    def test_http_source_serves_ranges(self):
        with _serve(self.dir) as base:
            with HttpSource(f"{base}/blob.bin") as s:
                self.assertTrue(s.random_access)
                self.assertEqual(s.size, len(self.data))
                self.assertEqual(s.pread(4096, 128), self.data[4096:4224])


class _serve:
    """A local static server, for the duration of a with-block."""

    def __init__(self, directory: str, ranges: bool = True):
        self.directory = directory
        self.ranges = ranges

    def __enter__(self) -> str:
        handler = type("H", (_Quiet,),
                       {"ranges": self.ranges,
                        "__init__": lambda s, *a, **k: _Quiet.__init__(
                            s, *a, directory=self.directory, **k)})
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __exit__(self, *_exc):
        self.srv.shutdown()
        self.srv.server_close()


class _fake_store:
    """A local object store that checks the signature, for a with-block.

    Enough of each protocol to prove the client, and no more: it re-derives the
    signature from the request it actually received and compares, so a client
    that signs the wrong canonical form fails here rather than against a real
    bucket. It answers `HEAD`, serves `Range` and `x-ms-range`, and rejects an
    unsigned request when a key is configured.

    Serving the bytes is the easy half and would pass with no signing at all.
    Re-deriving the signature is the half that makes this a test.
    """

    def __init__(self, directory, cloud="s3", key=None, account="acct",
                 require_auth=True):
        self.directory = directory
        self.cloud, self.key, self.account = cloud, key, account
        self.require_auth = require_auth
        self.seen = []
        self.uploads = {}
        self.blocks = {}
        self.fail_part = None

    def __enter__(self):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _blob(self):
                name = os.path.basename(urllib.parse.urlparse(self.path).path)
                full = os.path.join(outer.directory, name)
                if not os.path.exists(full):
                    return None
                with open(full, "rb") as fh:
                    return fh.read()

            def _authorised(self, body=b""):
                got = self.headers.get("Authorization")
                outer.seen.append({k.lower(): v for k, v in self.headers.items()})
                if not outer.require_auth:
                    return True
                if not got:
                    return False
                url = f"http://{self.headers.get('Host')}{self.path}"
                sent = {k: v for k, v in self.headers.items()
                        if k.lower() != "authorization"}
                if outer.cloud == "azure":
                    # Every header the client sent, not only `x-ms-*`:
                    # Content-Length is one of Azure's twelve positional lines,
                    # so filtering it out here re-derives a different string
                    # and rejects a correct request.
                    want = SIGN.shared_key_header(
                        method=self.command, url=url, account=outer.account,
                        key=outer.key, headers=sent)
                else:
                    stamp = self.headers.get("x-amz-date")
                    when = DT.datetime.strptime(
                        stamp, "%Y%m%dT%H%M%SZ").replace(
                            tzinfo=DT.timezone.utc)
                    signed = got.split("SignedHeaders=")[1].split(",")[0]
                    # Region and service come out of the Credential scope the
                    # client sent, not from a constant here: GCS signs with
                    # region "auto" and hardcoding "us-east-1" would have made
                    # this fake reject a correct request.
                    scope = got.split("Credential=")[1].split(",")[0].split("/")
                    want = SIGN.sigv4_headers(
                        method=self.command, url=url, region=scope[2],
                        service=scope[3], access_key=scope[0],
                        secret_key=outer.key,
                        headers={k: v for k, v in sent.items()
                                 if k.lower() in signed.split(";")
                                 and not k.lower().startswith(("x-amz-", "host"))},
                        payload_sha256=self.headers.get(
                            "x-amz-content-sha256") or SIGN.EMPTY_SHA256,
                        when=when)
                    # The client must hash the body it actually sent. Checking
                    # it here is what makes a write test a test: a signature
                    # over the empty hash would otherwise pass on every upload.
                    if body:
                        want_sha = hashlib.sha256(body).hexdigest()
                        if self.headers.get("x-amz-content-sha256") != want_sha:
                            return False
                return want["Authorization"] == got

            def _deny(self):
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_HEAD(self):
                body = self._blob()
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not self._authorised():
                    return self._deny()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _reply(self, code, body=b"", extra=None):
                self.send_response(code)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_DELETE(self):
                body = self._body()
                if not self._authorised(body):
                    return self._deny()
                q = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query)
                outer.uploads.pop(q.get("uploadId", [""])[0], None)
                self._reply(204)

            def do_POST(self):
                body = self._body()
                if not self._authorised(body):
                    return self._deny()
                p = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(p.query, keep_blank_values=True)
                if "uploads" in q:
                    uid = f"u{len(outer.uploads) + 1}"
                    outer.uploads[uid] = {}
                    return self._reply(200, (
                        f"<InitiateMultipartUploadResult><Bucket>b</Bucket>"
                        f"<Key>k</Key><UploadId>{uid}</UploadId>"
                        f"</InitiateMultipartUploadResult>").encode())
                uid = q.get("uploadId", [""])[0]
                parts = outer.uploads.pop(uid, None)
                if parts is None:
                    return self._reply(404)
                # Assemble in the order the manifest names, not the order the
                # parts arrived -- which is what a concurrent upload tests.
                order = [int(x) for x in RE.findall(
                    r"<PartNumber>(\d+)</PartNumber>",
                    body.decode("utf-8", "replace"))]
                blob = b"".join(parts[n] for n in order)
                with open(os.path.join(outer.directory,
                                       os.path.basename(p.path)), "wb") as fh:
                    fh.write(blob)
                self._reply(200, b"<CompleteMultipartUploadResult/>")

            def do_PUT(self):
                body = self._body()
                if not self._authorised(body):
                    return self._deny()
                p = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(p.query, keep_blank_values=True)
                name = os.path.join(outer.directory, os.path.basename(p.path))
                if "partNumber" in q:
                    if outer.fail_part == int(q["partNumber"][0]):
                        return self._reply(500, b"<Error>no space</Error>")
                    uid = q["uploadId"][0]
                    outer.uploads.setdefault(uid, {})[
                        int(q["partNumber"][0])] = body
                    return self._reply(
                        200, extra={"ETag": f'"{hashlib.md5(body).hexdigest()}"'})
                if q.get("comp", [""])[0] == "block":
                    outer.blocks.setdefault(name, {})[q["blockid"][0]] = body
                    return self._reply(201)
                if q.get("comp", [""])[0] == "blocklist":
                    ids = RE.findall(r"<Latest>([^<]+)</Latest>",
                                     body.decode("utf-8", "replace"))
                    with open(name, "wb") as fh:
                        fh.write(b"".join(outer.blocks[name][i] for i in ids))
                    return self._reply(201)
                with open(name, "wb") as fh:
                    fh.write(body)
                self._reply(201)

            def do_GET(self):
                body = self._blob()
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not self._authorised():
                    return self._deny()
                rng = (self.headers.get("Range")
                       or self.headers.get("x-ms-range"))
                if not rng:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                lo, _, hi = rng.split("=", 1)[1].partition("-")
                lo = int(lo)
                hi = int(hi) if hi else len(body) - 1
                cut = body[lo:hi + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(cut)))
                self.send_header("Content-Range",
                                 f"bytes {lo}-{lo + len(cut) - 1}/{len(body)}")
                self.end_headers()
                self.wfile.write(cut)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __exit__(self, *_exc):
        self.srv.shutdown()
        self.srv.server_close()


# -- the loader, both routes ----------------------------------------------
@needs_lmz
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.plain = make_model(cls.dir)
        with open(cls.plain, "rb") as fh:
            cls.ref = fh.read()
        cls.archive = os.path.join(cls.dir, "model.lmz")
        LMZ.compress(cls.plain, cls.archive)
        cls.mapped = os.path.join(cls.dir, "model.mapped.lmz")
        LMZ.compress(cls.plain, cls.mapped, mapped=True)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.dir, ignore_errors=True)

    def targets(self):
        return (self.plain, self.archive, self.mapped)

    def test_whole_model_is_byte_identical_on_every_route(self):
        for t in self.targets():
            with open_model(t) as m:
                self.assertEqual(bytes(m.load()), self.ref, t)

    def test_tensor_index_agrees_between_routes(self):
        with open_model(self.plain) as a, open_model(self.archive) as b:
            self.assertEqual(sorted(a.tensors), sorted(b.tensors))
            for name in a.tensors:
                x, y = a.tensors[name], b.tensors[name]
                self.assertEqual((x.start, x.end, x.dtype, x.shape),
                                 (y.start, y.end, y.dtype, y.shape), name)

    def test_one_tensor_at_a_time(self):
        for t in self.targets():
            with open_model(t) as m:
                for name, info in m.tensors.items():
                    self.assertEqual(bytes(m.tensor(name)),
                                     self.ref[info.start:info.end],
                                     f"{t} {name}")

    def test_partial_load_touches_only_what_was_asked_for(self):
        names = ["layer.3.weight", "layer.9.weight", "tiny.bias"]
        for t in self.targets():
            with open_model(t) as m:
                buf = m.load(names=names)
                for name in names:
                    info = m.tensors[name]
                    self.assertEqual(bytes(m.tensor(name, buffer=buf)),
                                     self.ref[info.start:info.end], name)

    def test_stream_yields_every_tensor_under_a_budget(self):
        for t in self.targets():
            with open_model(t) as m:
                got = {}
                for name, view in m.stream(budget=8192):
                    got[name] = bytes(view)
                self.assertEqual(sorted(got), sorted(m.tensors), t)
                for name, raw in got.items():
                    info = m.tensors[name]
                    self.assertEqual(raw, self.ref[info.start:info.end], name)

    def test_a_tensor_larger_than_the_budget_still_arrives(self):
        with open_model(self.archive) as m:
            names = [n for n, v in
                     sorted(m.tensors.items(), key=lambda kv: -kv[1].nbytes)][:1]
            got = dict(m.stream(names=names, budget=16))
            info = m.tensors[names[0]]
            self.assertEqual(bytes(got[names[0]]),
                             self.ref[info.start:info.end])

    def test_expanding_reproduces_the_file(self):
        out = os.path.join(self.dir, "expanded.safetensors")
        with open_model(self.archive) as m:
            m.to_file(out, overwrite=True)
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(), self.ref)

    def test_map_is_refused_on_a_coded_target(self):
        from lmsluice.model import NotMappable

        with open_model(self.archive) as m:
            with self.assertRaises(NotMappable):
                m.map()
        with open_model(self.plain) as m:
            self.assertEqual(bytes(m.map()), self.ref)

    def test_route_is_chosen_from_the_file_not_the_name(self):
        odd = os.path.join(self.dir, "not-obviously-an-archive.bin")
        with open(self.archive, "rb") as a, open(odd, "wb") as b:
            b.write(a.read())
        with open_model(odd) as m:
            self.assertEqual(m.route, "coded")
            self.assertEqual(bytes(m.load()), self.ref)

    def test_over_http(self):
        with _serve(self.dir) as base:
            for name in ("model.safetensors", "model.lmz"):
                with open_model(f"{base}/{name}") as m:
                    self.assertEqual(bytes(m.load()), self.ref, name)

    def test_over_http_without_range_support(self):
        """A server that ignores Range still delivers, by being fetched once."""
        from lmsluice.source import CachedSource

        with _serve(self.dir, ranges=False) as base:
            with open_model(f"{base}/model.lmz") as m:
                self.assertIsInstance(m._arc.source, CachedSource)
                self.assertEqual(bytes(m.load()), self.ref)

    def test_windows_respect_the_budget_where_they_can(self):
        with open_model(self.plain) as m:
            order = sorted(m.tensors, key=lambda n: m.tensors[n].start)
            for group in _windows(order, m.tensors, 8192):
                lo = m.tensors[group[0]].start
                hi = max(m.tensors[n].end for n in group)
                self.assertTrue(len(group) == 1 or hi - lo <= 8192)


class TestColdRateHonesty(unittest.TestCase):
    """A rate that is not cold must never be used as if it were.

    Three defects with one theme, all found after the router was already
    shipping. Each of these tests fails against the code as it was.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "blob.bin")
        with open(self.path, "wb") as fh:
            fh.write(os.urandom(4 << 20))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_refused_cache_drop_is_not_a_cold_measurement(self):
        """The bug: the failure string was truthy, so failure read as success.

        posix_fadvise is refused on some filesystems -- tmpfs among them -- so
        this fired on Linux too, and silently: the probe recorded a warm read
        under the name `cold`.
        """
        from lmsluice import probe
        from lmsluice.source import FileSource

        class Refuses(FileSource):
            def uncache(self, offset=0, length=0):
                return False, "posix_fadvise refused: [Errno 22] Invalid argument"

        with Refuses(self.path) as src:
            st = probe.read_rate(src, sample=1 << 20, depths=(1,))
        self.assertIsNone(st.cold)
        self.assertIsNotNone(st.warm)
        self.assertIn("refused", st.method)

    def test_source_never_returns_the_warm_rate(self):
        """The bug: `source` fell back to `warm`, a page-cache rate.

        On a platform that cannot drop its cache that made the gate come out
        about thirty times too small, so the router recommended the plain file
        on every such machine -- including the ones where the coded route wins.
        """
        from lmsluice.rates import Storage

        warm_only = Storage(key="k", cold=None, warm=40e9)
        self.assertIsNone(warm_only.source)
        self.assertEqual(warm_only.warm_bound, 40e9)

        measured = Storage(key="k", cold=2.4e9, warm=40e9)
        self.assertEqual(measured.source, 2.4e9)
        self.assertIsNone(measured.warm_bound,
                          "warm_bound is for when cold is missing, not always")

    def test_layering_is_not_claimed_when_the_drop_failed(self):
        """Layering is detected by two cold reads. Without a cold read there
        is no detection to report, and `None` already means 'not ruled out'."""
        from lmsluice import probe
        from lmsluice.source import FileSource

        class Refuses(FileSource):
            def uncache(self, offset=0, length=0):
                return False, "nope"

        with Refuses(self.path) as src:
            st = probe.read_rate(src, sample=1 << 20, depths=(1,))
        self.assertIsNone(st.layered)


@needs_lmz
class TestProbe(unittest.TestCase):
    def test_probe_measures_and_round_trips_through_json(self):
        from lmsluice import probe
        from lmsluice.rates import Profile

        d = tempfile.mkdtemp()
        try:
            plain = make_model(d)
            arc = os.path.join(d, "m.lmz")
            LMZ.compress(plain, arc)
            p = probe.run(arc, sample=1 << 20)
            self.assertTrue(p.storage)
            self.assertTrue(p.codecs)
            codec = p.codecs["lmz-cpu"]
            self.assertGreater(codec.decode, 0)
            self.assertGreater(codec.ratio, 0)
            path = p.save(os.path.join(d, "profile.json"))
            with open(path) as fh:
                back = Profile.from_json(json.load(fh))
            self.assertEqual(back.codecs["lmz-cpu"].decode, codec.decode)
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)

    def test_a_profile_from_another_machine_is_not_used(self):
        from lmsluice.rates import Profile

        d = tempfile.mkdtemp()
        try:
            p = Profile(host={"fingerprint": "SomeOtherBox|x86_64|elsewhere|4"},
                        measured_at=time.time())
            path = p.save(os.path.join(d, "profile.json"))
            self.assertIsNone(Profile.load(path))
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)


try:
    from lmsluice import cuda as _cuda

    _CUDA_OK, _CUDA_WHY = _cuda.available()
except Exception as exc:                  # noqa: BLE001
    _CUDA_OK, _CUDA_WHY = False, str(exc)

needs_cuda = unittest.skipUnless(_CUDA_OK, f"no CUDA device: {_CUDA_WHY}")


class TestVramPlan(unittest.TestCase):
    """The three-route arithmetic. No device needed -- rates are given."""

    N, C = 1_000_000_000, 671_000_000

    def test_device_decode_shortens_both_links(self):
        """Its whole claim: fewer bytes over storage AND over PCIe."""
        d = P.vram_plan(self.N, self.C, source=2.3e9, pcie=28.8e9,
                        host_decode=1.05e9, device_decode=111e9)
        routes = {r.name: r for r in d.routes}
        self.assertLess(routes["device-decode"].device_bytes,
                        routes["host-decode"].device_bytes)
        self.assertLess(routes["host-decode"].device_bytes,
                        routes["plain"].device_bytes)
        self.assertEqual(d.chosen.name, "device-decode")

    def test_host_decode_loses_to_a_disk_faster_than_the_cpu(self):
        d = P.vram_plan(self.N, self.C, source=6.3e9, pcie=28.8e9,
                        host_decode=1.05e9, device_decode=None)
        self.assertEqual(d.chosen.name, "plain")

    def test_host_decode_wins_on_a_slow_disk_with_no_gpu_decoder(self):
        d = P.vram_plan(self.N, self.C, source=0.55e9, pcie=28.8e9,
                        host_decode=1.05e9, device_decode=None)
        self.assertEqual(d.chosen.name, "host-decode")

    def test_unmeasured_pcie_makes_no_claim(self):
        d = P.vram_plan(self.N, self.C, source=2.3e9, pcie=None,
                        host_decode=1.05e9, device_decode=111e9)
        self.assertFalse(d.confident)
        self.assertEqual(d.chosen.name, "plain")

    def test_ties_go_to_the_route_that_moves_less(self):
        """Both read-limited at 1/f, so they finish together; the one that puts
        less on PCIe is still the better choice."""
        d = P.vram_plan(self.N, self.C, source=0.1e9, pcie=28.8e9,
                        host_decode=50e9, device_decode=50e9)
        self.assertEqual(d.chosen.name, "device-decode")


@needs_cuda
class TestCuda(unittest.TestCase):
    """The driver binding. Skipped wholesale on a machine with no device."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = _cuda.Context()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.close()

    def test_device_info(self):
        info = _cuda.device_info()
        self.assertTrue(info["name"])
        self.assertGreater(info["total_bytes"], 0)

    def test_roundtrip_through_vram(self):
        for src in (bytes(range(256)) * 512,            # read-only
                    bytearray(os.urandom(64 << 10))):   # writable
            with self.ctx.allocate(len(src)) as buf:
                buf.copy_from(src)
                self.assertEqual(bytes(buf.to_host()), bytes(src))

    def test_pinned_async_copy(self):
        data = os.urandom(1 << 20)
        with self.ctx.pinned(1 << 20) as pin, self.ctx.stream() as stream:
            pin.view()[:] = data
            with self.ctx.allocate(1 << 20) as buf:
                buf.copy_from(pin, stream=stream)
                stream.synchronize()
                self.assertEqual(bytes(buf.to_host()), data)

    def test_cuda_array_interface_is_well_formed(self):
        """The contract a tensor library reads. Wrong here means torch adopts
        the wrong bytes silently, so it is checked field by field."""
        with self.ctx.allocate(4096) as buf:
            i = buf.__cuda_array_interface__
            self.assertEqual(i["shape"], (4096,))
            self.assertEqual(i["typestr"], "|u1")
            self.assertEqual(i["version"], 3)
            self.assertIsNone(i["strides"])
            ptr, readonly = i["data"]
            self.assertGreater(ptr, 0)
            self.assertFalse(readonly)

    def test_a_copy_past_the_end_is_refused(self):
        with self.ctx.allocate(1024) as buf:
            with self.assertRaises(ValueError):
                buf.copy_from(bytes(2048))


@needs_cuda
@needs_lmz
class TestToDevice(unittest.TestCase):
    """The path itself, end to end, byte for byte."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.plain = make_model(cls.dir)
        with open(cls.plain, "rb") as fh:
            cls.ref = fh.read()
        cls.archive = os.path.join(cls.dir, "model.lmz")
        LMZ.compress(cls.plain, cls.archive)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_both_routes_land_identical_bytes_in_vram(self):
        for target in (self.plain, self.archive):
            with open_model(target) as m:
                dev = m.to_device()
                try:
                    self.assertEqual(bytes(dev.to_host()), self.ref, target)
                finally:
                    dev.close()

    def test_a_window_smaller_than_the_model_still_lands_whole(self):
        """Staging is a window, so the seams are where this breaks."""
        with open_model(self.archive) as m:
            dev = m.to_device(window=4096)
            try:
                self.assertEqual(bytes(dev.to_host()), self.ref)
            finally:
                dev.close()

    def test_many_small_windows_still_land_intact(self):
        """Staging buffers are reused, and reuse is where a DMA race lives.

        A tiny window forces many wraparounds of the pinned pair, so a buffer
        overwritten while its own copy is still in flight shows up as wrong
        bytes. With the guard written incorrectly this failed intermittently
        and passed most of the time, which is why the window is small enough
        here to make the reuse dominate rather than the reading.
        """
        for target in (self.plain, self.archive):
            with open_model(target) as m:
                dev = m.to_device(window=8192)
                try:
                    self.assertEqual(bytes(dev.to_host()), self.ref, target)
                finally:
                    dev.close()

    def test_windows_are_cut_on_block_boundaries(self):
        """Otherwise a block straddling a seam is decoded twice."""
        with open_model(self.archive) as m:
            spans = [(m._base, m._end)]
            windows = m._device_windows(spans, 4096)
            edges = {c.dst + c.rlen for c in m._arc.chunks_for(spans)}
            edges.add(m._end)
            for _lo, hi in windows:
                self.assertIn(hi, edges)

    def test_device_plan_prices_three_routes(self):
        with open_model(self.archive) as m:
            names = [r.name for r in m.device_plan().routes]
            self.assertEqual(names, ["plain", "host-decode", "device-decode"])


@needs_cuda
@needs_lmz
class TestDeviceDecode(unittest.TestCase):
    """Coded bytes over PCIe, plaintext made in VRAM."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        # fp32, so lmz codes it as CODEC_SPLIT -- independent rANS planes,
        # which is the shape the device decoder takes. A BF16 fixture would
        # come out CODEC_BF16C and exercise only the fallback.
        rng = random.Random(11)
        tensors = {}
        for i in range(6):
            rows = 512 + i * 128
            raw = struct.pack(f"<{rows * 32}f",
                              *[rng.gauss(0, 0.02) for _ in range(rows * 32)])
            tensors[f"w{i}"] = ("F32", (rows, 32), raw)
        cls.plain = write_safetensors(
            os.path.join(cls.dir, "m32.safetensors"), tensors)
        with open(cls.plain, "rb") as fh:
            cls.ref = fh.read()
        cls.archive = os.path.join(cls.dir, "m32.lmz")
        LMZ.compress(cls.plain, cls.archive)
        cls.ctx = _cuda.Context()
        from lmsluice.devdecode import DeviceDecoder

        cls.dd = DeviceDecoder(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        import shutil

        cls.ctx.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_fixture_really_is_the_shape_under_test(self):
        """Guard against the test quietly passing on the fallback path."""
        from lmsluice.archive import Archive

        with Archive(self.archive) as arc:
            codecs = {u.opaque.codec
                      for u in arc.chunks_for([(0, arc.plain_bytes)])}
        self.assertIn(2, codecs, "fixture should contain CODEC_SPLIT chunks")

    def test_decodes_in_vram_byte_identically(self):
        if not self.dd.available:
            self.skipTest(self.dd.why)
        with open_model(self.archive) as m:
            dev = m.to_device(route="device")
            try:
                self.assertEqual(bytes(dev.to_host()), self.ref)
                self.assertEqual(m.device_route, "device")
                self.assertGreater(m.device_stats.device_share, 0.9)
            finally:
                dev.close()

    def test_only_coded_bytes_cross_the_link(self):
        """The claim that makes this route worth having."""
        if not self.dd.available:
            self.skipTest(self.dd.why)
        with open_model(self.archive) as m:
            dev = m.to_device(route="device")
            try:
                self.assertLess(m.device_stats.coded_bytes_uploaded,
                                m.plain_bytes)
            finally:
                dev.close()

    def test_a_codec_it_cannot_read_falls_back_and_says_why(self):
        from lmsluice.archive import Archive

        with Archive(self.archive) as arc:
            chunk = arc.chunks_for([(0, arc.plain_bytes)])[0]

        class Rec:
            codec, esize, flags = 5, 2, 0

        class Conditioned:
            rlen, dst, opaque = chunk.rlen, 0, Rec()

        why = self.dd.takes(Conditioned())
        self.assertIn("CODEC_BF16C", why)

    def test_plane_header_parses_and_refuses_a_truncated_one(self):
        from lmsluice.archive import Archive
        from lmsluice.devdecode import planes_of

        with Archive(self.archive) as arc:
            chunk = [u for u in arc.chunks_for([(0, arc.plain_bytes)])
                     if u.opaque.codec == 2][0]
            payload = arc.fetch_chunk(chunk)
        nplanes, nelem, planes = planes_of(chunk, payload)
        self.assertEqual(nplanes, chunk.opaque.esize)
        self.assertEqual(nplanes * nelem, chunk.rlen)
        self.assertEqual(len(planes), nplanes)
        self.assertIsNone(planes_of(chunk, payload[:4]))

    def test_the_merge_kernel_transposes_correctly(self):
        """Checked against the Python transpose it replaces, for each width."""
        if not self.dd.available:
            self.skipTest(self.dd.why)
        import ctypes

        for nplanes, name in ((2, "lmz_merge_2"), (4, "lmz_merge_4"),
                              (8, "lmz_merge_8"), (3, "lmz_merge_n")):
            nelem = 4096
            planes = bytes((p * 31 + i) & 0xFF
                           for p in range(nplanes) for i in range(nelem))
            want = bytearray(nplanes * nelem)
            for p in range(nplanes):
                want[p::nplanes] = planes[p * nelem:(p + 1) * nelem]
            src = self.ctx.allocate(len(planes))
            dst = self.ctx.allocate(nplanes * nelem)
            try:
                src.copy_from(planes)
                args = [ctypes.c_ulonglong(src.ptr), ctypes.c_ulonglong(dst.ptr)]
                if name == "lmz_merge_n":
                    args.append(ctypes.c_uint(nplanes))
                args.append(ctypes.c_ulonglong(nelem))
                self.dd.module.launch(name, (nelem + 255) // 256, 256, args)
                self.ctx.synchronize()
                self.assertEqual(bytes(dst.to_host()), bytes(want), name)
            finally:
                src.close()
                dst.close()


@needs_zstd
class TestZstdCodec(unittest.TestCase):
    """The codec that needs nothing installed. No lmz in any of these."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = random.Random(3)
        tensors = {f"w{i}": ("BF16", (512,), bf16_bytes(rng, 512))
                   for i in range(6)}
        self.plain = write_safetensors(
            os.path.join(self.dir, "m.safetensors"), tensors)
        with open(self.plain, "rb") as fh:
            self.ref = fh.read()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trips_byte_identically(self):
        from lmsluice.zstdcodec import encoder

        dst = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, dst)
        with open_model(dst) as m:
            self.assertEqual(bytes(m.load()), self.ref)

    def test_the_archive_records_what_it_cost(self):
        """The rates are in the container, so the second load needs no probe."""
        from lmsluice.zstdcodec import encoder, open_codec

        dst = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, dst)
        c = open_codec(dst)
        try:
            self.assertGreater(c.measured_decode_rate or 0, 0)
            self.assertIn("measured", c.cost_model().source)
        finally:
            c.close()

    def test_a_corrupt_chunk_is_refused_not_returned(self):
        """A silently wrong weight is worse than a slow one."""
        from lmsluice.zstdcodec import encoder, open_codec

        dst = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, dst)
        c = open_codec(dst)
        try:
            u = c.units()[0]
            bad = bytearray(c.source.pread(u.off, u.clen))
            # Re-compress different plaintext: valid zstd, wrong bytes, so only
            # the checksum can catch it.
            from lmsluice.zstdcodec import _zstd

            bad = _zstd().compress(b"\x00" * u.rlen, 1)
            with self.assertRaises(ValueError):
                c.decode_chunk(u, bad, memoryview(bytearray(u.rlen)), 0)
        finally:
            c.close()

    def test_tensor_index_survives_into_the_archive(self):
        from lmsluice.zstdcodec import encoder

        dst = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, dst)
        with open_model(self.plain) as a, open_model(dst) as b:
            self.assertEqual(sorted(a.tensors), sorted(b.tensors))
            for n in a.tensors:
                self.assertEqual((a.tensors[n].start, a.tensors[n].end),
                                 (b.tensors[n].start, b.tensors[n].end), n)

    def test_it_declares_that_it_has_no_device_decoder(self):
        from lmsluice.zstdcodec import encoder, open_codec

        dst = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, dst)
        c = open_codec(dst)
        try:
            cap = c.capabilities()
            self.assertFalse(cap.device)
            self.assertFalse(cap.derived, "no kernel to be uncertain about")
        finally:
            c.close()


@needs_zstd
class TestCache(unittest.TestCase):
    """Compress on first fetch — and only where the gate says it pays."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = random.Random(5)
        self.plain = write_safetensors(
            os.path.join(self.dir, "m.safetensors"),
            {f"w{i}": ("BF16", (4096,), bf16_bytes(rng, 4096))
             for i in range(4)})
        with open(self.plain, "rb") as fh:
            self.ref = fh.read()

    def tearDown(self):
        import shutil

        from lmsluice import cache

        cache.clear(self.plain)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_refuses_without_a_measured_link(self):
        """A cache built on a guess is the thing this package argues against."""
        from lmsluice import cache
        from lmsluice.rates import Profile

        v = cache.worth_caching(self.plain, Profile())
        self.assertFalse(v.worth_it)
        self.assertIn("has not been measured", v.why)

    def test_a_fast_link_is_told_to_keep_the_plain_file(self):
        from lmsluice import cache
        from lmsluice.rates import Profile, Storage, storage_key

        p = Profile()
        p.storage[storage_key(self.plain)] = Storage(key="k", cold=50e9)
        v = cache.worth_caching(self.plain, p)
        self.assertFalse(v.worth_it)
        self.assertIn("plain file is the right cache entry", v.why)

    def test_a_slow_link_is_told_to_cache(self):
        from lmsluice import cache
        from lmsluice.rates import Profile, Storage, storage_key

        p = Profile()
        p.storage[storage_key(self.plain)] = Storage(key="k", cold=0.05e9)
        v = cache.worth_caching(self.plain, p)
        self.assertTrue(v.worth_it, v.why)
        self.assertGreater(v.speedup, 1.0)

    def test_a_built_cache_serves_identical_bytes(self):
        from lmsluice import cache

        entry = cache.build(self.plain, codec="zstd")
        try:
            self.assertTrue(os.path.exists(entry))
            with open_model(self.plain) as m:
                self.assertTrue(m.from_cache)
                self.assertEqual(m.route, "coded")
                self.assertEqual(bytes(m.load()), self.ref)
        finally:
            cache.clear(self.plain)

    def test_nothing_is_cached_merely_by_opening(self):
        """Writing gigabytes as a side effect of opening a file is not a favour."""
        from lmsluice import cache

        with open_model(self.plain) as m:
            m.load()
        self.assertIsNone(cache.find(self.plain))

    def test_an_edited_file_misses_rather_than_serving_stale_weights(self):
        from lmsluice import cache

        cache.build(self.plain, codec="zstd")
        self.assertIsNotNone(cache.find(self.plain))
        with open(self.plain, "r+b") as fh:      # touch size and mtime
            fh.seek(0, os.SEEK_END)
            fh.write(b"\x00")
        self.assertIsNone(cache.find(self.plain),
                          "a changed file must miss, never serve stale bytes")


class TestBoundary(unittest.TestCase):
    """The division in `docs/boundary.md`, enforced rather than described.

    Walks the package *source* rather than the import graph, so a new module
    that reaches across the line is caught the day it is written and not the
    day someone happens to import it on a machine where lmz is absent.
    """

    # Located through the imported package rather than by path arithmetic from
    # this file, so the boundary is checked on the tree that actually got
    # imported -- which is the installed one when the suite runs against a
    # wheel, and that is the tree whose contents ship.
    PACKAGE = os.path.dirname(os.path.abspath(_lmsluice_pkg.__file__))

    # The modules allowed to know lmz exists. `_lmz` finds it, `lmzcodec` is
    # its adapter, `devdecode` is a registered loan (`merge.cu`'s dispatch).
    # Every name here is a line in the loan register or an adapter itself.
    ADAPTERS = {"_lmz.py", "lmzcodec.py", "devdecode.py"}

    # Modules that implement a codec and therefore parse one coded stream by
    # definition -- theirs. `zstdcodec` is the second implementation of the
    # `Codec` protocol and the reason that protocol is no longer theoretical.
    CODEC_ADAPTERS = {"lmzcodec.py", "zstdcodec.py"}

    # Coded-stream vocabulary. A module outside the adapter mentioning these is
    # parsing format structure, which is I4's line.
    # Residue: vocabulary that cannot be derived from anything the adapter
    # declares, and so has to be written down. Kept deliberately short, and
    # each entry is here because there is no declaration to read it from --
    # not because nobody got round to it.
    FORMAT_RESIDUE = (
        "TableSet",        # an lmz internal type, named in no public surface
        "_chunk_decoder",  # likewise
        "nplanes",         # the plane split is format structure with no API
        "rANS",            # the coder's name
    )

    def format_words(self):
        """The vocabulary, derived where it can be.

        Hand-maintaining this list was the failure mode: fourteen entries is
        past the point where a genuine miss is plausible, and the miss is
        silent -- an encoder option lmz adds tomorrow would simply not be
        checked. Codec IDs come from the adapter's own constants and encoder
        option names from `encode_options()`, so the check grows when the
        codec does.
        """
        words = list(self.FORMAT_RESIDUE)
        try:
            from lmsluice import lmzcodec

            words += [n for n in dir(lmzcodec)
                      if n.startswith(("CODEC_", "METHOD_", "COND_"))]
            # Format and schedule options only. An `observe` option is a
            # callback that touches neither the bytes nor the schedule, so
            # naming one outside the adapter leaks nothing -- and its names
            # are generic enough to collide with unrelated local API. The
            # derived check caught `transport()`'s own `progress=` parameter,
            # which is the collision this exclusion exists for.
            words += [f"{name}=" for name, opt
                      in lmzcodec.LmzEncoder().encode_options().items()
                      if opt.kind != "observe"]
        except Exception:                 # noqa: BLE001 -- no lmz, no extras
            pass
        return tuple(dict.fromkeys(words))

    def sources(self):
        for name in sorted(os.listdir(self.PACKAGE)):
            if name.endswith(".py"):
                with open(os.path.join(self.PACKAGE, name)) as fh:
                    yield name, fh.read()

    # The one module allowed to know torch exists. "Runs with no torch" is a
    # recorded competitive difference, not a preference -- competition.md §4
    # lists ZipNN, tensorizer, Run:ai and fastsafetensors as unable to import
    # without a tensor framework, and §7 counts "no install required" as a row
    # only lmsluice ticks.
    TORCH_ADAPTERS = {"torchadapter.py"}

    VLLM_ADAPTERS = {"vllmloader.py"}

    def test_only_the_vllm_loader_imports_vllm(self):
        """Same containment as lmz and torch, for the same reason.

        `vllmloader` is imported by vLLM and by nothing else, so it may import
        vLLM at module scope. Anywhere else that import would make the package
        require a multi-gigabyte inference server to open a file.
        """
        offenders = []
        for name, src in self.sources():
            if name in self.VLLM_ADAPTERS:
                continue
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith(("import vllm", "from vllm")):
                    offenders.append(f"{name}: {stripped}")
        self.assertEqual(offenders, [],
                         "only lmsluice/vllmloader.py may import vllm")

    def test_only_the_torch_adapter_imports_torch(self):
        """One stray import would retire a published claim, silently."""
        offenders = []
        for name, src in self.sources():
            if name in self.TORCH_ADAPTERS:
                continue
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith(("import torch", "from torch")):
                    offenders.append(f"{name}: {stripped}")
        self.assertEqual(offenders, [],
                         "only lmsluice/torchadapter.py may import torch -- "
                         "'needs nothing installed' is a competitive claim in "
                         "docs/competition.md §4 and §7")

    def test_the_core_imports_with_torch_made_unavailable(self):
        """Proved by importing it that way, not by grepping for the import.

        A source walk catches `import torch`; it does not catch a dependency
        that pulls torch in transitively, or a module-level `numpy` that only
        happens to be installed here. This imports every core module in a
        fresh interpreter with torch forced to fail, which is the claim as a
        user on a phone or a CUDA-less laptop would experience it.
        """
        import subprocess

        prog = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        return self if name == 'torch' or name.startswith('torch.') else None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch is hidden by the test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "sys.path.insert(0, %r)\n"
            "import importlib, pkgutil, lmsluice\n"
            "bad = []\n"
            "for m in pkgutil.iter_modules(lmsluice.__path__):\n"
            "    if m.name in ('torchadapter', 'vllmloader'):\n"
            "        continue\n"
            "    try:\n"
            "        importlib.import_module('lmsluice.' + m.name)\n"
            "    except Exception as e:\n"
            "        bad.append(m.name + ': ' + type(e).__name__ + ': ' + str(e))\n"
            "print('OK' if not bad else 'FAILED ' + '; '.join(bad))\n"
        ) % os.path.dirname(self.PACKAGE)
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, timeout=180)
        self.assertIn("OK", r.stdout,
                      f"the core does not import without torch:\n"
                      f"{r.stdout}\n{r.stderr[-600:]}")

    def test_the_torch_adapter_itself_imports_without_torch(self):
        """Importing the adapter must not require torch either -- only calling it.

        Otherwise `from lmsluice import torchadapter` inside a `try` becomes
        the way people test for the feature, and an ImportError at import time
        is indistinguishable from the package being broken.
        """
        import subprocess

        prog = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('hidden')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "sys.path.insert(0, %r)\n"
            "from lmsluice import torchadapter\n"
            "ok, why = torchadapter.available()\n"
            "assert ok is False, 'available() should report torch missing'\n"
            "try:\n"
            "    torchadapter.load_file('x.safetensors')\n"
            "except ImportError as e:\n"
            "    assert 'lmsluice[torch]' in str(e), str(e)\n"
            "    print('OK')\n"
        ) % os.path.dirname(self.PACKAGE)
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, timeout=180)
        self.assertIn("OK", r.stdout, f"{r.stdout}\n{r.stderr[-600:]}")

    def test_only_the_adapters_import_lmz(self):
        offenders = []
        for name, src in self.sources():
            if name in self.ADAPTERS:
                continue
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Matched as statements, not substrings: prose like "from
                # lmz's published cost model" inside a description string is
                # not an import, and a check that cannot tell the difference
                # trains people to ignore it.
                if (stripped.startswith(("import lmz", "from lmz",
                                         "from ._lmz"))
                        or stripped.startswith("from ._lmz import")):
                    offenders.append(f"{name}: {stripped}")
        self.assertEqual(offenders, [],
                         "only the adapter may import lmz -- see docs/boundary.md")

    def test_only_the_adapters_know_the_coded_stream(self):
        offenders = []
        for name, src in self.sources():
            if name in self.ADAPTERS or name in self.CODEC_ADAPTERS:
                continue
            body = "\n".join(l for l in src.splitlines()
                              if not l.strip().startswith("#"))
            # The module docstring discusses the boundary by necessity; strip
            # it before looking, so prose about the rule is not a breach of it.
            if body.count('"""') >= 2:
                body = body.split('"""', 2)[2]
            for word in self.format_words():
                if word in body:
                    offenders.append(f"{name}: {word}")
        self.assertEqual(offenders, [],
                         "coded-stream structure belongs in the adapter (I4)")

    @needs_repo
    def test_the_boundary_document_exists_and_is_linked(self):
        root = os.path.dirname(self.PACKAGE)
        doc = os.path.join(root, "docs", "boundary.md")
        self.assertTrue(os.path.exists(doc))
        text = open(doc).read()
        for required in ("three-question test", "loan register",
                         "I1", "I5", "compressor/decompressor",
                         "transport facilitator"):
            self.assertIn(required, text, f"boundary.md must state {required!r}")
        for linker in ("README.md", "CLAUDE.md"):
            with open(os.path.join(root, linker)) as fh:
                self.assertIn("boundary.md", fh.read(),
                              f"{linker} must link the boundary document")


@needs_lmz
class TestLoansRetire(unittest.TestCase):
    """Each loan fails when lmz ships the interface that retires it.

    A loan with no expiry is just a drift with an apology attached. These are
    the expiries: when one fails, delete the loan and its entry in
    `docs/boundary.md` rather than editing the test to pass.
    """

    def test_decode_chunk_loan_is_retired_and_the_adapter_uses_it(self):
        """This loan is RETIRED. The test now guards the retirement instead.

        It used to look for module-level names -- `decode_chunk`,
        `decode_block`, `decode_unit` -- and lmz shipped the replacement as a
        *method*, `ArchiveIndex.decode`. So the expiry test did not fire when
        the thing it was watching for arrived, which is the exact failure the
        loan register exists to prevent: the register outlived its reason and
        nothing said so.

        The lesson, kept because it generalises: an expiry test that guesses at
        the *shape* of a future API only fires if the guess was right. Watch
        for the capability, not the spelling -- and when it lands, invert the
        test into one that keeps the adapter on the public path.
        """
        index = getattr(LMZ, "ArchiveIndex", None)
        if index is None or not hasattr(index, "decode"):
            self.skipTest("this lmz predates ArchiveIndex; the loan is live")
        d = tempfile.mkdtemp()
        try:
            plain = make_model(d)
            arc = os.path.join(d, "m.lmz")
            LMZ.compress(plain, arc)
            from lmsluice.archive import Archive

            with Archive(arc) as a:
                self.assertEqual(
                    a.route, "public",
                    "lmz publishes ArchiveIndex.decode, so the adapter must "
                    "use it rather than reaching for api._chunk_decoder")
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)

    def test_device_decode_returns_plaintext_loan(self):
        """Retired by a device entry point that returns plaintext (I2).

        Asserts two things, because `Unit.opaque` exists only to serve this
        loan: the interface has not arrived, *and* nothing outside
        `devdecode.py` has started reaching through `opaque`. When this fires,
        the kernel, its dispatch and `opaque` come out together -- leaving
        `opaque` behind would keep the hole open after its reason had gone.
        """
        import os as _os

        try:
            from lmsluice._lmz import require

            (gpu,) = require("gpu")
        except Exception:                 # noqa: BLE001
            self.skipTest("lmz.gpu not importable")
        newer = [n for n in ("decode_chunks_dev", "decode_to_plaintext_dev",
                             "decode_batch_plain_dev") if hasattr(gpu, n)]
        self.assertEqual(
            newer, [],
            f"lmz.gpu now exposes {newer}: retire merge.cu, its dispatch in "
            f"devdecode.py, AND Unit.opaque -- they share one cause")

        pkg = TestBoundary.PACKAGE
        reaches = []
        for name in sorted(_os.listdir(pkg)):
            # `codec.py` declares the field and `lmzcodec.py` is what puts
            # the record in it -- for the adapter, reaching its own payload is
            # definitional rather than a reach across the line. `devdecode.py`
            # is the loan this field exists to serve. Everything else is a
            # transport module and must not know the field exists.
            if not name.endswith(".py") or name in (
                    "devdecode.py", "codec.py", "lmzcodec.py",
                    # A codec adapter owns the payload it puts in `opaque`;
                    # reaching its own record is definitional, not a reach.
                    "zstdcodec.py"):
                continue
            with open(_os.path.join(pkg, name)) as fh:
                for line in fh:
                    if ".opaque" in line and not line.strip().startswith("#"):
                        reaches.append(f"{name}: {line.strip()}")
        self.assertEqual(reaches, [],
                         "Unit.opaque serves the merge loan only; it is not "
                         "general interface")

    def test_capability_declaration_loan(self):
        """Retired by a per-archive decoder-capability declaration (I3).

        `Capabilities.advice` expires with this too: it exists to tell a
        writer which flag to set, and a declaring format makes that
        unnecessary.
        """
        d = tempfile.mkdtemp()
        try:
            plain = make_model(d)
            arc = os.path.join(d, "m.lmz")
            LMZ.compress(plain, arc)
            info = LMZ.info(arc)
            manifest = info.get("manifest", {})
            declared = [k for k in ("decoders", "capabilities", "readable_by")
                        if k in manifest]
            self.assertEqual(
                declared, [],
                f"lmz's manifest now declares {declared}: retire "
                f"gpu_chunk_size() and read the declaration instead")
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)


class TestDestinationBuffer(unittest.TestCase):
    """The destination costs more than the transport, so how it is got matters.

    Two of the three things that make the cheap path cheap fail *silently* when
    they regress -- a shared mapping and an unadvised one both allocate fine and
    simply stop being backed by huge pages -- so these check the effect, not the
    call.
    """

    def test_it_reads_as_zeroes_like_the_bytearray_it_replaces(self):
        """The contract, and the reason this is not `numpy.empty`.

        A partial load leaves the ranges nobody asked for untouched, and callers
        are entitled to read them as zero rather than as the previous tenant's
        bytes.
        """
        from lmsluice.buffer import destination

        for n in (0, 1, 4096, 1 << 20, 32 << 20):
            buf = destination(n)
            self.assertEqual(len(buf), n)
            self.assertEqual(bytes(memoryview(buf)), b"\x00" * n,
                             f"{n} bytes came back non-zero")

    def test_it_is_writable_through_a_memoryview(self):
        """Every route fills it through one, so that is what must work."""
        from lmsluice.buffer import destination

        n = 32 << 20
        buf = destination(n)
        view = memoryview(buf).cast("B")
        view[:5] = b"hello"
        view[n - 5:] = b"world"
        self.assertEqual(bytes(view[:5]), b"hello")
        self.assertEqual(bytes(view[n - 5:]), b"world")

    def test_small_buffers_do_not_pay_for_a_syscall(self):
        """Measured parity below ~8 MB, so the mapping buys nothing there."""
        from lmsluice.buffer import destination, strategy

        how, _why, floor = strategy()
        if how != "mapped":
            self.skipTest(f"no mapped path here: {_why}")
        self.assertIsInstance(destination(floor // 2), bytearray)
        self.assertNotIsInstance(destination(floor), bytearray)

    def test_the_threshold_is_derived_from_the_machine_not_named(self):
        """arm64 alone ships 64 KiB, 2 MiB, 32 MiB and 512 MiB huge pages."""
        from lmsluice.buffer import THRESHOLD_PAGES, huge_page_bytes, strategy

        how, _why, floor = strategy()
        if how != "mapped":
            self.skipTest(_why)
        self.assertEqual(floor, THRESHOLD_PAGES * huge_page_bytes())

    def test_the_mapping_actually_gets_huge_pages(self):
        """The regression guard for `MAP_PRIVATE` and for the advice.

        Drop either and this still allocates, still zeroes, still passes every
        other test here -- and quietly costs 5x. So it is checked against the
        kernel's own accounting rather than against the arguments passed.
        """
        from lmsluice.buffer import destination, huge_page_bytes, strategy

        how, why, floor = strategy()
        if how != "mapped":
            self.skipTest(why)
        n = max(floor, 64 << 20)
        buf = destination(n)
        view = memoryview(buf).cast("B")
        for i in range(0, n, huge_page_bytes()):   # fault it in
            view[i] = 1
        try:
            with open("/proc/self/smaps") as fh:
                backed = max((int(l.split()[1]) for l in fh
                              if l.startswith("AnonHugePages:")), default=0)
        except OSError:
            self.skipTest("no /proc/self/smaps to check against")
        self.assertGreaterEqual(
            backed * 1024, n // 2,
            "the destination is not backed by huge pages: MAP_PRIVATE or "
            "MADV_HUGEPAGE has been dropped, and nothing else would show it")

    def test_a_machine_without_huge_pages_says_so_and_falls_back(self):
        """`strategy()` may not claim a mapping the kernel will not back."""
        from lmsluice import buffer

        import mmap as _mmap

        saved_state, saved_probe = buffer._strategy, buffer._thp_mode
        try:
            buffer._strategy = None
            buffer._thp_mode = lambda: "never"
            how, why, floor = buffer.strategy()
            self.assertEqual(how, "bytes")
            self.assertEqual(floor, 0)
            # The reason depends on which check declines first, and that is a
            # platform fact rather than a preference. Where the mmap constants
            # exist at all, `strategy()` reaches the THP mode and should name
            # it; on macOS it stops earlier, at the missing constant, and
            # naming *that* is the correct answer. Asserting "never" everywhere
            # asserted a Linux-only path and failed on the first Mac to run it.
            has_consts = all(hasattr(_mmap, n) for n in
                             ("MAP_PRIVATE", "MAP_ANONYMOUS", "MADV_HUGEPAGE"))
            self.assertIn("never" if has_consts else "absent", why)
            self.assertTrue(why, "the fallback must say why")
            buffer._strategy = ("bytes", "forced", 0)
            self.assertIsInstance(buffer.destination(64 << 20), bytearray)
        finally:
            buffer._strategy, buffer._thp_mode = saved_state, saved_probe


class TestGateModel(unittest.TestCase):
    """The cycles curve: what it takes from the codec, and what it refuses.

    lmz 1.3.0 separated the two readings this model used to report side by
    side. These pin the properties that made that safe rather than the numbers,
    which are the codec's and move when it does.
    """

    def setUp(self):
        from lmsluice import gate

        self.gate = gate

    def test_k_comes_from_the_codec_not_from_a_copy(self):
        """A constant copied across the boundary goes stale without saying so."""
        from lmsluice.codec import Cost, CodecCost

        published = CodecCost(
            k=Cost("k", low=111.0, high=222.0,
                   unit=self.gate.K_UNIT, provenance="a test codec"),
            source="test")
        c = self.gate.cost_from(published)
        self.assertEqual((c.k_low, c.k_high), (111.0, 222.0))
        self.assertNotEqual((c.k_low, c.k_high),
                            self.gate.FALLBACK["k_cycles_per_byte"])

    def test_a_k_in_the_wrong_unit_is_refused_not_converted(self):
        """The refusal predates lmz answering the question and must survive it.

        Being right today is not a reason to stop checking: this is how the
        FP32 proxy survived as long as it did.
        """
        from lmsluice.codec import Cost, CodecCost

        wrong = CodecCost(
            k=Cost("k", low=9.0, high=9.0, unit="FP32 FLOP per byte",
                   provenance="a codec using the retired unit"),
            source="test")
        c = self.gate.cost_from(wrong)
        self.assertIn("REFUSED", c.provenance)
        self.assertEqual((c.k_low, c.k_high),
                         self.gate.FALLBACK["k_cycles_per_byte"],
                         "a refused k must not reach the curve")

    def test_a_refusal_still_uses_the_fields_that_were_in_the_right_unit(self):
        """One bad field is not a reason to discard a whole published record."""
        from lmsluice.codec import Cost, CodecCost

        wrong = CodecCost(
            k=Cost("k", low=9.0, high=9.0, unit="nonsense", provenance="x"),
            expansion=Cost("expansion", low=4.0, high=4.0,
                           unit="plain bytes per coded byte", provenance="x"),
            source="test")
        c = self.gate.cost_from(wrong)
        self.assertIn("REFUSED", c.provenance)
        self.assertEqual(c.expansion, 4.0)

    def test_compute_scales_by_lanes_and_clock_and_never_by_flops(self):
        """k is dependent-load latency, not an issue rate.

        lmz establishes this by decode being linear in resident lanes across an
        84x range, where an issue-limited kernel would have flattened. The
        consumer-facing consequence is that a device's FP32 rate must not enter
        the compute term -- so `gflops` may move by any amount and change
        nothing.
        """
        c = self.gate.cost_from(None)
        base = self.gate.Device("d", 100.0, "test", units=4,
                                shmem_pool_per_unit=128 << 10,
                                shmem_cap_per_block=128 << 10,
                                max_threads_per_unit=2048, clock_ghz=1.0,
                                gflops=1_000)
        flops = self.gate.Device("d", 100.0, "test", units=4,
                                 shmem_pool_per_unit=128 << 10,
                                shmem_cap_per_block=128 << 10,
                                 max_threads_per_unit=2048, clock_ghz=1.0,
                                 gflops=1_000_000)
        self.assertEqual(base.compute_bound(c), flops.compute_bound(c),
                         "a FLOP rate reached the compute term")

        faster = self.gate.Device("d", 100.0, "test", units=4,
                                  shmem_pool_per_unit=128 << 10,
                                shmem_cap_per_block=128 << 10,
                                  max_threads_per_unit=2048, clock_ghz=2.0)
        a, b = base.compute_bound(c), faster.compute_bound(c)
        self.assertAlmostEqual(b[0] / a[0], 2.0, places=6)

        wider = self.gate.Device("d", 100.0, "test", units=8,
                                 shmem_pool_per_unit=128 << 10,
                                shmem_cap_per_block=128 << 10,
                                 max_threads_per_unit=2048, clock_ghz=1.0)
        self.assertAlmostEqual(wider.compute_bound(c)[0] / a[0], 2.0, places=6)

    def test_it_picks_only_when_k_came_from_a_sweep(self):
        """One point fits both readings; a sweep across residency separates them."""
        from lmsluice.codec import Cost, CodecCost

        def built(single):
            return self.gate.cost_from(CodecCost(
                k=Cost("k", low=217.0, high=248.0, unit=self.gate.K_UNIT,
                       provenance="test"),
                k_single_point=single, source="test"))

        self.assertTrue(built(False).can_pick)
        self.assertFalse(built(True).can_pick,
                         "a single-point fit must not be quoted as a pick")

    def test_the_model_reproduces_lmz_own_measured_occupancy(self):
        """The cross-check that catches a wrong shared-memory budget.

        lmz publishes both the occupancy it measured at and the compute
        ceilings that occupancy implies, so on the one device where both sides
        are known this model can be tested rather than trusted. It failed this
        for a while: using the RTX 5080's 228 KiB per-SM pool gives 7 blocks a
        unit where lmz measured 3, over-predicting residency 2.3x and every
        compute ceiling with it. The opt-in per-block figure, 99 KiB, is what
        lmz divides by and reproduces all of it.

        Without this test the error is invisible, because the reference row is
        bandwidth-bound either way and its printed decode rate does not move.
        """
        from lmsluice.lmzcodec import LmzCodec

        published = LmzCodec.cost_model(None)
        if published.blocks_per_unit_at_measurement.low is None:
            self.skipTest("this lmz does not publish its measurement occupancy")
        c = self.gate.cost_from(published)
        ref = [d for d in self.gate.DEVICES if "5080" in d.name][0]
        lo = int(published.blocks_per_unit_at_measurement.low)
        hi = int(published.blocks_per_unit_at_measurement.high)
        self.assertIsNotNone(ref.shmem_pool_per_unit, "pool not recorded")
        self.assertIsNotNone(ref.shmem_cap_per_block, "cap not recorded")
        got = ref.blocks_per_unit(c)
        self.assertGreaterEqual(got, lo, "residency over- or under-predicted")
        self.assertLessEqual(got, hi,
                             f"model derives {got} blocks/unit where lmz "
                             f"measured {lo}-{hi}: the shared-memory budget "
                             f"this uses is not the one lmz divides by")

    def test_cap_and_pool_are_not_interchangeable(self):
        """lmz's residency_formula needs both, and they mean different things.

        `blocks = floor(pool / shm) if shm <= cap else 0`. The cap decides
        whether one block is legal; the pool decides how many fit. On discrete
        NVIDIA parts they agree within a kilobyte, so conflating them is
        invisible there and diverges by up to 3x on integrated parts -- the
        device class this model exists for. This repository has now made that
        conflation twice, in opposite directions.
        """
        c = self.gate.cost_from(None)
        shm = c.shmem_per_block()

        # A pool big enough for three, but a cap that forbids the block: zero.
        self.assertEqual(c.blocks_per_unit(shm * 3, shm - 1, 4096), 0)
        # The same pool with a permissive cap: three.
        self.assertEqual(c.blocks_per_unit(shm * 3, shm, 4096), 3)
        # A generous cap does not add residency; only the pool does.
        self.assertEqual(c.blocks_per_unit(shm, shm * 100, 4096), 1)

    def test_residency_takes_the_smaller_of_two_limits(self):
        """A unit caps resident threads as well as shared memory.

        lmz's published formula carried only the shared-memory term, which held
        on its card because shared memory binds there at every block size. It
        holds on ours too -- but "it does not bind on the devices we have
        looked at" is the reasoning that produced several wrong numbers here,
        and this model projects onto parts nobody owns.

        The hazard is live rather than hypothetical: the shared-memory term is
        large mostly because of a fixed 16 KiB lookup table, so narrowing that
        table shrinks the very margin by which it currently wins.
        """
        c = self.gate.cost_from(None)
        shm = c.shmem_per_block()

        # A pool with room for sixteen, a thread budget with room for one.
        self.assertEqual(
            c.blocks_per_unit(shm * 16, shm * 16, c.block_threads), 1,
            "the threads-per-unit limit is not being applied")
        # And the reverse, so the min() is genuinely a min and not a swap.
        self.assertEqual(
            c.blocks_per_unit(shm, shm, c.block_threads * 16), 1)

        for d in self.gate.DEVICES:
            binds = d.residency_binds(c)
            if binds is not None:
                self.assertEqual(binds, "shared memory",
                                 f"{d.name}: the binding residency term has "
                                 f"changed; every row derived from it needs "
                                 f"rechecking")

    def test_a_device_with_no_pool_gets_no_compute_term(self):
        """An unobtained pool is not a small pool and must not collapse to one.

        The 2-CU adapter has a queried cap (Vulkan's per-workgroup limit) and
        no pool -- no core Vulkan query reports one, and the device is not
        reachable from Vulkan under WSL2 at all. Substituting the cap gives
        1 block and 1.9 GB/s; substituting an RDNA2 architectural figure gives
        4 and 7.8. Both have been published by this file as though settled.
        """
        c = self.gate.cost_from(None)
        small = [d for d in self.gate.DEVICES if "2-CU" in d.name][0]
        self.assertIsNotNone(small.shmem_cap_per_block)
        self.assertIsNone(small.shmem_pool_per_unit,
                          "a pool was supplied for a device nobody has queried")
        self.assertIsNone(small.blocks_per_unit(c))
        self.assertIsNone(small.compute_bound(c))
        lo, hi, why = small.predict(c)
        self.assertEqual(lo, 0.0)
        self.assertIn("bandwidth ceiling only", why)

    def test_the_small_igpu_row_stays_one_sided(self):
        """A tighter interval is not a measurement.

        Nobody has run a 2-CU device. The narrowing came from a sweep on an
        RTX 5080, so the row must still be a ceiling with no floor -- letting a
        resolved ambiguity read as a measured small-device number is the error
        this model exists to avoid.
        """
        c = self.gate.cost_from(None)
        small = [d for d in self.gate.DEVICES if "2-CU" in d.name][0]
        lo, hi, why = small.predict(c)
        self.assertEqual(lo, 0.0, "the 2-CU row grew a floor it has not earned")
        self.assertFalse(small.occupancy_verified)

    def test_it_still_runs_with_nothing_installed(self):
        """The whole point is that it can be copied onto someone else's box."""
        import shutil
        import subprocess

        d = tempfile.mkdtemp()
        try:
            shutil.copy(
                os.path.join(os.path.dirname(_lmsluice_pkg.__file__), "gate.py"),
                os.path.join(d, "gate.py"))
            r = subprocess.run([sys.executable, "gate.py"], cwd=d,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertIn("decode faster than its disk", r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


try:
    import torch
    import torch as _TORCH
except Exception:                              # noqa: BLE001 -- absence is the point
    _TORCH = None
needs_torch = unittest.skipIf(_TORCH is None, "torch is not installed")


@needs_torch
class TestTorchAdapter(unittest.TestCase):
    """`load_file` against the bytes it claims to reproduce.

    The comparison is with the fixture's own bytes rather than with
    `safetensors`, so the suite does not acquire a second optional dependency
    to test the first. (It has also been checked against
    `safetensors.torch.load_file` on 1.87 GB of real BF16 weights, on both
    routes and on CUDA -- see the report; that is not something a unit test
    should carry.)
    """

    DTYPES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1}

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = random.Random(11)
        self.raw = {}
        spec = {}
        for i, (dt, width) in enumerate(self.DTYPES.items()):
            n = 8 + i
            payload = bytes(rng.randrange(256) for _ in range(n * width))
            self.raw[f"w_{dt}"] = payload
            spec[f"w_{dt}"] = (dt, (n,), payload)
        # One 2-D tensor, because reshape is the part most easily got wrong.
        flat = bytes(rng.randrange(256) for _ in range(2 * 3 * 4))
        self.raw["w_2d"] = flat
        spec["w_2d"] = ("F32", (2, 3), flat)
        self.plain = write_safetensors(
            os.path.join(self.dir, "m.safetensors"), spec)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def _check(self, got):
        self.assertEqual(set(got), set(self.raw))
        for name, want in self.raw.items():
            t = got[name]
            self.assertEqual(
                t.detach().cpu().contiguous().view(torch.uint8)
                 .reshape(-1).numpy().tobytes(), want,
                f"{name}: bytes differ")
        self.assertEqual(tuple(got["w_2d"].shape), (2, 3))
        self.assertEqual(got["w_BF16"].dtype, torch.bfloat16)

    def test_the_plain_route_reproduces_the_file_bytes(self):
        from lmsluice.torchadapter import load_file

        self._check(load_file(self.plain))

    def test_the_coded_route_reproduces_the_same_bytes(self):
        """The offset arithmetic differs by route, so both are checked.

        A plain safetensors file bases its buffer at 0 and a coded archive at
        the member's own start, so `t.start - base` is not the same expression
        in the two cases even though the adapter code is.
        """
        from lmsluice.torchadapter import load_file
        from lmsluice.zstdcodec import encoder

        coded = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, coded)
        self._check(load_file(coded))

    def test_the_tensors_are_views_into_one_buffer(self):
        """Zero-copy is the reason this lives in the package, so it is checked.

        Every tensor should point inside a single allocation the size of the
        model, not into `len(tensors)` separate ones.
        """
        from lmsluice.torchadapter import load_file

        got = load_file(self.plain)
        ptrs = sorted(t.data_ptr() for t in got.values())
        total = sum(t.numel() * t.element_size() for t in got.values())
        self.assertLessEqual(ptrs[-1] - ptrs[0], total,
                             "tensors are spread across separate allocations")

    def test_a_device_it_cannot_fill_is_refused_by_name(self):
        from lmsluice.torchadapter import load_file

        with self.assertRaises(ValueError) as caught:
            load_file(self.plain, device="mps")
        self.assertIn("mps", str(caught.exception))

    def test_every_mapped_dtype_exists_in_this_torch(self):
        """The mapping is data, so a typo in it fails here and not at a user."""
        from lmsluice import torchadapter

        missing = [(k, v) for k, v in torchadapter._DTYPES.items()
                   if not hasattr(torch, v)]
        # Only the newest dtypes may legitimately be absent on an older torch.
        unexpected = [k for k, _ in missing
                      if k not in {"F8_E4M3", "F8_E5M2", "U16", "U32", "U64"}]
        self.assertEqual(unexpected, [],
                         f"dtype names not present in torch {torch.__version__}")


@needs_torch
@needs_cuda
class TestTorchAdapterCuda(unittest.TestCase):
    """The device path, which has an alignment problem the host path does not."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = random.Random(12)
        payload = bf16_bytes(rng, 256)
        self.raw = {"w": payload}
        self.plain = write_safetensors(
            os.path.join(self.dir, "m.safetensors"), {"w": ("BF16", (256,), payload)})

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_lands_on_the_device_with_the_right_bytes(self):
        from lmsluice.torchadapter import load_file

        got = load_file(self.plain, device="cuda")
        self.assertEqual(got["w"].device.type, "cuda")
        self.assertEqual(got["w"].dtype, torch.bfloat16)
        self.assertEqual(
            got["w"].detach().cpu().contiguous().view(torch.uint8)
                   .reshape(-1).numpy().tobytes(), self.raw["w"])

    def test_it_copies_only_what_alignment_forces(self):
        """A misaligned tensor must be copied, and the result must say so.

        safetensors puts tensor data straight after a JSON header of arbitrary
        length, so whether any tensor can be viewed in place is a property of
        the file. The adapter reports the bytes it had to move rather than
        claiming a zero-copy it may not have achieved.
        """
        from lmsluice.torchadapter import load_file

        got = load_file(self.plain, device="cuda")
        moved = getattr(got["w"], "_lmsluice_copied_bytes", None)
        self.assertIsNotNone(moved, "the device path did not report its copies")
        self.assertIn(moved, (0, len(self.raw["w"])),
                      "copied bytes should be none of the tensor or all of it")


class TestVllmLoader(unittest.TestCase):
    """The vLLM loader, against a stub of the two symbols it builds on.

    vLLM is multi-gigabyte and is not installed here, so what can be checked is
    the contract: that the decorator registers under the name users will type,
    that the class satisfies `BaseModelLoader`, and that `load_weights` hands
    the model a generator of `(name, tensor)` rather than a materialised dict.
    The stub is written from vLLM's published source for `BaseModelLoader` and
    `register_model_loader`.

    **What this cannot check is that vLLM accepts it**, and nothing on this
    machine can. That needs a box with vLLM installed and is recorded as such.
    """

    def _stub_vllm(self):
        """Install a minimal `vllm.model_executor.model_loader` and return the
        registry the decorator writes into."""
        import types

        registry = {}

        class BaseModelLoader:
            def __init__(self, load_config=None):
                self.load_config = load_config

        def register_model_loader(load_format):
            def wrap(cls):
                if not issubclass(cls, BaseModelLoader):
                    raise TypeError("not a BaseModelLoader")
                registry[load_format] = cls
                return cls
            return wrap

        loader = types.ModuleType("vllm.model_executor.model_loader")
        loader.BaseModelLoader = BaseModelLoader
        loader.register_model_loader = register_model_loader
        executor = types.ModuleType("vllm.model_executor")
        executor.model_loader = loader
        root = types.ModuleType("vllm")
        root.model_executor = executor
        for name, mod in (("vllm", root),
                          ("vllm.model_executor", executor),
                          ("vllm.model_executor.model_loader", loader)):
            self._added.append(name)
            sys.modules[name] = mod
        return registry

    def _load_module(self):
        import importlib

        sys.modules.pop("lmsluice.vllmloader", None)
        self._added.append("lmsluice.vllmloader")
        return importlib.import_module("lmsluice.vllmloader")

    def setUp(self):
        self._added = []
        self.dir = tempfile.mkdtemp()
        rng = random.Random(21)
        self.payload = bf16_bytes(rng, 64)
        self.plain = write_safetensors(
            os.path.join(self.dir, "model.safetensors"),
            {"w": ("BF16", (64,), self.payload)})

    def tearDown(self):
        import shutil

        for name in self._added:
            sys.modules.pop(name, None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_registers_under_the_name_users_type(self):
        registry = self._stub_vllm()
        mod = self._load_module()
        self.assertIn("lmsluice", registry,
                      "--load-format lmsluice would not resolve")
        self.assertIs(registry["lmsluice"], mod.LmsluiceModelLoader)
        mod.register()          # the entry point must be callable and harmless

    def test_it_implements_both_abstract_methods(self):
        self._stub_vllm()
        mod = self._load_module()
        for name in ("download_model", "load_weights"):
            self.assertTrue(callable(getattr(mod.LmsluiceModelLoader, name, None)),
                            f"{name} is required by BaseModelLoader")

    def test_a_repository_id_is_refused_with_the_reason(self):
        """Failing here beats failing after the engine is half up."""
        self._stub_vllm()
        mod = self._load_module()

        class Cfg:
            model = "meta-llama/Llama-3.1-8B"

        loader = mod.LmsluiceModelLoader(load_config=None)
        with self.assertRaises(ValueError) as caught:
            loader.download_model(Cfg())
        self.assertIn("no Hub", str(caught.exception))

    def test_a_local_path_needs_no_download(self):
        self._stub_vllm()
        mod = self._load_module()

        class Cfg:
            model = self.dir

        self.assertIsNone(
            mod.LmsluiceModelLoader(load_config=None).download_model(Cfg()))

    @needs_zstd
    def test_a_coded_archive_is_preferred_over_the_plain_file(self):
        """Both present is the normal case while someone is evaluating this;
        reading both would load every weight twice."""
        self._stub_vllm()
        mod = self._load_module()
        from lmsluice.zstdcodec import encoder

        encoder().encode(self.plain, os.path.join(self.dir, "model.lmsl"))
        got = mod.model_files(self.dir)
        self.assertEqual([os.path.basename(f) for f in got], ["model.lmsl"])

    @needs_torch
    def test_it_reports_a_weight_loading_time_like_vllm_does(self):
        """vLLM's own timing line comes from DefaultModelLoader, not the base.

        A loader subclassing BaseModelLoader directly inherits no counters and
        emits nothing, so comparing this against the default loader on vLLM's
        reported weight-loading time would have had a figure on one side and
        silence on the other. Verified against vLLM 0.28.0's released source.
        """
        self._stub_vllm()
        mod = self._load_module()

        class Cfg:
            model = self.dir

        class FakeModel:
            def load_weights(self, weights):
                for _ in weights:
                    pass

        loader = mod.LmsluiceModelLoader(load_config=None)
        with self.assertLogs("lmsluice", level="INFO") as caught:
            loader.load_weights(FakeModel(), Cfg())
        self.assertTrue(
            any("Loading weights took" in line for line in caught.output),
            f"no comparable timing line was emitted: {caught.output}")
        self.assertGreaterEqual(loader.weight_load_seconds, 0.0)

    @needs_torch
    def test_load_weights_hands_the_model_a_generator_of_pairs(self):
        """The contract that matters: a generator, not a dict.

        vLLM's signature is a generator so a large checkpoint never exists
        twice. Passing a dict would satisfy the type and defeat the point.
        """
        import inspect

        self._stub_vllm()
        mod = self._load_module()

        class Cfg:
            model = self.dir

        seen = []

        class FakeModel:
            def load_weights(self, weights):
                self.kind = weights
                for name, tensor in weights:
                    seen.append((name, tensor))

        fake = FakeModel()
        mod.LmsluiceModelLoader(load_config=None).load_weights(fake, Cfg())
        self.assertTrue(inspect.isgenerator(fake.kind),
                        "a materialised container was passed, not a generator")
        self.assertEqual([n for n, _ in seen], ["w"])
        name, ten = seen[0]
        self.assertEqual(ten.dtype, torch.bfloat16)
        self.assertEqual(tuple(ten.shape), (64,))
        self.assertEqual(
            ten.detach().cpu().contiguous().view(torch.uint8)
               .reshape(-1).numpy().tobytes(), self.payload)


class TestColdMeasurementIsRestored(unittest.TestCase):
    """`uncache` is an event on Linux and a mode on macOS, and the probe has to
    survive both.

    macOS's `F_NOCACHE` is set on the descriptor: every later read bypasses the
    cache *and does not populate it*. So a "warm" rate measured straight after
    it is not warm, and a `Source` that was probed keeps bypassing the cache for
    the rest of its life. Neither shows up on Linux, where `DONTNEED` drops
    pages once and changes nothing about the descriptor — which is why this is
    tested by contract rather than by running it.
    """

    def test_a_file_source_can_restore_caching(self):
        from lmsluice.source import FileSource

        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "x.bin")
            with open(path, "wb") as fh:
                fh.write(os.urandom(1 << 20))
            src = FileSource(path)
            try:
                self.assertTrue(hasattr(src, "recache"),
                                "the probe needs a way to undo uncache")
                ok, how = src.recache()
                self.assertTrue(ok, how)
                self.assertTrue(how, "recache must say what it did")
                # Still readable afterwards, whichever platform this is.
                self.assertEqual(len(src.pread(0, 4096)), 4096)
            finally:
                src.close()
        finally:
            import shutil

            shutil.rmtree(d, ignore_errors=True)

    def test_the_probe_restores_and_primes_before_timing_warm(self):
        """Both steps, because clearing the mode is not enough on macOS.

        Every read up to that point bypassed the cache, so the region is not in
        it. Clearing `F_NOCACHE` and timing the next read would time a second
        cold read and call it warm. The probe must clear, prime untimed, then
        time.
        """
        from lmsluice import probe

        calls = []

        class FakeSource:
            name = "/fake/model.bin"
            size = 8 << 20

            def uncache(self, *a):
                calls.append("uncache")
                return True, "fake"

            def recache(self):
                calls.append("recache")
                return True, "fake"

            def pread(self, offset, length):
                calls.append("read")
                return b"\0" * length

        probe.read_rate(FakeSource(), sample=1 << 20, depths=(1,))

        self.assertIn("recache", calls, "the probe never restored caching")
        after = calls[calls.index("recache"):]
        self.assertGreaterEqual(
            after.count("read"), 2,
            "after restoring, the probe must prime the cache with an untimed "
            "read before timing the warm one")


# -- encryption at rest ----------------------------------------------------

class TestCrypt(unittest.TestCase):
    """The cipher layer on its own: keys, nonces, tags, and no backend."""

    def setUp(self):
        from lmsluice import crypt

        self.crypt = crypt
        if not crypt.available():
            self.skipTest(f"no crypto backend: {crypt.backend()[1]}")

    def test_a_raw_key_is_never_stripped(self):
        """A 32-byte key whose first or last byte is ASCII whitespace.

        `bytes.strip()` removes 0x09, 0x0a, 0x0b, 0x0c, 0x0d and 0x20, so
        reading a key file as `fh.read().strip()` silently turns roughly one
        random key in twenty into a 31-byte error. It happened on the first key
        this module ever generated, and the failure is intermittent by
        construction: the same code accepts the next key it is given.
        """
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for byte in (0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d):
            for pos in (0, 31):
                raw = bytearray(os.urandom(32))
                raw[pos] = byte
                path = os.path.join(d, "k.bin")
                with open(path, "wb") as fh:
                    fh.write(bytes(raw))
                self.assertEqual(
                    self.crypt.load_key(path), bytes(raw),
                    f"a key with {byte:#04x} at position {pos} was mangled")

    def test_a_hex_key_file_is_accepted_and_whitespace_around_it_is_not_data(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "k.hex")
        with open(path, "w") as fh:
            fh.write("  " + "ab" * 32 + "\n")
        self.assertEqual(self.crypt.load_key(path), bytes.fromhex("ab" * 32))

    def test_a_short_key_is_refused_rather_than_hashed_into_shape(self):
        """Accepting a passphrase here would give a file that looks encrypted
        and is not, which is worse than refusing."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "k.bin")
        with open(path, "wb") as fh:
            fh.write(b"hunter2")
        with self.assertRaises(self.crypt.CryptoUnavailable):
            self.crypt.load_key(path)

    def test_the_key_is_never_taken_from_argv(self):
        """argv is readable by every process on the machine out of `/proc`, so
        a flag that took a key would hand it to the box. The CLI must offer a
        path and nothing else."""
        from lmsluice import cli

        parser = cli.build_parser()
        text = "\n".join(
            [parser.format_help()]
            + [sub.format_help()
               for action in parser._subparsers._group_actions
               for sub in action.choices.values()])
        self.assertNotIn("--key ", text, "no flag may take a key directly")
        self.assertNotIn("--key=", text)
        self.assertIn("--key-file", text, "there must be a way to name one")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "lmsluice", "cli.py")) as fh:
            src = fh.read()
        self.assertNotIn('add_argument("--key"', src)

    def test_round_trip_is_exact_from_one_byte_to_eight_megabytes(self):
        key = self.crypt.subkey(self.crypt.new_key(), self.crypt.new_salt())
        for n in (0, 1, 15, 16, 17, 4096, 1 << 20):
            data = bytes((i * 7) & 0xFF for i in range(n))
            blob = self.crypt.seal(key, 3, data)
            self.assertEqual(len(blob), n + self.crypt.TAG_BYTES)
            self.assertEqual(bytes(self.crypt.open_chunk(key, 3, blob)), data)

    def test_a_flipped_bit_is_refused_and_no_plaintext_comes_back(self):
        key = self.crypt.subkey(self.crypt.new_key(), self.crypt.new_salt())
        blob = bytearray(self.crypt.seal(key, 0, b"weights" * 600))
        for pos in (0, len(blob) // 2, len(blob) - 1):
            bad = bytearray(blob)
            bad[pos] ^= 0x01
            with self.assertRaises(self.crypt.DecryptionFailed):
                self.crypt.open_chunk(key, 0, bytes(bad))

    def test_the_wrong_nonce_is_refused(self):
        """The nonce is the chunk ordinal, so reading a chunk as if it were a
        different chunk must fail rather than return rearranged bytes."""
        key = self.crypt.subkey(self.crypt.new_key(), self.crypt.new_salt())
        blob = self.crypt.seal(key, 4, b"x" * 4096)
        with self.assertRaises(self.crypt.DecryptionFailed):
            self.crypt.open_chunk(key, 5, blob)

    def test_two_archives_under_one_key_get_different_subkeys(self):
        """What makes a counter nonce sound. Chunk 0 of every archive uses the
        same nonce; only the per-archive subkey keeps that from being nonce
        reuse, which breaks GCM completely."""
        master = self.crypt.new_key()
        salts = [self.crypt.new_salt() for _ in range(32)]
        self.assertEqual(len(set(salts)), 32)
        keys = {self.crypt.subkey(master, s) for s in salts}
        self.assertEqual(len(keys), 32, "a salt did not change the subkey")
        self.assertEqual(self.crypt.nonce_for(0), self.crypt.nonce_for(0))

    def test_the_key_identifier_reveals_nothing_and_still_distinguishes(self):
        keys = [self.crypt.new_key() for _ in range(16)]
        ids = [self.crypt.key_id(k) for k in keys]
        self.assertEqual(len(set(ids)), 16)
        for k, i in zip(keys, ids):
            self.assertNotIn(k.hex()[:8], i)
            self.assertNotIn(k[:4], k[:0] + i.encode())


class TestCryptWithNoBackend(unittest.TestCase):
    """Everything still works; encryption says it cannot, and writes nothing.

    Run in a subprocess with the backend disabled, because `crypt` memoises the
    probe once per process -- which is the point of it, and means this cannot
    be tested by patching a global in this one.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(self, body):
        import subprocess

        prog = f"import sys; sys.path.insert(0, {self.ROOT!r})\n" + body
        env = dict(os.environ, LMSLUICE_NO_CRYPTO="1")
        return subprocess.run([sys.executable, "-c", prog], capture_output=True,
                              text=True, timeout=120, env=env)

    def test_the_package_imports_and_says_encryption_is_unavailable(self):
        r = self._run(
            "import lmsluice\n"
            "from lmsluice import crypt\n"
            "name, why = crypt.backend()\n"
            "assert name == 'none', name\n"
            "assert not crypt.available()\n"
            "assert 'unavailable' in crypt.describe()\n"
            "print('OK')\n")
        self.assertIn("OK", r.stdout, r.stdout + r.stderr[-600:])

    def test_it_refuses_to_write_a_file_it_cannot_protect(self):
        """The failure mode to avoid is a file that looks encrypted and is not."""
        r = self._run(
            "import tempfile, os\n"
            "from lmsluice import crypt, sealed\n"
            "d = tempfile.mkdtemp()\n"
            "src = os.path.join(d, 'x'); open(src, 'wb').write(b'0' * 64)\n"
            "dst = os.path.join(d, 'y')\n"
            "try:\n"
            "    sealed.seal_file(src, dst, None)\n"
            "    print('FAILED: it wrote something')\n"
            "except crypt.CryptoUnavailable:\n"
            "    print('OK' if not os.path.exists(dst) else 'FAILED: left a file')\n")
        self.assertIn("OK", r.stdout, r.stdout + r.stderr[-600:])

    def test_the_whole_suite_of_plain_operations_still_runs(self):
        r = self._run(
            "import tempfile, os, json, struct\n"
            "from lmsluice.zstdcodec import encoder\n"
            "from lmsluice.model import Model\n"
            "d = tempfile.mkdtemp()\n"
            "hdr = {'w': {'dtype': 'BF16', 'shape': [16],"
            " 'data_offsets': [0, 32]}}\n"
            "b = json.dumps(hdr).encode(); b += b' ' * ((8 - len(b) % 8) % 8)\n"
            "p = os.path.join(d, 'm.safetensors')\n"
            "open(p, 'wb').write(struct.pack('<Q', len(b)) + b + bytes(32))\n"
            "c = os.path.join(d, 'm.lmsl'); encoder().encode(p, c)\n"
            "with Model(c) as m:\n"
            "    assert bytes(m.load()) == open(p, 'rb').read()\n"
            "print('OK')\n")
        self.assertIn("OK", r.stdout, r.stdout + r.stderr[-600:])


class TestSealedEnvelope(unittest.TestCase):
    """The envelope over a real archive, through the real load path."""

    def setUp(self):
        from lmsluice import crypt, sealed

        self.crypt, self.sealed = crypt, sealed
        if not crypt.available():
            self.skipTest(f"no crypto backend: {crypt.backend()[1]}")
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.old_env = os.environ.get("LMSLUICE_KEY_FILE")
        self.addCleanup(self._restore_env)

        self.plain = os.path.join(self.dir, "m.safetensors")
        self.names, hdr, off, body = [f"blk.{i}" for i in range(8)], {}, 0, bytearray()
        rnd = random.Random(4)
        for n in self.names:
            nb = 8192
            body += bytes(b for _ in range(nb // 2)
                          for b in (rnd.getrandbits(8), 0x3E))
            hdr[n] = {"dtype": "BF16", "shape": [nb // 2],
                      "data_offsets": [off, off + nb]}
            off += nb
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        blob += b" " * ((8 - len(blob) % 8) % 8)
        with open(self.plain, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + bytes(body))
        with open(self.plain, "rb") as fh:
            self.truth = fh.read()

        self.key_file = os.path.join(self.dir, "key.bin")
        with open(self.key_file, "wb") as fh:
            fh.write(crypt.new_key())
        self.other_key = os.path.join(self.dir, "other.bin")
        with open(self.other_key, "wb") as fh:
            fh.write(crypt.new_key())

        from lmsluice.zstdcodec import encoder

        self.inner = os.path.join(self.dir, "m.lmsl")
        encoder().encode(self.plain, self.inner, chunk_size=8192)
        self.outer = os.path.join(self.dir, "m.sealed")
        sealed.seal_file(self.inner, self.outer, self.key_file)

    def _restore_env(self):
        if self.old_env is None:
            os.environ.pop("LMSLUICE_KEY_FILE", None)
        else:
            os.environ["LMSLUICE_KEY_FILE"] = self.old_env

    def _key(self, path):
        if path is None:
            os.environ.pop("LMSLUICE_KEY_FILE", None)
        else:
            os.environ["LMSLUICE_KEY_FILE"] = path

    def test_round_trip_is_byte_identical(self):
        from lmsluice.model import Model

        self._key(self.key_file)
        with Model(self.outer) as m:
            self.assertEqual(bytes(m.load()), self.truth)

    def test_stream_returns_every_tensor_unchanged(self):
        """`stream()` and `load()` are different code paths -- one windows, the
        other gathers -- so a decrypting source has to satisfy both."""
        from lmsluice.model import Model

        self._key(self.key_file)
        with Model(self.outer) as m:
            seen = 0
            for name, mv in m.stream():
                t = m.tensors[name]
                self.assertEqual(bytes(mv), self.truth[t.start:t.end], name)
                seen += 1
            self.assertEqual(seen, len(self.names))

    def test_a_named_subset_reads_only_what_it_asked_for(self):
        from lmsluice.model import Model

        self._key(self.key_file)
        want = self.names[2:4]
        with Model(self.outer) as m:
            buf = bytes(m.load(want))
            base = m.base
            for n in want:
                t = m.tensors[n]
                self.assertEqual(buf[t.start - base:t.end - base],
                                 self.truth[t.start:t.end], n)
            # The view spans the member by contract, so the evidence that only
            # part of it was decoded is that the rest is still untouched: a
            # tensor at the far end, in no block that any requested tensor
            # touches, must not have been decrypted into the buffer.
            far = m.tensors[self.names[-1]]
            self.assertNotEqual(buf[far.start - base:far.end - base],
                                self.truth[far.start:far.end],
                                "the whole archive was decoded for two tensors")

    def test_the_structure_reads_without_a_key_and_the_weights_do_not(self):
        """The trade this design makes, and it must hold in both directions:
        `info` works for someone who cannot decrypt, and `load` does not."""
        from lmsluice.model import Model

        self._key(None)
        with Model(self.outer) as m:
            self.assertEqual(sorted(m.tensors), sorted(self.names))
            self.assertEqual(m.route, "coded")
            self.assertIsNotNone(m.encrypted)
            self.assertFalse(m._arc.source.verified)
            with self.assertRaises(self.crypt.CryptoUnavailable):
                m.load()

    def test_a_sealed_archive_is_never_mappable(self):
        """There is no arrangement of page tables that decrypts, so the plain
        route cannot serve this file however fast the disk is."""
        from lmsluice.model import Model, NotMappable

        self._key(self.key_file)
        with Model(self.outer) as m:
            with self.assertRaises(NotMappable):
                m.map()

    def test_the_wrong_key_is_refused_when_the_file_is_opened(self):
        from lmsluice.model import Model

        self._key(self.other_key)
        with self.assertRaises(ValueError) as caught:
            Model(self.outer).close()
        self.assertIn("different key", str(caught.exception))
        self.assertIn("Nothing was decrypted", str(caught.exception))

    def _tamper(self, pos, name):
        with open(self.outer, "rb") as fh:
            raw = bytearray(fh.read())
        raw[pos] ^= 0x01
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(bytes(raw))
        return path

    def test_a_tampered_payload_is_refused_by_the_tag_and_the_unit_is_named(self):
        from lmsluice.model import Model
        from lmsluice.source import open_source

        self._key(self.key_file)
        with self.sealed.SealedSource(open_source(self.outer)) as src:
            rows = [r for r in src._rows if r[4] == self.sealed.SEALED]
        self.assertGreaterEqual(len(rows), 2, "need a sealed extent to hit")
        path = self._tamper(rows[1][2] + 3, "tampered.lmsl")
        with self.assertRaises(ValueError) as caught:
            with Model(path) as m:
                m.load()
        text = str(caught.exception)
        self.assertIn("failed authentication", text)
        self.assertIn("unit", text, "the failing unit must be named")

    def test_a_rewritten_index_is_refused_by_the_mac(self):
        """The attack the payload tags cannot see. Every byte here is a valid
        archive: the index decompresses, the extents are well formed, and the
        ciphertext is untouched -- only which ciphertext each unit points at
        has changed. Without the MAC an attacker who cannot read a weight can
        still decide which weight gets loaded."""
        from lmsluice.model import Model

        self._key(self.key_file)
        with open(self.outer, "rb") as fh:
            raw = bytearray(fh.read())
        ioff, ilen, _ = self.sealed.FOOTER.unpack(raw[-self.sealed.FOOTER.size:])
        index = json.loads(zlib.decompress(bytes(raw[ioff:ioff + ilen])))
        i, j = [k for k, r in enumerate(index["extents"])
                if r[4] == self.sealed.SEALED][:2]
        for f in (2, 3):
            index["extents"][i][f], index["extents"][j][f] = \
                index["extents"][j][f], index["extents"][i][f]
        blob = zlib.compress(json.dumps(index, separators=(",", ":")).encode(), 6)
        path = os.path.join(self.dir, "rewritten.lmsl")
        with open(path, "wb") as fh:
            fh.write(bytes(raw[:ioff]) + blob
                     + self.sealed.FOOTER.pack(ioff, len(blob), self.sealed.TAIL))
        with self.assertRaises(ValueError) as caught:
            Model(path).close()
        self.assertIn("structure failed authentication", str(caught.exception))

    def test_a_tampered_clear_region_is_refused_by_the_mac(self):
        from lmsluice.model import Model
        from lmsluice.source import open_source

        self._key(self.key_file)
        with self.sealed.SealedSource(open_source(self.outer)) as src:
            clear = [r for r in src._rows if r[4] == self.sealed.CLEAR and r[3]]
        self.assertTrue(clear, "the container should have a header or an index")
        path = self._tamper(clear[-1][2], "clear.lmsl")
        with self.assertRaises(ValueError) as caught:
            Model(path).close()
        self.assertIn("structure failed authentication", str(caught.exception))

    def test_the_key_never_reaches_the_archive(self):
        with open(self.key_file, "rb") as fh:
            key = fh.read()
        with open(self.outer, "rb") as fh:
            whole = fh.read()
        self.assertNotIn(key, whole)
        self.assertNotIn(key.hex().encode(), whole)
        self.assertNotIn(key.hex().upper().encode(), whole)
        # Nor into the header that is meant to be public.
        from lmsluice.source import open_source

        with self.sealed.SealedSource(open_source(self.outer)) as src:
            text = json.dumps(src.encryption).encode()
        self.assertNotIn(key, text)
        self.assertNotIn(key.hex().encode(), text)

    def test_two_archives_of_the_same_file_under_one_key_differ(self):
        """Same input, same key, different bytes -- because the salt is fresh.
        Identical ciphertext would leak that two files are the same file."""
        a = os.path.join(self.dir, "a.sealed")
        b = os.path.join(self.dir, "b.sealed")
        self.sealed.seal_file(self.inner, a, self.key_file)
        self.sealed.seal_file(self.inner, b, self.key_file)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            self.assertNotEqual(fa.read(), fb.read())
        from lmsluice.source import open_source

        with self.sealed.SealedSource(open_source(a), key_file=self.key_file) as x, \
             self.sealed.SealedSource(open_source(b), key_file=self.key_file) as y:
            self.assertNotEqual(x.encryption["salt"], y.encryption["salt"])
            self.assertEqual(x.encryption["key_id"], y.encryption["key_id"])
            self.assertNotEqual(x._key, y._key)

    def test_a_sealed_archive_asks_for_a_fetch_depth_chosen_for_computing(self):
        """The fetch pool decrypts, so it is no longer a stage that only waits.
        Measured on the development box, the default of 16 was the worst
        setting available; see `probe.SEALED_FETCH`."""
        from lmsluice.model import Model
        from lmsluice.probe import SEALED_FETCH, default_threads

        self._key(self.key_file)
        with Model(self.outer) as sealed_model, Model(self.inner) as plain_model:
            self.assertEqual(sealed_model._fetch_threads, SEALED_FETCH)
            self.assertEqual(plain_model._fetch_threads, default_threads()[0])
        with Model(self.outer, fetch_threads=7) as m:
            self.assertEqual(m._fetch_threads, 7, "a caller may still override")

    def test_writing_a_sealed_archive_is_priced_as_two_passes(self):
        """The cost of not editing lmz. The envelope wraps a finished
        container, so the archive is written and then read back and rewritten
        sealed: two terms a 70 GB checkpoint should see before it starts, not
        after. Wall clock as a second phase, disk as a high-water mark."""
        from lmsluice.plan import seal_rate, write_plan

        GB = 1_000_000_000
        rate = seal_rate(source=6.34 * GB, sink=2.0 * GB, encrypt=18.0 * GB)
        self.assertLess(rate, 2.0 * GB, "a serial loop cannot beat its slowest part")
        d = write_plan(70 * GB, 0.673, sink=2.0 * GB, encode=0.9 * GB, seal=rate)
        by = {r.name: r for r in d.routes}
        self.assertIn("sealed", by)
        # The second pass is charged, not absorbed into the first.
        self.assertGreater(by["sealed"].seconds, by["coded"].seconds)
        # And the first pass's own overlap is still credited: charging every
        # stage serially would make it encode + write + seal.
        serial = sum(s.seconds for s in by["sealed"].stages)
        self.assertLess(by["sealed"].seconds, serial)
        # Traffic is the same; disk is not.
        self.assertEqual(by["sealed"].device_bytes, by["coded"].device_bytes)
        self.assertEqual(by["sealed"].peak_bytes, 2 * by["coded"].peak_bytes)
        self.assertIn("GB on disk", d.explain())

    def test_an_unsealed_write_plan_is_exactly_what_it_always_was(self):
        """Phases and a disk column must cost nothing where nothing seals."""
        from lmsluice.plan import write_plan

        GB = 1_000_000_000
        d = write_plan(70 * GB, 0.673, sink=2.0 * GB, encode=0.9 * GB)
        self.assertEqual([r.name for r in d.routes], ["plain", "coded"])
        for r in d.routes:
            self.assertEqual(r.peak_bytes, r.device_bytes)
        self.assertNotIn("GB on disk", d.explain())

    def test_the_seal_rate_is_derived_from_the_machine_and_not_a_constant(self):
        """`seal_rate` re-evaluates elsewhere without re-measuring: a slow disk
        with a fast cipher and a fast disk with a phone-class one must give
        different answers, and a missing term must give none at all."""
        from lmsluice.plan import seal_rate

        GB = 1_000_000_000
        fast_disk = seal_rate(source=6.34 * GB, sink=6.0 * GB, encrypt=0.3 * GB)
        slow_disk = seal_rate(source=0.5 * GB, sink=0.2 * GB, encrypt=18.0 * GB)
        self.assertLess(fast_disk, 0.3 * GB, "the cipher binds when it is slow")
        self.assertLess(slow_disk, 0.2 * GB, "the disk binds when it is slow")
        self.assertNotAlmostEqual(fast_disk, slow_disk)
        self.assertIsNone(seal_rate(source=None, sink=1 * GB, encrypt=1 * GB))
        self.assertIsNone(seal_rate(source=1 * GB, sink=None, encrypt=1 * GB))
        self.assertIsNone(seal_rate(source=1 * GB, sink=1 * GB, encrypt=None))

    def test_the_plan_can_name_decryption_as_the_binding_stage(self):
        """A rate model that cannot express a stage cannot be told the stage is
        the problem. Decryption runs on the fetch pool, so it is serial with
        the read and parallel with the decode."""
        from lmsluice.plan import read_plan

        GB = 1_000_000_000
        common = dict(source=6.34 * GB, decode=5.95 * GB)
        fast = read_plan(10 * GB, 6.7 * GB, decrypt=40 * GB, **common)
        slow = read_plan(10 * GB, 6.7 * GB, decrypt=0.4 * GB, **common)
        coded = lambda d: [r for r in d.routes if r.name == "coded"][0]
        self.assertNotEqual(coded(fast).limited_by, "decrypt")
        self.assertEqual(coded(slow).limited_by, "decrypt")
        self.assertGreater(coded(slow).seconds, coded(fast).seconds)
        # And an unsealed archive is priced exactly as it was before.
        none = read_plan(10 * GB, 6.7 * GB, **common)
        self.assertEqual(len(coded(none).stages), 2)


class TestEncryptionStaysOutOfTheCodecs(unittest.TestCase):
    """`docs/boundary.md`, applied to this feature.

    Encryption changes when the threat model changes, not when the format
    does, so it belongs to lmsluice and no codec may mention it. The practical
    half of the same rule: `LmzEncoder.encode` hands the whole job to lmz,
    which writes its own container, so a codec-level option could only ever
    have encrypted the fallback codec.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ALLOWED = {"crypt.py", "sealed.py", "cli.py"}

    def sources(self):
        d = os.path.join(self.ROOT, "lmsluice")
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                with open(os.path.join(d, name)) as fh:
                    yield name, fh.read()

    def test_only_crypt_reaches_a_cipher_library(self):
        offenders = []
        for name, src in self.sources():
            if name == "crypt.py":
                continue
            for line in src.splitlines():
                s = line.strip()
                if s.startswith("#"):
                    continue
                if (s.startswith(("import cryptography", "from cryptography"))
                        or "libcrypto" in s and s.startswith(("import", "from"))
                        or s.startswith("from ctypes.util") and "crypto" in s):
                    offenders.append(f"{name}: {s}")
        self.assertEqual(offenders, [],
                         "only lmsluice/crypt.py may reach a cipher library")

    def test_no_codec_module_mentions_encryption(self):
        offenders = []
        for name, src in self.sources():
            if name in self.ALLOWED or "codec" not in name:
                continue
            body = "\n".join(l for l in src.splitlines()
                              if not l.strip().startswith("#"))
            if body.count('"""') >= 2:
                body = body.split('"""', 2)[2]
            for word in ("encrypt", "decrypt", "AES", "GCM", "key_file",
                         "nonce", "cipher"):
                if word in body:
                    offenders.append(f"{name}: {word}")
        self.assertEqual(offenders, [],
                         "a codec must not know that encryption exists -- "
                         "see docs/boundary.md and lmsluice/sealed.py")

    def test_no_module_ever_logs_or_prints_key_material(self):
        """A key that reaches a log has left the machine's memory and entered
        its disk, its journal and whatever ships those elsewhere. The variables
        holding one are named here so a `print` or a `log` of them is a
        grep-able mistake rather than a silent one."""
        # `key` means two different things in this tree -- a storage key in
        # `cli.py` and `rates.py`, key material in the two crypto modules -- so
        # the ambiguous names are checked only where they hold bytes, and the
        # unambiguous ones everywhere.
        anywhere = ("master", "subkey", "mac_key")
        in_crypto = ("key", "raw", "self._key")
        offenders = []
        for name, src in self.sources():
            held = anywhere + (in_crypto if name in ("crypt.py", "sealed.py")
                               else ())
            for i, line in enumerate(src.splitlines(), 1):
                s = line.strip()
                if s.startswith("#") or not (
                        s.startswith(("print(", "log.", "logger.", "warnings."))
                        or ".info(" in s or ".debug(" in s or ".warning(" in s):
                    continue
                for var in held:
                    if f"{{{var}}}" in s or f", {var})" in s or f"({var})" in s:
                        offenders.append(f"{name}:{i}: {s}")
        self.assertEqual(offenders, [],
                         "a key, a subkey or a raw key file must never be "
                         "printed or logged")

    def test_an_archive_is_portable_between_backends(self):
        """The backend named in the header is informational, not a
        requirement: AES-256-GCM is AES-256-GCM. A file sealed where libcrypto
        was reachable has to open where only the extra is, or the header would
        be pinning archives to whichever machine wrote them."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            self.skipTest("the `cryptography` extra is not installed here")
        from lmsluice import crypt

        if crypt.backend()[0] != "openssl":
            self.skipTest("this process is not running the openssl backend")
        key = crypt.subkey(crypt.new_key(), crypt.new_salt())
        for n in (0, 1, 4096, 1 << 20):
            data = bytes((i * 5) & 0xFF for i in range(n))
            nonce = crypt.nonce_for(7)
            # openssl seals, the extra opens.
            self.assertEqual(
                AESGCM(key).decrypt(nonce, crypt.seal(key, 7, data), None), data)
            # the extra seals, openssl opens.
            self.assertEqual(
                bytes(crypt.open_chunk(
                    key, 7, AESGCM(key).encrypt(nonce, data, None))), data)

    def test_crypt_imports_nothing_that_is_not_the_standard_library(self):
        with open(os.path.join(self.ROOT, "lmsluice", "crypt.py")) as fh:
            src = fh.read()
        stdlib = {"ctypes", "hashlib", "hmac", "os", "secrets", "threading",
                  "ssl", "ctypes.util", "__future__"}
        bad = []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("import ") and not s.startswith("import ."):
                mod = s[len("import "):].split()[0].split(".")[0]
                if mod not in stdlib and mod != "cryptography":
                    bad.append(s)
        self.assertEqual(bad, [], "crypt.py must need nothing installed")


# -- object storage --------------------------------------------------------

class TestSigning(unittest.TestCase):
    """Known-answer tests against each vendor's own published example.

    These are the tests that matter for signing, because a signature is either
    byte-exact or worthless and nothing in between is observable: a wrong
    canonical form produces a well-formed header that is always rejected, with
    no clue as to which of a dozen rules was misread. A round trip against a
    fake I also wrote would only prove I was consistent with myself.
    """

    AK = "AKIAIOSFODNN7EXAMPLE"
    SK = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    WHEN = DT.datetime(2013, 5, 24, tzinfo=DT.timezone.utc)

    def sig(self, url, headers=None):
        h = SIGN.sigv4_headers(
            method="GET", url=url, region="us-east-1", service="s3",
            access_key=self.AK, secret_key=self.SK, headers=headers,
            when=self.WHEN)
        return h["Authorization"].rsplit("Signature=", 1)[1]

    def test_aws_get_object_with_a_range(self):
        """AWS, "Signature Calculations -- Transferring Payload in a Single
        Chunk", Example: GET Object. This is the exact request shape this
        package sends on every read."""
        self.assertEqual(
            self.sig("https://examplebucket.s3.amazonaws.com/test.txt",
                     {"Range": "bytes=0-9"}),
            "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")

    def test_aws_query_parameter_with_an_empty_value(self):
        """`?lifecycle` signs as `lifecycle=`. The trailing `=` is easy to omit
        and produces a signature that is wrong only for URLs with a valueless
        parameter, which is a bug that hides until it does not."""
        self.assertEqual(
            self.sig("https://examplebucket.s3.amazonaws.com/?lifecycle"),
            "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543")

    def test_aws_query_parameters_are_sorted(self):
        self.assertEqual(
            self.sig("https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J"),
            "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")

    def test_s3_paths_are_not_normalised(self):
        """`a//b`, `a/./b` and `a/b` are three different object keys. A
        signature over a tidied path authorises a request for a different
        object, so S3's rule is to leave the path alone -- and the generic
        rule, which does normalise, must still be available for the services
        that use it."""
        raw = SIGN.canonical_uri("/x/./y//z")
        self.assertEqual(raw, "/x/./y//z")
        self.assertEqual(SIGN.canonical_uri("/x/./y//z", normalise=True),
                         "/x/y//z")
        self.assertEqual(SIGN.canonical_uri("/a b+c"), "/a%20b%2Bc")

    def test_azure_string_to_sign_matches_the_documented_example(self):
        """Microsoft, "Authorize with Shared Key". The HMAC is `hmac` and
        cannot be wrong; the canonical form is the part that can, so that is
        what is pinned -- byte for byte, including the twelve positional lines
        and the query parameters appended lowercased and sorted."""
        self.assertEqual(
            SIGN.azure_string_to_sign(
                "GET", "myaccount", "/mycontainer",
                "restype=container&comp=list&timeout=20",
                {"x-ms-date": "Fri, 26 Jun 2015 23:39:12 GMT",
                 "x-ms-version": "2015-02-21"}),
            "GET\n\n\n\n\n\n\n\n\n\n\n\n"
            "x-ms-date:Fri, 26 Jun 2015 23:39:12 GMT\n"
            "x-ms-version:2015-02-21\n"
            "/myaccount/mycontainer\ncomp:list\nrestype:container\ntimeout:20")

    def test_azure_content_length_zero_signs_as_empty(self):
        """Changed in API version 2015-02-21 and the single most common cause
        of a 403 against a correct key."""
        s = SIGN.azure_string_to_sign("GET", "a", "/c/b", "",
                                      {"Content-Length": "0"})
        self.assertEqual(s.split("\n")[3], "")

    def test_azure_repeated_query_parameters_join_sorted_with_commas(self):
        s = SIGN.canonical_azure_resource("a", "/c", "k=2&k=1&j=x")
        self.assertEqual(s, "/a/c\nj:x\nk:1,2")

    def test_the_azure_key_is_decoded_before_it_keys_the_hmac(self):
        """Azure publishes the account key as base64 everywhere. Signing with
        the base64 *text* gives a well-formed header that is always rejected,
        which reads as a wrong key rather than as a bug."""
        import base64 as b64
        import hashlib
        import hmac as _hmac

        raw = bytes(range(32))
        key = b64.b64encode(raw).decode()
        h = SIGN.shared_key_header(
            method="GET", url="https://a.blob.core.windows.net/c/b",
            account="a", key=key,
            when=DT.datetime(2020, 1, 1, tzinfo=DT.timezone.utc))
        to_sign = SIGN.azure_string_to_sign(
            "GET", "a", "/c/b", "",
            {k: v for k, v in h.items() if k.lower().startswith("x-ms-")})
        want = b64.b64encode(_hmac.new(raw, to_sign.encode(),
                                       hashlib.sha256).digest()).decode()
        self.assertEqual(h["Authorization"], f"SharedKey a:{want}")

    def test_a_session_token_is_signed_and_not_merely_sent(self):
        """A temporary credential's token is part of the canonical request. A
        client that sends it unsigned works against nothing."""
        h = SIGN.sigv4_headers(
            method="GET", url="https://b.s3.amazonaws.com/k", region="us-east-1",
            access_key=self.AK, secret_key=self.SK, token="TOKEN",
            when=self.WHEN)
        self.assertIn("x-amz-security-token", h["Authorization"])
        self.assertEqual(h["x-amz-security-token"], "TOKEN")


class TestCloudSources(unittest.TestCase):
    """The three stores, against a fake that re-derives every signature."""

    KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    AZKEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.data = os.urandom(200_000)
        with open(os.path.join(self.dir, "blob.bin"), "wb") as fh:
            fh.write(self.data)
        self.saved = {k: os.environ.get(k) for k in (
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
            "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
            "LMSLUICE_S3_ENDPOINT", "LMSLUICE_GCS_ENDPOINT",
            "LMSLUICE_AZURE_ENDPOINT", "AZURE_STORAGE_ACCOUNT",
            "AZURE_STORAGE_KEY", "AZURE_STORAGE_SAS_TOKEN",
            "AZURE_STORAGE_CONNECTION_STRING", "GCS_HMAC_ACCESS_KEY",
            "GCS_HMAC_SECRET", "GOOGLE_OAUTH_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS")}
        for k in self.saved:
            os.environ.pop(k, None)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_s3_reads_ranges_through_a_signature_checking_store(self):
        from lmsluice.source import open_source

        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        with _fake_store(self.dir, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with open_source("s3://bucket/blob.bin") as s:
                self.assertEqual(s.size, len(self.data))
                self.assertTrue(s.random_access)
                self.assertEqual(s.pread(4096, 128), self.data[4096:4224])
                self.assertEqual(s.pread(0, 64), self.data[:64])
                self.assertEqual(s.cloud, "s3")
                self.assertEqual(s.mode, "stdlib")
                self.assertFalse(s.anonymous)

    def test_gcs_hmac_is_the_same_signature_at_a_different_host(self):
        """The finding that made GCS cost nothing: its XML API accepts SigV4,
        so an HMAC key pair is the S3 path with the host changed."""
        from lmsluice.source import open_source

        os.environ.update(GCS_HMAC_ACCESS_KEY="AK", GCS_HMAC_SECRET=self.KEY)
        with _fake_store(self.dir, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_GCS_ENDPOINT"] = base
            with open_source("gs://bucket/blob.bin") as s:
                self.assertEqual(s.pread(1024, 256), self.data[1024:1280])
                self.assertEqual(s.cloud, "gcs")
                self.assertEqual(s.auth, "hmac")

    def test_azure_shared_key_and_the_x_ms_range_header(self):
        from lmsluice.source import open_source

        os.environ.update(AZURE_STORAGE_ACCOUNT="acct",
                          AZURE_STORAGE_KEY=self.AZKEY)
        store = _fake_store(self.dir, "azure", key=self.AZKEY, account="acct")
        with store as base:
            os.environ["LMSLUICE_AZURE_ENDPOINT"] = base
            with open_source("az://acct/cont/blob.bin") as s:
                self.assertEqual(s.pread(99, 321), self.data[99:420])
                self.assertEqual(s.cloud, "azure")
                self.assertEqual(s.auth, "shared-key")
        # It must use Azure's own range header, not the HTTP one.
        ranges = [h for h in store.seen if "x-ms-range" in h]
        self.assertTrue(ranges, "no request carried x-ms-range")
        self.assertFalse(any("range" in h and "x-ms-range" not in h
                             for h in store.seen))

    def test_a_public_bucket_needs_nothing_configured(self):
        """The case a first-time reader is most likely to try. No credentials
        in the environment must mean an unsigned request, not an error."""
        from lmsluice.source import open_source

        with _fake_store(self.dir, "s3", require_auth=False) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with open_source("s3://bucket/blob.bin") as s:
                self.assertTrue(s.anonymous)
                self.assertEqual(s.pread(10, 20), self.data[10:30])

    def test_an_empty_object_opens_as_empty_rather_than_unreachable(self):
        """Found against real Azure, on the first blob a public container
        listed. A one-byte probe on a zero-length object is a 416, which is
        indistinguishable from a failure unless it is handled -- so an object
        that exists and is simply empty was being reported as unreachable.
        Spark leaves a zero-byte `_SUCCESS` marker in every output directory,
        so this is common rather than exotic."""
        from lmsluice.source import open_source

        with open(os.path.join(self.dir, "empty.bin"), "wb"):
            pass
        with _fake_store(self.dir, "s3", require_auth=False) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with open_source("s3://bucket/empty.bin") as s:
                self.assertEqual(s.size, 0)
                self.assertTrue(s.random_access)
                self.assertEqual(s.pread(0, 0), b"")

    def test_the_other_empty_object_reply_is_handled_too(self):
        """Stores disagree about an unsatisfiable range on a zero-length
        object: the fake above answers 206 with an empty body, real Azure
        answers 416. Both mean empty, and only one of them was handled when
        this was first written."""
        import urllib.error

        from lmsluice.source import HttpSource

        class Probe(HttpSource):
            def __init__(self):
                self.name = self.url = "https://example.invalid/empty"
                self.timeout = 5
                self.headers = {}
                self._signer = None
                self.range_header = "Range"
                self._local = threading.local()
                self.size, self.random_access = self._head()

        real = urllib.request.urlopen

        def boom(*a, **k):
            raise urllib.error.HTTPError(
                "https://example.invalid/empty", 416, "Range Not Satisfiable",
                {}, None)

        urllib.request.urlopen = boom
        try:
            p = Probe()
        finally:
            urllib.request.urlopen = real
        self.assertEqual(p.size, 0)
        self.assertTrue(p.random_access)
        self.assertIsNone(p.head_error)

    def test_a_wrong_signature_is_refused_by_the_store(self):
        """Proves the fake is actually checking, which is what makes every
        other test in this class mean something."""
        from lmsluice.cloud import CloudError
        from lmsluice.source import open_source

        os.environ.update(AWS_ACCESS_KEY_ID="AK",
                          AWS_SECRET_ACCESS_KEY="not-the-right-secret",
                          AWS_REGION="us-east-1")
        with _fake_store(self.dir, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with self.assertRaises((OSError, CloudError)):
                open_source("s3://bucket/blob.bin").pread(0, 16)

    def test_aws_credentials_come_from_the_profile_file(self):
        from lmsluice.cloud import aws_credentials

        path = os.path.join(self.dir, "creds")
        with open(path, "w") as fh:
            fh.write("[default]\naws_access_key_id = D\n"
                     "aws_secret_access_key = ds\n\n"
                     "[work]\naws_access_key_id = W\n"
                     "aws_secret_access_key = ws\nregion = eu-west-2\n")
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = path
        self.assertEqual(aws_credentials()["access_key"], "D")
        self.assertEqual(aws_credentials("work")["access_key"], "W")
        self.assertEqual(aws_credentials("work")["region"], "eu-west-2")
        os.environ["AWS_PROFILE"] = "work"
        self.assertEqual(aws_credentials()["access_key"], "W")

    def test_the_whole_pipeline_reads_a_model_from_a_bucket(self):
        """The test of the abstraction rather than a claim about it: `Model`,
        `load` and `stream` were never told what a source is, so a bucket URL
        must work with no code path of its own."""
        from lmsluice.model import Model
        from lmsluice.zstdcodec import encoder

        plain = os.path.join(self.dir, "m.safetensors")
        hdr, off, body = {}, 0, bytearray()
        rnd = random.Random(12)
        for i in range(6):
            nb = 8192
            body += bytes(b for _ in range(nb // 2)
                          for b in (rnd.getrandbits(8), 0x3E))
            hdr[f"t{i}"] = {"dtype": "BF16", "shape": [nb // 2],
                            "data_offsets": [off, off + nb]}
            off += nb
        blob = json.dumps(hdr, separators=(",", ":")).encode()
        blob += b" " * ((8 - len(blob) % 8) % 8)
        with open(plain, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)) + blob + bytes(body))
        with open(plain, "rb") as fh:
            truth = fh.read()
        encoder().encode(plain, os.path.join(self.dir, "m.lmsl"),
                         chunk_size=8192)

        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        with _fake_store(self.dir, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with Model("s3://bucket/m.lmsl") as m:
                self.assertEqual(m.route, "coded")
                self.assertEqual(bytes(m.load()), truth)
                for name, mv in m.stream():
                    t = m.tensors[name]
                    self.assertEqual(bytes(mv), truth[t.start:t.end], name)


class TestCloudWrites(unittest.TestCase):
    """Uploads, against a fake that checks the signature *and* the body hash.

    Checking the body hash is what makes these tests mean something. SigV4
    signs the SHA-256 of the payload, and a client that signs the empty-payload
    hash on every request -- the value every *read* correctly uses -- produces
    a request that a lax fake accepts and a real store rejects.
    """

    KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    AZKEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        self.saved = dict(os.environ)
        self.addCleanup(self._restore)
        for k in list(os.environ):
            if k.startswith(("AWS_", "AZURE_", "GCS_", "GOOGLE_", "LMSLUICE_")):
                os.environ.pop(k, None)

    def _restore(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def _fixture(self, size):
        path = os.path.join(self.dir, "up.bin")
        with open(path, "wb") as fh:
            fh.write(os.urandom(size))
        with open(path, "rb") as fh:
            return path, fh.read()

    def test_a_small_object_goes_in_one_request(self):
        from lmsluice.cloud import put

        path, data = self._fixture(64_000)
        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        with _fake_store(self.out, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            report = put(path, "s3://bucket/up.bin")
        self.assertEqual(report["method"], "single PUT")
        self.assertEqual(report["parts"], 1)
        with open(os.path.join(self.out, "up.bin"), "rb") as fh:
            self.assertEqual(fh.read(), data)

    def test_a_large_object_goes_in_parts_and_arrives_in_order(self):
        from lmsluice.cloud import put

        path, data = self._fixture(300_000)
        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        with _fake_store(self.out, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            report = put(path, "s3://bucket/up.bin", part_bytes=64_000)
        self.assertEqual(report["method"], "multipart")
        self.assertEqual(report["parts"], 5)
        with open(os.path.join(self.out, "up.bin"), "rb") as fh:
            self.assertEqual(fh.read(), data, "parts were assembled wrong")

    def test_azure_stages_blocks_and_commits_a_block_list(self):
        from lmsluice.cloud import put

        path, data = self._fixture(300_000)
        os.environ.update(AZURE_STORAGE_ACCOUNT="acct",
                          AZURE_STORAGE_KEY=self.AZKEY)
        with _fake_store(self.out, "azure", key=self.AZKEY,
                         account="acct") as base:
            os.environ["LMSLUICE_AZURE_ENDPOINT"] = base
            report = put(path, "az://acct/cont/up.bin", part_bytes=64_000)
        self.assertEqual(report["method"], "block blob")
        self.assertEqual(report["parts"], 5)
        with open(os.path.join(self.out, "up.bin"), "rb") as fh:
            self.assertEqual(fh.read(), data)

    def test_gcs_uploads_over_the_same_multipart_as_s3(self):
        from lmsluice.cloud import put

        path, data = self._fixture(200_000)
        os.environ.update(GCS_HMAC_ACCESS_KEY="AK", GCS_HMAC_SECRET=self.KEY)
        with _fake_store(self.out, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_GCS_ENDPOINT"] = base
            put(path, "gs://bucket/up.bin", part_bytes=64_000)
        with open(os.path.join(self.out, "up.bin"), "rb") as fh:
            self.assertEqual(fh.read(), data)

    def test_a_failed_multipart_is_cancelled_rather_than_left_billing(self):
        """Parts of an abandoned upload are stored, billed, and absent from a
        listing. Leaving them is a charge the user cannot find the cause of, so
        a failure part way through must abort the upload on its way out."""
        from lmsluice.cloud import CloudError, put

        path, _ = self._fixture(300_000)
        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        store = _fake_store(self.out, "s3", key=self.KEY)
        store.fail_part = 3
        with store as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            with self.assertRaises(CloudError) as caught:
                put(path, "s3://bucket/up.bin", part_bytes=64_000)
            self.assertIn("UploadPart 3", str(caught.exception))
            self.assertEqual(store.uploads, {},
                             "an abandoned multipart was left on the store")

    def test_writing_without_credentials_is_refused_before_anything_is_sent(self):
        from lmsluice.cloud import CloudError, put

        path, _ = self._fixture(1024)
        with self.assertRaises(CloudError) as caught:
            put(path, "s3://bucket/up.bin")
        self.assertIn("credentials", str(caught.exception))

    def test_a_round_trip_through_a_bucket_is_byte_identical(self):
        """The pair that matters: written by `put`, read back by the ordinary
        source, and compared to what went in."""
        from lmsluice.cloud import put
        from lmsluice.source import open_source

        path, data = self._fixture(250_000)
        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=self.KEY,
                          AWS_REGION="us-east-1")
        with _fake_store(self.out, "s3", key=self.KEY) as base:
            os.environ["LMSLUICE_S3_ENDPOINT"] = base
            put(path, "s3://bucket/up.bin", part_bytes=64_000)
            with open_source("s3://bucket/up.bin") as s:
                self.assertEqual(s.size, len(data))
                self.assertEqual(s.pread(0, s.size), data)


class TestCredentialsNeverEscape(unittest.TestCase):
    """The rule `crypt.py` set, applied to a second kind of secret."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_no_flag_takes_a_credential(self):
        """argv is readable by every process on the machine out of `/proc`, and
        it is in shell history everywhere."""
        from lmsluice import cli

        parser = cli.build_parser()
        text = "\n".join(
            [parser.format_help()]
            + [sub.format_help()
               for action in parser._subparsers._group_actions
               for sub in action.choices.values()])
        for flag in ("--secret", "--secret-key", "--password", "--access-key",
                     "--account-key", "--sas", "--token"):
            self.assertNotIn(flag, text, f"{flag} would put a secret in argv")

    def test_a_signature_is_redacted_from_an_error(self):
        from lmsluice.cloud import redact

        text = ("GET failed: Authorization: AWS4-HMAC-SHA256 Credential=AK/x, "
                "SignedHeaders=host, Signature=deadbeefcafe\n"
                "x-amz-security-token: SESSIONTOKEN\nHost: example")
        out = redact(text)
        self.assertNotIn("deadbeefcafe", out)
        self.assertNotIn("SESSIONTOKEN", out)
        self.assertIn("Host: example", out, "redaction must not eat the rest")

    def test_a_sas_token_in_a_url_is_redacted(self):
        """A SAS *is* a signature, carried in the query string -- so a URL can
        itself be a credential, and a URL is exactly what error messages print."""
        from lmsluice.cloud import redact

        url = ("https://a.blob.core.windows.net/c/b?sv=2021-08-06&"
               "sig=abc%2Fdef%2Bghi%3D&se=2026-01-01")
        out = redact(f"could not read {url}")
        self.assertNotIn("abc%2Fdef", out)
        self.assertIn("sv=2021-08-06", out, "only the signature is secret")

    def test_a_live_credential_is_stripped_even_when_the_name_is_unknown(self):
        from lmsluice.cloud import redact_env

        secret = "aVerySecretKeyValue1234567890"
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        try:
            self.assertNotIn(secret, redact_env(f"boom: {secret} in a message"))
        finally:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    def test_opening_a_bad_bucket_url_does_not_leak_the_environment(self):
        from lmsluice.cloud import open_cloud

        secret = "SECRETSECRETSECRET123456"
        os.environ.update(AWS_ACCESS_KEY_ID="AK", AWS_SECRET_ACCESS_KEY=secret,
                          LMSLUICE_S3_ENDPOINT="http://127.0.0.1:1")
        try:
            with self.assertRaises(Exception) as caught:
                open_cloud("s3://bucket/nope")
            self.assertNotIn(secret, str(caught.exception))
        finally:
            for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                      "LMSLUICE_S3_ENDPOINT"):
                os.environ.pop(k, None)

    def test_no_cloud_module_imports_an_sdk_at_module_scope(self):
        """The `no install required` row, held the way the torch and lmz rows
        are held: by walking the source, because one stray import retires a
        published claim silently."""
        d = os.path.join(self.ROOT, "lmsluice")
        offenders = []
        for name in sorted(os.listdir(d)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(d, name)) as fh:
                src = fh.read()
            for line in src.splitlines():
                s = line.strip()
                if s.startswith("#"):
                    continue
                for sdk in ("boto3", "botocore", "google.cloud", "google.auth",
                            "azure.storage", "azure.identity", "requests"):
                    if s.startswith((f"import {sdk}", f"from {sdk}")):
                        offenders.append(f"{name}: {s}")
        self.assertEqual(offenders, [],
                         "object storage must need nothing installed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
