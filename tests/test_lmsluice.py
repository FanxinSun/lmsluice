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

import http.server
import json
import os
import random
import struct
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    PACKAGE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lmsluice")

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

        saved_state, saved_probe = buffer._strategy, buffer._thp_mode
        try:
            buffer._strategy = None
            buffer._thp_mode = lambda: "never"
            how, why, floor = buffer.strategy()
            self.assertEqual(how, "bytes")
            self.assertIn("never", why)
            self.assertEqual(floor, 0)
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
            shutil.copy(os.path.join(os.path.dirname(__file__), "..",
                                     "lmsluice", "gate.py"),
                        os.path.join(d, "gate.py"))
            r = subprocess.run([sys.executable, "gate.py"], cwd=d,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertIn("decode faster than its disk", r.stdout)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
