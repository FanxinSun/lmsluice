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
        self.assertAlmostEqual(P.gate(2e9, 1e9), 2.0)
        self.assertIsNone(P.gate(None, 1e9))
        self.assertIsNone(P.gate(1e9, 0))

    def test_agrees_with_the_standalone_gate(self):
        """`plan.py` and `gate.py` state one identity and must not drift.

        They are separate files on purpose -- `gate.py` predicts a decode rate
        for a machine that is not here and has no imports so it can be run on
        one, while this predicts nothing and prices what was measured. Since
        neither can import the other without losing what makes it useful, the
        agreement is pinned here instead.
        """
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "gate", os.path.join(root, "gate.py"))
        gate = importlib.util.module_from_spec(spec)
        # Registered before executing: gate.py declares a dataclass, and
        # dataclasses resolves annotations through sys.modules[__module__],
        # which is absent for a module loaded straight off a path.
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)

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
        cls.ref = open(cls.plain, "rb").read()
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
        self.assertEqual(open(out, "rb").read(), self.ref)

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
            back = Profile.from_json(json.load(open(path)))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
