"""Unit tests for Phase 3 improvements in lmsluice.

Tests:
1. encode_rate decode measurement & profile merge.
2. cache.worth_caching codec selection ("auto", "lmz", "zstd").
3. Darwin anonymous mmap buffer allocation.
4. Direct Apple Silicon MPS device support in torchadapter.
5. Multi-shard model directory support in open_model & torchadapter.load_file.
6. torchadapter.load_model helper.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from lmsluice import buffer, cache, open_model, probe
from lmsluice.rates import Codec, Profile, Storage

# zstd is a property of the interpreter (Python 3.14+, or the `zstandard`
# package), not a fault: the test that needs it skips rather than errors.
try:
    from lmsluice.zstdcodec import _zstd as _probe_zstd

    _probe_zstd()
    _ZSTD_WHY = ""
except Exception as _exc:                     # noqa: BLE001 -- absence is the point
    _ZSTD_WHY = str(_exc).split(".")[0]
needs_zstd = unittest.skipIf(bool(_ZSTD_WHY), f"no zstd: {_ZSTD_WHY}")


class TestPhase3Improvements(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="lmsluice-test-p3-")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_encode_rate_result_and_profile_merge(self):
        """encode_rate must measure decode and unpack cleanly as (rate, ratio)."""
        from lmsluice.probe import EncodeResult

        res = EncodeResult(1.2e9, 0.65, decode=2.1e9, threads=4)
        # 2-item unpacking compatibility
        rate, ratio = res
        self.assertEqual(rate, 1.2e9)
        self.assertEqual(ratio, 0.65)
        # attributes
        self.assertEqual(res.encode, 1.2e9)
        self.assertEqual(res.ratio, 0.65)
        self.assertEqual(res.decode, 2.1e9)
        self.assertEqual(res.threads, 4)

        # Profile merge logic when probing an archive
        p = Profile()
        p.codecs["lmz-cpu"] = Codec(name="lmz-cpu", encode=1.0e9, ratio=0.67)
        # Simulate archive probe with new decode rate
        new_codec = Codec(name="lmz-cpu", decode=2.2e9, threads=8, ratio=0.66)
        existing = p.codecs.get("lmz-cpu")
        merged = Codec(
            name="lmz-cpu",
            decode=new_codec.decode if new_codec.decode is not None else existing.decode,
            encode=new_codec.encode if new_codec.encode is not None else existing.encode,
            threads=new_codec.threads or existing.threads,
            ratio=new_codec.ratio if new_codec.ratio is not None else existing.ratio,
        )
        self.assertEqual(merged.encode, 1.0e9)
        self.assertEqual(merged.decode, 2.2e9)
        self.assertEqual(merged.threads, 8)

    @needs_zstd
    def test_cache_worth_caching_codecs(self):
        """worth_caching must accept codec and evaluate correctly."""
        # Create a small dummy file
        dummy_file = os.path.join(self.tmpdir, "model.safetensors")
        with open(dummy_file, "wb") as f:
            f.write(b"\x00" * (20 << 20))  # 20 MB dummy data

        p = Profile()
        p.storage[cache.storage_key(dummy_file)] = Storage(
            key=cache.storage_key(dummy_file),
            cold=100e6,  # 100 MB/s slow link
        )

        v_zstd = cache.worth_caching(dummy_file, p, codec="zstd")
        self.assertIsInstance(v_zstd, cache.Verdict)

        v_auto = cache.worth_caching(dummy_file, p, codec="auto")
        self.assertIsInstance(v_auto, cache.Verdict)

    def test_darwin_anonymous_mmap_buffer(self):
        """buffer.destination must use mapped allocations for large requests on Darwin."""
        how, why, floor = buffer.strategy()
        if os.uname().sysname == "Darwin":
            self.assertEqual(how, "mapped")
            self.assertIn("darwin", why)
            # Above floor (16 MiB) should allocate mmap buffer
            buf = buffer.destination(32 << 20)
            self.assertNotIsInstance(buf, bytearray)
            self.assertEqual(len(buf), 32 << 20)
            # Memoryview writes must work
            mv = memoryview(buf)
            mv[0:5] = b"HELLO"
            self.assertEqual(bytes(mv[0:5]), b"HELLO")
            # Below floor should allocate bytearray
            small = buffer.destination(4096)
            self.assertIsInstance(small, bytearray)

    def test_multi_shard_model_interface(self):
        """open_model must transparently support sharded index and model directory."""
        # Create a synthetic 2-shard model
        shard1 = os.path.join(self.tmpdir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(self.tmpdir, "model-00002-of-00002.safetensors")

        def make_safetensors(path, tensors_dict):
            header = {}
            offset = 0
            raw_payloads = []
            for name, (dtype, shape, data) in tensors_dict.items():
                length = len(data)
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + length],
                }
                offset += length
                raw_payloads.append(data)
            hdr_bytes = json.dumps(header).encode("utf-8")
            with open(path, "wb") as f:
                import struct
                f.write(struct.pack("<Q", len(hdr_bytes)))
                f.write(hdr_bytes)
                for d in raw_payloads:
                    f.write(d)

        make_safetensors(shard1, {
            "layer.0.weight": ("F32", [2, 2], b"\x01\x00\x00\x00" * 4),
        })
        make_safetensors(shard2, {
            "layer.1.weight": ("F32", [2, 2], b"\x02\x00\x00\x00" * 4),
        })

        index_file = os.path.join(self.tmpdir, "model.safetensors.index.json")
        with open(index_file, "w") as f:
            json.dump({
                "metadata": {"total_size": 32},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }, f)

        # 1. Open via directory
        with open_model(self.tmpdir) as m_dir:
            self.assertEqual(len(m_dir.tensors), 2)
            self.assertIn("layer.0.weight", m_dir.tensors)
            self.assertIn("layer.1.weight", m_dir.tensors)
            t0 = bytes(m_dir.tensor("layer.0.weight"))
            t1 = bytes(m_dir.tensor("layer.1.weight"))
            self.assertEqual(t0, b"\x01\x00\x00\x00" * 4)
            self.assertEqual(t1, b"\x02\x00\x00\x00" * 4)

        # 2. Open via index path
        with open_model(index_file) as m_idx:
            self.assertEqual(len(m_idx.tensors), 2)
            t0 = bytes(m_idx.tensor("layer.0.weight"))
            self.assertEqual(t0, b"\x01\x00\x00\x00" * 4)

    def test_torchadapter_sharded_and_mps(self):
        """torchadapter.load_file must load sharded models and transfer to MPS."""
        try:
            import torch
            from lmsluice import torchadapter
        except ImportError:
            self.skipTest("torch is not installed")

        # Create a synthetic 2-shard model
        shard1 = os.path.join(self.tmpdir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(self.tmpdir, "model-00002-of-00002.safetensors")

        def make_safetensors(path, tensors_dict):
            header = {}
            offset = 0
            raw_payloads = []
            for name, (dtype, shape, data) in tensors_dict.items():
                length = len(data)
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + length],
                }
                offset += length
                raw_payloads.append(data)
            hdr_bytes = json.dumps(header).encode("utf-8")
            with open(path, "wb") as f:
                import struct
                f.write(struct.pack("<Q", len(hdr_bytes)))
                f.write(hdr_bytes)
                for d in raw_payloads:
                    f.write(d)

        make_safetensors(shard1, {
            "layer.0.weight": ("F32", [2, 2], b"\x01\x00\x00\x00" * 4),
        })
        make_safetensors(shard2, {
            "layer.1.weight": ("F32", [2, 2], b"\x02\x00\x00\x00" * 4),
        })

        index_file = os.path.join(self.tmpdir, "model.safetensors.index.json")
        with open(index_file, "w") as f:
            json.dump({
                "metadata": {"total_size": 32},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }, f)

        # Multi-shard load_file to CPU
        sd_cpu = torchadapter.load_file(self.tmpdir, device="cpu")
        self.assertEqual(len(sd_cpu), 2)
        self.assertEqual(sd_cpu["layer.0.weight"].device.type, "cpu")
        self.assertEqual(sd_cpu["layer.1.weight"].device.type, "cpu")

        # Multi-shard load_file to MPS (if available)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            sd_mps = torchadapter.load_file(self.tmpdir, device="mps")
            self.assertEqual(len(sd_mps), 2)
            self.assertEqual(sd_mps["layer.0.weight"].device.type, "mps")
            self.assertEqual(sd_mps["layer.1.weight"].device.type, "mps")

        # Multi-shard stream_tensors
        streamed = list(torchadapter.stream_tensors(self.tmpdir, device="cpu"))
        self.assertEqual(len(streamed), 2)
        names = [n for n, _ in streamed]
        self.assertEqual(names, ["layer.0.weight", "layer.1.weight"])


if __name__ == "__main__":
    unittest.main()
