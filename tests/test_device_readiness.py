"""Focused tests for the portable-device readiness contract and observer path."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.device_readiness.fixtures import generate_bundle
from lmsluice import ReadinessRecord
from lmsluice.model import open_model
from lmsluice.transport import transport


class TestDeviceReadiness(unittest.TestCase):
    def test_frozen_fixture_is_reproducible_and_role_complete(self):
        with tempfile.TemporaryDirectory(prefix="lmsluice-readiness-test-") as first:
            left = generate_bundle(first)
        with tempfile.TemporaryDirectory(prefix="lmsluice-readiness-test-") as second:
            right = generate_bundle(second)
        self.assertEqual(left["plain"]["bytes"], 145012)
        self.assertEqual(left["plain"]["sha256"],
                         "067ffed1b1199b4b01fb53c78ec66e10a2d70ed9471a280e84984d26e8aaa61c")
        self.assertEqual(left["plain"]["sha256"], right["plain"]["sha256"])
        self.assertEqual(left["plain"]["tensors"], right["plain"]["tensors"])
        self.assertEqual(
            {item["role"] for item in left["plain"]["tensors"]},
            {"language", "speech-in", "speech-out", "vision", "specialist", "memory-edge"},
        )

    def test_record_has_one_clock_and_explicit_missing_events(self):
        with tempfile.TemporaryDirectory(prefix="lmsluice-readiness-test-") as directory:
            record = ReadinessRecord(
                metadata={"case": "observer-contract"},
                sample_interval=0.001, max_samples=8,
            ).start()
            record.mark("first_payload", bytes=12)
            record.mark("first_payload", bytes=99)
            record.mark("consumer_first_useful", output="synthetic")
            record.set_route(source="https://example.invalid/model?token=secret")
            record.failure_event(ValueError(
                "https://example.invalid/model?token=secret"), phase="test")
            path = os.path.join(directory, "record.json")
            record.finish()
            record.write_json(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)

        self.assertEqual(data["schema"], 1)
        self.assertEqual(data["time_origin"]["clock"], "time.monotonic_ns")
        self.assertEqual(data["events"]["first_payload"]["details"]["bytes"], 12)
        self.assertIsNone(data["events"]["consumer_ready"])
        self.assertEqual(data["failure"]["message"],
                         "https://example.invalid/model?token=<redacted>")
        self.assertNotIn("?token=secret", json.dumps(data))
        self.assertGreaterEqual(data["resources"]["sampling"]["samples"], 1)

    def test_transport_observer_report_is_additive_and_workers_leave(self):
        record = ReadinessRecord(metadata={"case": "transport-observer"}).start()
        placed = bytearray()
        report = transport(
            [(0, 7), (7, 11), (18, 5)],
            lambda job: bytes([job[0] & 0xFF]) * job[1],
            lambda _job, payload: placed.extend(payload) or len(payload),
            fetch_threads=2, place_threads=1, inflight=2, observer=record,
        )
        data = record.finish(report)
        self.assertEqual(data["bytes"]["fetched"], 23)
        self.assertEqual(data["bytes"]["transferred"], 23)
        self.assertEqual(data["transport"]["fetched_bytes"], 23)
        self.assertEqual(data["transport"]["placed_bytes"], 23)
        self.assertIsNotNone(data["events"]["first_payload"])
        self.assertIsNotNone(data["events"]["transfer_complete"])
        self.assertEqual(data["failure"], None)
        self.assertEqual(
            [thread.name for thread in threading.enumerate()
             if thread.name.startswith("lmsluice-fetch-") or
             thread.name.startswith("lmsluice-place-")],
            [],
        )

    def test_observer_can_account_for_events_without_resource_sampling(self):
        record = ReadinessRecord(
            metadata={"case": "observer-no-resource-sampling"},
            sample_resources=False,
        ).start()
        placed = bytearray()
        report = transport(
            [(0, 4)], lambda job: b"data",
            lambda _job, payload: placed.extend(payload) or len(payload),
            fetch_threads=1, place_threads=1, observer=record,
        )
        data = record.finish(report)
        self.assertEqual(bytes(placed), b"data")
        self.assertIsNotNone(data["events"]["transfer_complete"])
        self.assertEqual(data["resources"]["sampling"]["status"], "DISABLED")
        self.assertEqual(data["resources"]["sampling"]["samples"], 0)

    def test_model_observer_preserves_plain_and_coded_consumer_bytes(self):
        with tempfile.TemporaryDirectory(prefix="lmsluice-readiness-test-") as directory:
            bundle = generate_bundle(directory)
            with open(os.path.join(directory, bundle["plain"]["path"]), "rb") as fh:
                expected = hashlib.sha256(fh.read()).hexdigest()
            for path in (os.path.join(directory, bundle["plain"]["path"]),
                         os.path.join(directory, bundle["coded"]["path"])):
                if not os.path.exists(path):
                    self.skipTest("stdlib coded fixture unavailable")
                record = ReadinessRecord(metadata={"case": "model-observer"}).start()
                with open_model(path, cache="off", observer=record,
                                fetch_threads=2, place_threads=1) as model:
                    output = bytes(model.load(observer=record))
                    record.mark("consumer_first_useful", output="sha256")
                    record.mark("consumer_ready", output="sha256")
                data = record.finish()
                self.assertEqual(hashlib.sha256(output).hexdigest(), expected)
                self.assertEqual(data["route"]["actual"], model.route)
                self.assertIsNotNone(data["events"]["first_tensor"])
                self.assertIsNotNone(data["events"]["consumer_ready"])

    def test_stream_marks_transfer_only_after_full_consumption(self):
        with tempfile.TemporaryDirectory(prefix="lmsluice-readiness-test-") as directory:
            bundle = generate_bundle(directory)
            path = os.path.join(directory, bundle["plain"]["path"])
            record = ReadinessRecord(metadata={"case": "stream-completion"}).start()
            with open_model(path, cache="off", observer=record) as model:
                stream = model.stream(budget=32 << 10, observer=record)
                next(stream)
                stream.close()
                self.assertIsNone(record.to_dict()["events"]["transfer_complete"])
                for _name, _view in model.stream(budget=32 << 10, observer=record):
                    pass
            data = record.finish()
        self.assertIsNotNone(data["events"]["transfer_complete"])


if __name__ == "__main__":
    unittest.main()
