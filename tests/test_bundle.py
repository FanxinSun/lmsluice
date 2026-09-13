"""Focused MM-SLUICE-01 bundle, lifecycle and optional-boundary tests."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from experiments.mm_sluice.fixtures import audit_onnx_graph, generate_bundle
from lmsluice import ReadinessRecord
from lmsluice import bundle as B
from lmsluice.onnxruntime_adapter import OnnxCPUConsumer, capability


class TestBundle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="lmsluice-mm-bundle-test-")
        self.fixture = generate_bundle(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _destination(self, name="materialized"):
        return os.path.join(self.directory, name)

    def test_complete_identity_validation_and_plain_roundtrip(self):
        descriptor = B.resolve_bundle(self.fixture["source_root"])
        self.assertEqual(descriptor.schema, "1.0")
        self.assertEqual(descriptor.identity, self.fixture["manifest_sha256"])
        self.assertEqual(descriptor.graph.path, "model.onnx")
        self.assertEqual(descriptor.total_bytes, self.fixture["total_bytes"])
        self.assertEqual(
            audit_onnx_graph(self.fixture["graph_path"], self.fixture["weights_path"]),
            {
                "operator": "Add",
                "external_location": "weights.bin",
                "external_offset": 3,
                "external_length": 4,
                "external_float": 1.0,
            },
        )
        graph = next(entry for entry in descriptor.entries if entry.path == "model.onnx")
        self.assertEqual(graph.dependencies[0].path, "weights.bin")
        self.assertEqual(graph.dependencies[0].offset, 3)
        self.assertEqual(graph.dependencies[0].length, 4)
        report = B.validate_bundle(
            self.fixture["source_root"],
            expected_manifest_sha256=descriptor.identity,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["entries"]), 7)
        inventory = B.inventory_bundle(self.fixture["source_root"])
        self.assertEqual(inventory["declared_materialized_bytes"],
                         self.fixture["total_bytes"])
        result = B.materialize_bundle(self.fixture["source_root"], self._destination())
        self.assertEqual(result.manifest_sha256, descriptor.identity)
        self.assertEqual(result.materialized_bytes, self.fixture["total_bytes"])
        self.assertEqual(result.entry_point, "model.onnx")
        self.assertEqual(result.consumer["backend"], "CPUExecutionProvider")
        for entry in descriptor.entries:
            left = os.path.join(self.fixture["source_root"], entry.path)
            right = os.path.join(result.destination, entry.path)
            with open(left, "rb") as left_fh, open(right, "rb") as right_fh:
                self.assertEqual(left_fh.read(), right_fh.read())

    def test_identity_changes_with_every_entry_identity(self):
        with open(self.fixture["manifest_path"], encoding="utf-8") as fh:
            manifest = json.load(fh)
        before = B.canonical_manifest_sha256(manifest)
        manifest["bundle"]["entries"][1]["sha256"] = "0" * 64
        self.assertNotEqual(before, B.canonical_manifest_sha256(manifest))

    def test_invalid_manifest_forms_have_structured_codes(self):
        with open(self.fixture["manifest_path"], encoding="utf-8") as fh:
            original = json.load(fh)
        cases = []
        duplicate = json.loads(json.dumps(original))
        duplicate["bundle"]["entries"].append(
            dict(duplicate["bundle"]["entries"][0]))
        cases.append((duplicate, "duplicate_path"))
        duplicate_id = json.loads(json.dumps(original))
        duplicate_id["bundle"]["entries"][0]["id"] = "same"
        duplicate_id["bundle"]["entries"][1]["id"] = "same"
        cases.append((duplicate_id, "duplicate_identity"))
        missing = json.loads(json.dumps(original))
        missing["bundle"]["entries"][0]["dependencies"] = [{"path": "gone.bin"}]
        cases.append((missing, "missing_dependency"))
        cyclic = json.loads(json.dumps(original))
        cyclic["bundle"]["entries"][0]["dependencies"] = [{"path": "weights.bin"}]
        cyclic["bundle"]["entries"][1]["dependencies"] = [{"path": "model.onnx"}]
        cases.append((cyclic, "dependency_cycle"))
        bad_range = json.loads(json.dumps(original))
        bad_range["bundle"]["entries"][0]["dependencies"][0]["length"] = 10000
        cases.append((bad_range, "invalid_range"))
        unsafe = json.loads(json.dumps(original))
        unsafe["bundle"]["entries"][0]["path"] = "../escape"
        cases.append((unsafe, "unsafe_path"))
        unsupported = json.loads(json.dumps(original))
        unsupported["bundle"]["schema"] = "2.0"
        cases.append((unsupported, "unsupported_major"))
        for manifest, code in cases:
            with tempfile.TemporaryDirectory(dir=self.directory) as root:
                for entry in self.fixture["entries"]:
                    source = entry["actual_path"]
                    target = os.path.join(root, entry["path"])
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copyfile(source, target)
                B.write_manifest(os.path.join(root, "bundle.json"), manifest)
                with self.assertRaises(B.BundleError) as context:
                    B.resolve_bundle(root)
                self.assertEqual(context.exception.code, code)

    def test_wrong_digest_missing_truncated_and_corrupt_source_fail(self):
        with self.assertRaises(B.BundleError) as context:
            B.validate_bundle(self.fixture["source_root"],
                              expected_manifest_sha256="0" * 64)
        self.assertEqual(context.exception.code, "expected_digest")
        target = os.path.join(self.fixture["source_root"], "weights.bin")
        with open(target, "rb") as fh:
            original = fh.read()
        for mutated in (original[:3], original[:3] + b"X" + original[4:]):
            with open(target, "wb") as fh:
                fh.write(mutated)
            with self.assertRaises(B.BundleError) as context:
                B.materialize_bundle(self.fixture["source_root"], self._destination())
            self.assertIn(context.exception.code, {"length_mismatch", "digest_mismatch"})
            self.assertFalse(os.path.exists(self._destination()))
            if os.path.exists(target):
                os.unlink(target)
            with open(target, "wb") as fh:
                fh.write(original)

    def test_existing_destination_and_atomic_appearance_are_no_replace(self):
        destination = self._destination()
        os.mkdir(destination)
        sentinel = os.path.join(destination, "sentinel")
        with open(sentinel, "wb") as fh:
            fh.write(b"caller-owned")
        with self.assertRaises(B.BundleError) as context:
            B.materialize_bundle(self.fixture["source_root"], destination)
        self.assertEqual(context.exception.code, "destination_exists")
        with open(sentinel, "rb") as fh:
            self.assertEqual(fh.read(), b"caller-owned")
        shutil.rmtree(destination)

        original_publish = B._atomic_publish_no_replace

        def appears(stage, parent, name):
            os.mkdir(os.path.join(parent, name))
            return original_publish(stage, parent, name)

        with mock.patch.object(B, "_atomic_publish_no_replace", appears):
            with self.assertRaises(B.BundleError) as context:
                B.materialize_bundle(self.fixture["source_root"], destination)
        self.assertEqual(context.exception.code, "destination_exists")
        self.assertTrue(os.path.isdir(destination))
        self.assertEqual([name for name in os.listdir(destination)], [])
        self.assertEqual([name for name in os.listdir(self.directory)
                          if name.startswith(".lmsluice-bundle-")], [])

    @unittest.skipUnless(os.name == "posix", "POSIX special-file safety fixture")
    def test_symlink_and_special_source_are_rejected(self):
        target = os.path.join(self.fixture["source_root"], "weights.bin")
        saved = target + ".saved"
        os.rename(target, saved)
        os.symlink(saved, target)
        with self.assertRaises(B.BundleError) as context:
            B.resolve_bundle(self.fixture["source_root"])
        self.assertEqual(context.exception.code, "unsafe_source")
        os.unlink(target)
        os.rename(saved, target)
        fifo = os.path.join(self.fixture["source_root"], "fifo")
        os.mkfifo(fifo)
        with open(self.fixture["manifest_path"], encoding="utf-8") as fh:
            manifest = json.load(fh)
        entry = {"path": "fifo", "role": "opaque", "length": 0,
                 "sha256": "0" * 64}
        manifest["bundle"]["entries"].append(entry)
        B.write_manifest(self.fixture["manifest_path"], manifest)
        with self.assertRaises(B.BundleError) as context:
            B.resolve_bundle(self.fixture["source_root"])
        self.assertEqual(context.exception.code, "unsafe_source")

    def test_stale_generation_and_ceiling_fail_before_publish(self):
        target = os.path.join(self.fixture["source_root"], "config.json")
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("changed\n")
        with self.assertRaises(B.BundleError) as context:
            B.resolve_bundle(self.fixture["source_root"],
                             source_generation={"stale": True})
        self.assertEqual(context.exception.code, "source_changed")
        # Restore the manifest-declared file by regenerating the fixture in a
        # fresh source, then exercise the ceiling independently.
        shutil.rmtree(self.directory)
        self.directory = tempfile.mkdtemp(prefix="lmsluice-mm-bundle-test-")
        self.fixture = generate_bundle(self.directory)
        with self.assertRaises(B.BundleError) as context:
            B.materialize_bundle(self.fixture["source_root"], self._destination(),
                                 max_bytes=self.fixture["total_bytes"] - 1)
        self.assertEqual(context.exception.code, "resource_limit")
        self.assertFalse(os.path.exists(self._destination()))

    def test_cancellation_cleans_owned_staging_and_records_boundary(self):
        calls = [0]

        def cancellation():
            calls[0] += 1
            return calls[0] > 3

        record = ReadinessRecord(sample_resources=False).start()
        with self.assertRaises(B.BundleCancelled):
            B.materialize_bundle(self.fixture["source_root"], self._destination(),
                                 observer=record, cancellation=cancellation)
        data = record.finish()
        self.assertIsNotNone(data["events"]["cancelled"])
        self.assertIsNotNone(data["events"]["release"])
        self.assertIsNone(data["events"]["materialization_complete"])
        self.assertEqual(data["execution"]["cleanup"]["owned_staging"], "removed")
        self.assertEqual(data["execution"]["cancellation_boundary"],
                         data["events"]["cancelled"]["details"]["boundary"])
        self.assertFalse(os.path.exists(self._destination()))
        self.assertEqual([name for name in os.listdir(self.directory)
                          if name.startswith(".lmsluice-bundle-")], [])

    def test_observer_lifecycle_is_additive_and_consumer_events_stay_missing(self):
        record = ReadinessRecord(sample_resources=False).start()
        result = B.materialize_bundle(self.fixture["source_root"],
                                      self._destination(), observer=record)
        record.mark("release", owner="caller", cleanup="complete")
        data = record.finish()
        self.assertEqual(data["route"]["provider"], "plain")
        self.assertEqual(data["route"]["bundle_identity"], result.manifest_sha256)
        self.assertEqual(data["bytes"]["materialized"], result.materialized_bytes)
        self.assertEqual(data["execution"]["cleanup"]["owned_staging"], "published")
        self.assertEqual(data["execution"]["resource_bytes"]["staging_reserved"],
                         result.materialized_bytes)
        self.assertIsNotNone(data["events"]["bundle_requested"])
        self.assertIsNotNone(data["events"]["bundle_resolved"])
        self.assertIsNotNone(data["events"]["bundle_verified"])
        self.assertIsNotNone(data["events"]["fetch_started"])
        self.assertIsNotNone(data["events"]["reconstruction_complete"])
        self.assertIsNotNone(data["events"]["materialization_complete"])
        self.assertIsNone(data["events"]["consumer_initialized"])
        self.assertIsNone(data["events"]["consumer_first_valid_output"])
        self.assertIsNone(data["events"]["consumer_ready"])
        self.assertEqual(data["lifecycle"]["order_violations"], [])
        self.assertEqual(data["lifecycle"]["events"][-1], "release")

    def test_failure_record_is_terminal_and_release_is_retained(self):
        record = ReadinessRecord(sample_resources=False).start()
        with self.assertRaises(B.BundleError):
            B.materialize_bundle(self.fixture["source_root"], self._destination(),
                                 expected_manifest_sha256="f" * 64,
                                 observer=record)
        data = record.finish()
        self.assertIsNotNone(data["failure"])
        self.assertIsNotNone(data["events"]["terminal_failure"])
        self.assertIsNotNone(data["events"]["release"])
        self.assertIsNone(data["events"]["consumer_ready"])
        self.assertEqual(data["execution"]["cleanup"]["owned_staging"], "removed")


class TestOptionalConsumer(unittest.TestCase):
    def test_capability_is_explicit_and_does_not_require_ort(self):
        got = capability()
        self.assertEqual(got["provider"], "onnxruntime")
        if not got["available"]:
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(B.BundleError) as context:
                    OnnxCPUConsumer(root).initialize()
                self.assertIn(context.exception.code,
                              {"provider_unavailable", "consumer_init_failed"})

    def test_consumer_cancellation_releases_after_initialization(self):
        with tempfile.TemporaryDirectory(prefix="lmsluice-mm-consumer-") as root:
            with open(os.path.join(root, "model.onnx"), "wb") as fh:
                fh.write(b"fixture")
            fake = mock.Mock()
            fake.__version__ = "1.17.0"
            fake.get_available_providers.return_value = ["CPUExecutionProvider"]
            fake.InferenceSession.return_value = mock.Mock()
            calls = [0]

            def cancellation():
                calls[0] += 1
                return calls[0] >= 2

            record = ReadinessRecord(sample_resources=False).start()
            consumer = OnnxCPUConsumer(root, observer=record,
                                       cancellation=cancellation)
            with mock.patch.dict("sys.modules", {"onnxruntime": fake}):
                with self.assertRaises(B.BundleCancelled) as context:
                    consumer.initialize()
            data = record.finish()
            self.assertEqual(context.exception.details["boundary"],
                             "after_consumer_init")
            self.assertIsNotNone(data["events"]["consumer_initialized"])
            self.assertIsNotNone(data["events"]["cancelled"])
            self.assertIsNotNone(data["events"]["terminal_failure"])
            self.assertIsNotNone(data["events"]["release"])
            self.assertEqual(data["execution"]["cleanup"]["consumer_session"],
                             "released")


if __name__ == "__main__":
    unittest.main()
