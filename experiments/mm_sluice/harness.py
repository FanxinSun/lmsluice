"""Reproducible MM-SLUICE-01 bundle and consumer-boundary campaign.

The campaign is intentionally small and local.  It proves manifest identity,
safe plain materialization, lifecycle evidence, cleanup and optional-provider
boundaries.  It does not turn the synthetic ONNX Add graph into a speech,
vision, language or device-quality result.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from unittest import mock

from experiments.device_readiness.archive import create_archive
from experiments.mm_sluice.fixtures import audit_onnx_graph, generate_bundle, file_sha256
from lmsluice import ReadinessRecord
from lmsluice import bundle as B
from lmsluice import onnxruntime_adapter as ORT
from lmsluice.onnxruntime_adapter import OnnxCPUConsumer


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _error(exc):
    if exc is None:
        return None
    if hasattr(exc, "as_dict"):
        return exc.as_dict()
    return {"type": type(exc).__name__, "message": str(exc)[:400]}


def _write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _digest(path):
    return {"bytes": os.path.getsize(path), "sha256": file_sha256(path)}


def _memory_info():
    out = {}
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            for line in fh:
                key, sep, value = line.partition(":")
                if sep and key in ("MemTotal", "MemAvailable"):
                    out[key] = value.strip()
    except OSError:
        pass
    return out


def _module_capability(name):
    spec = importlib.util.find_spec(name)
    return {"available": spec is not None,
            "origin": getattr(spec, "origin", None) if spec else None}


def _active_staging(root):
    found = []
    for directory, dirnames, _filenames in os.walk(root):
        for name in dirnames:
            if name.startswith(".lmsluice-bundle-"):
                found.append(os.path.relpath(os.path.join(directory, name), root))
    return sorted(found)


@dataclass
class Campaign:
    out: str
    repo_root: str

    def __post_init__(self):
        self.out = os.path.abspath(self.out)
        self.repo_root = os.path.abspath(self.repo_root)
        self.records_dir = os.path.join(self.out, "records")
        self.errors_dir = os.path.join(self.out, "errors")
        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        self.rows = []
        self.record_data = {}
        self.probe_exceptions = []

    @property
    def required_ok(self):
        return all(row.get("status") == "PASS"
                   for row in self.rows if row.get("required"))

    def row(self, case, status, *, required=False, **facts):
        item = {"schema": 1, "case": case, "status": status,
                "required": bool(required), "recorded_at_utc": _utc_now()}
        item.update(_json_safe(facts))
        self.rows.append(item)
        with open(os.path.join(self.out, "results.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        return item

    def record(self, recorder: ReadinessRecord, *, report=None):
        data = recorder.finish(report)
        path = os.path.join(self.records_dir, f"{recorder.run_id}.json")
        recorder.write_json(path)
        self.record_data[recorder.run_id] = data
        return os.path.relpath(path, self.out), data

    def exception(self, case, exc):
        name = f"{case}.txt"
        path = os.path.join(self.errors_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_error(exc), fh, indent=2, sort_keys=True)
            fh.write("\n")
        self.probe_exceptions.append({"case": case, "error": _error(exc),
                                     "path": os.path.relpath(path, self.out)})
        return os.path.relpath(path, self.out)

    def write(self, *, fixture, capabilities, baseline):
        _write_json(os.path.join(self.out, "capabilities.json"), capabilities)
        _write_json(os.path.join(self.out, "fixture.json"), fixture)
        counts = {}
        for row in self.rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        required_failures = [row["case"] for row in self.rows
                             if row.get("required") and row.get("status") != "PASS"]
        summary = {
            "schema": 1,
            "campaign": "MM-SLUICE-01",
            "status": "PASS" if self.required_ok else "FAIL",
            "engineering_status": "PASS" if self.required_ok else "FAIL",
            "recorded_at_utc": _utc_now(),
            "fixture_version": fixture.get("fixture_version"),
            "manifest_sha256": fixture.get("manifest_sha256"),
            "result_counts": counts,
            "required_failures": required_failures,
            "expected_induced_failures": [row["case"] for row in self.rows
                                           if row.get("expected_failure")],
            "probe_exceptions": self.probe_exceptions,
            "capabilities": capabilities,
            "baseline": baseline,
            "resource_scope": {
                "max_busy_cpu_workers": 2,
                "fixture_limit_bytes": 128 << 20,
                "run_storage_limit_bytes": 2 << 30,
                "process_working_set_planning_limit_bytes": 1 << 30,
                "measurement": "ReadinessRecord current-process RSS/PSS/HWM when exposed; backend/device/energy remain UNMEASURED",
            },
            "claim_boundary": {
                "fixture": "synthetic structural and byte-preservation evidence",
                "real_speech_vision_language_quality": "UNAVAILABLE",
                "target_device_power_memory_thermal": "UNAVAILABLE",
                "customer_affordability_specialist_value": "INCONCLUSIVE/UNAVAILABLE",
            },
        }
        _write_json(os.path.join(self.out, "summary.json"), summary)
        return summary


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _capabilities(repo_root, fixture):
    optional = {name: _module_capability(name)
                for name in ("lmz", "onnxruntime", "onnx", "torch")}
    lmz_cap = B.LmzBundleProvider().capability()
    ort_cap = ORT.capability()
    try:
        from lmsluice.zstdcodec import _zstd
        _zstd()
    except Exception as exc:
        stdlib_codec = {"available": False,
                        "reason": f"{type(exc).__name__}: {exc}"}
    else:
        stdlib_codec = {"available": True, "reason": None}
    try:
        from lmsluice import cuda
        cuda_available, cuda_reason = cuda.available()
        cuda_cap = {"available": bool(cuda_available), "reason": cuda_reason}
    except Exception as exc:
        cuda_cap = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        disk = shutil.disk_usage(repo_root)
        free_disk = disk.free
    except OSError:
        free_disk = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        commit = f"unavailable: {type(exc).__name__}"
    return {
        "repository": repo_root,
        "commit": commit,
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "optional_modules": optional,
        "lmz_provider": lmz_cap,
        "onnxruntime_provider": ort_cap,
        "stdlib_codec": stdlib_codec,
        "cuda": cuda_cap,
        "resource_start": {"free_disk_bytes": free_disk,
                            "memory": _memory_info(),
                            "max_busy_workers": 2,
                            "fixture_limit_bytes": 128 << 20,
                            "run_storage_limit_bytes": 2 << 30,
                            "working_set_plan_bytes": 1 << 30},
        "fixture": {
            "version": fixture.get("fixture_version"),
            "manifest_sha256": fixture.get("manifest_sha256"),
            "total_bytes": fixture.get("total_bytes"),
            "graph": "model.onnx with external weights.bin range [3,7)",
        },
        "availability_policy": {
            "plain": "required and always exercised",
            "lmz": "optional local provider only; unavailable is NOT_RUN",
            "onnxruntime": "optional CPU provider only; unavailable is NOT_RUN",
            "onnx": "optional parser only; no installation",
        },
    }


def _git_value(repo_root, *args):
    try:
        return subprocess.run(["git", *args], cwd=repo_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def _clone_fixture(bundle, destination):
    os.makedirs(destination, exist_ok=True)
    for entry in bundle["entries"]:
        source = entry["actual_path"]
        target = os.path.join(destination, entry["path"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(bundle["manifest_path"], os.path.join(destination, "bundle.json"))
    return destination


def _negative_source(campaign, name):
    return os.path.join(campaign.out, "negative-sources", name)


def _new_record(case, *, sampled=False):
    return ReadinessRecord(
        metadata={"case": case, "availability": "synthetic_fixture",
                  "evidence_class": "synthetic_transport"},
        sample_interval=0.005 if sampled else 0.01,
        max_samples=64, sample_resources=sampled,
    ).start()


def run_plain_success(campaign, bundle):
    destination = os.path.join(campaign.out, "plain-materialized")
    record = _new_record("plain-materialization-success", sampled=True)
    error = None
    result = None
    try:
        result = B.materialize_bundle(bundle["source_root"], destination,
                                      observer=record, fetch_threads=1,
                                      place_threads=1, inflight=2)
        checks = []
        for entry in bundle["entries"]:
            output = os.path.join(destination, entry["path"])
            facts = _digest(output)
            checks.append({"path": entry["path"], "bytes": facts["bytes"],
                           "sha256": facts["sha256"],
                           "equal": facts["sha256"] == entry["sha256"]})
        if not all(item["equal"] for item in checks):
            raise AssertionError("materialized entry differs from manifest")
        graph_dep = next(entry for entry in bundle["entries"] if entry["path"] == "model.onnx")
        dep = next(item for item in graph_dep["dependencies"])
        with open(os.path.join(destination, dep["path"]), "rb") as fh:
            fh.seek(dep["offset"])
            dependency_bytes = fh.read(dep["length"])
        if dependency_bytes != b"\x00\x00\x80?":
            raise AssertionError("external dependency range differs")
        record.mark("release", owner="campaign", cleanup="complete")
        status = "PASS"
    except BaseException as exc:
        error = exc
        record.failure_event(exc, phase="plain_success")
        status = "FAIL"
        campaign.exception("plain-materialization-success", exc)
    path, data = campaign.record(record)
    campaign.row(
        "plain-materialization-success", status, required=True,
        evidence_class="synthetic_transport", record=path,
        result=result.to_dict() if result else None,
        entries=checks if result else None,
        events=data.get("events"), lifecycle=data.get("lifecycle"),
        bytes=data.get("bytes"), resources=data.get("resources"),
        error=_error(error), consumer_events={
            "consumer_initialized": data["events"].get("consumer_initialized"),
            "consumer_first_valid_output": data["events"].get("consumer_first_valid_output"),
            "consumer_ready": data["events"].get("consumer_ready"),
        },
    )
    return result


def run_contract(campaign, bundle):
    descriptor = None
    error = None
    try:
        descriptor = B.resolve_bundle(bundle["source_root"])
        report = B.validate_bundle(bundle["source_root"],
                                   expected_manifest_sha256=descriptor.identity)
        inventory = B.inventory_bundle(bundle["source_root"])
        ok = (descriptor.identity == bundle["manifest_sha256"] and
              report["valid"] and inventory["valid"] and
              len(descriptor.entries) == len(bundle["entries"]) and
              descriptor.graph.path == "model.onnx")
        if not ok:
            raise AssertionError("complete bundle identity/validation mismatch")
        campaign.row("bundle-contract", "PASS", required=True,
                     evidence_class="synthetic_transport",
                     manifest_sha256=descriptor.identity,
                     bundle_sha256=report["bundle_sha256"],
                     entry_count=len(descriptor.entries),
                     roles=sorted({entry.role for entry in descriptor.entries}),
                     dependency_ranges=[dep.to_dict()
                                        for entry in descriptor.entries
                                        for dep in entry.dependencies],
                     publisher_authenticated=report["publisher_authenticated"],
                     trusted_expected_digest=report["trusted_expected_digest"])
    except BaseException as exc:
        error = exc
        campaign.exception("bundle-contract", exc)
        campaign.row("bundle-contract", "FAIL", required=True,
                     evidence_class="synthetic_transport", error=_error(exc))


def run_fixture_audit(campaign, bundle):
    try:
        audit = audit_onnx_graph(bundle["graph_path"], bundle["weights_path"])
    except BaseException as exc:
        campaign.exception("onnx-graph-structure", exc)
        campaign.row("onnx-graph-structure", "FAIL", required=True,
                     evidence_class="synthetic_transport", error=_error(exc))
    else:
        campaign.row("onnx-graph-structure", "PASS", required=True,
                     evidence_class="synthetic_transport", **audit,
                     note="wire-format fixture audit; no ONNX parser or runtime dependency")


def run_negative(campaign, bundle, name, action, *, evidence_class="simulated_constraint"):
    record = _new_record(name)
    destination = os.path.join(campaign.out, "negative", name)
    error = None
    detected = False
    try:
        action(record, destination)
    except BaseException as exc:
        error = exc
        detected = True
        if os.path.lexists(destination):
            # A caller-owned destination is intentionally retained by the
            # destination-race case; the action can report that exception.
            if name not in ("destination-appearance-race", "pre-existing-destination"):
                raise AssertionError("failed operation left destination")
    finally:
        try:
            if error is not None and record.failure is None:
                record.failure_event(error, phase=name)
            path, data = campaign.record(record)
        except BaseException as exc:
            path, data = None, {}
            campaign.exception(f"{name}-record", exc)
    campaign.row(name, "FAIL" if detected else "PASS", required=False,
                 expected_failure=True, evidence_class=evidence_class,
                 record=path, error=_error(error),
                 cleanup={"destination_exists": os.path.lexists(destination),
                          "owned_staging": _active_staging(campaign.out)},
                 terminal_failure=data.get("events", {}).get("terminal_failure"))
    campaign.row(f"{name}-detection", "PASS" if detected else "FAIL",
                 required=True, evidence_class=evidence_class,
                 expected_failure_detection=True, source_case=name,
                 error_code=getattr(error, "code", type(error).__name__ if error else None),
                 cleanup_verified=(not os.path.exists(destination) or
                                   name in ("destination-appearance-race",
                                            "pre-existing-destination")),
                 release_event=data.get("events", {}).get("release"),
                 terminal_failure=data.get("events", {}).get("terminal_failure"))


def run_failures(campaign, bundle):
    def missing(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "missing-entry"))
        os.unlink(os.path.join(root, "weights.bin"))
        B.materialize_bundle(root, destination, observer=record)

    def truncated(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "truncated-entry"))
        with open(os.path.join(root, "weights.bin"), "r+b") as fh:
            fh.truncate(4)
        B.materialize_bundle(root, destination, observer=record)

    def corrupt(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "corrupt-entry"))
        path = os.path.join(root, "model.onnx")
        with open(path, "r+b") as fh:
            byte = fh.read(1)
            fh.seek(0)
            fh.write(bytes([byte[0] ^ 1]))
        B.materialize_bundle(root, destination, observer=record)

    def wrong_digest(record, destination):
        B.materialize_bundle(bundle["source_root"], destination, observer=record,
                             expected_manifest_sha256="0" * 64)

    def invalid_range(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "invalid-range"))
        with open(os.path.join(root, "bundle.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["bundle"]["entries"][0]["dependencies"][0]["length"] = 10000
        B.write_manifest(os.path.join(root, "bundle.json"), manifest)
        B.materialize_bundle(root, destination, observer=record)

    def unsafe_path(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "unsafe-path"))
        with open(os.path.join(root, "bundle.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["bundle"]["entries"][0]["path"] = "../escape"
        B.write_manifest(os.path.join(root, "bundle.json"), manifest)
        B.materialize_bundle(root, destination, observer=record)

    def symlink_source(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "symlink-source"))
        path = os.path.join(root, "weights.bin")
        saved = path + ".saved"
        os.rename(path, saved)
        os.symlink(saved, path)
        B.materialize_bundle(root, destination, observer=record)

    def special_source(record, destination):
        if os.name != "posix":
            raise B.BundleError("special-file fixture unavailable", code="not_run")
        root = _clone_fixture(bundle, _negative_source(campaign, "special-source"))
        fifo = os.path.join(root, "fifo")
        os.mkfifo(fifo)
        with open(os.path.join(root, "bundle.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["bundle"]["entries"].append(
            {"path": "fifo", "role": "opaque", "length": 0, "sha256": "0" * 64})
        B.write_manifest(os.path.join(root, "bundle.json"), manifest)
        B.materialize_bundle(root, destination, observer=record)

    def destination_existing(record, destination):
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, "caller-owned"), "wb") as fh:
            fh.write(b"keep")
        B.materialize_bundle(bundle["source_root"], destination, observer=record)

    def destination_race(record, destination):
        original = B._atomic_publish_no_replace

        def appears(stage, parent, name):
            os.mkdir(os.path.join(parent, name))
            return original(stage, parent, name)

        with mock.patch.object(B, "_atomic_publish_no_replace", appears):
            B.materialize_bundle(bundle["source_root"], destination, observer=record)

    def source_race(record, destination):
        root = _clone_fixture(bundle, _negative_source(campaign, "source-race"))
        path = os.path.join(root, "weights.bin")
        calls = [0]

        def cancel():
            calls[0] += 1
            if calls[0] == 1:
                with open(path, "r+b") as fh:
                    fh.seek(3)
                    fh.write(b"RACE")
            return False

        B.materialize_bundle(root, destination, observer=record,
                             cancellation=cancel)

    def interrupted(record, destination):
        with mock.patch.object(B._FdSource, "pread",
                               side_effect=ConnectionResetError("injected source reset")):
            B.materialize_bundle(bundle["source_root"], destination, observer=record)

    def ceiling(record, destination):
        B.materialize_bundle(bundle["source_root"], destination, observer=record,
                             max_bytes=bundle["total_bytes"] - 1)

    def cancelled(record, destination):
        B.materialize_bundle(bundle["source_root"], destination, observer=record,
                             cancellation=lambda: True)

    def incompatible_provider(record, destination):
        del record, destination
        B.provider("unknown-provider")

    for name, action in (
            ("missing-entry", missing), ("truncated-entry", truncated),
            ("corrupt-entry", corrupt), ("wrong-expected-digest", wrong_digest),
            ("invalid-dependency-range", invalid_range), ("unsafe-path", unsafe_path),
            ("symlink-source", symlink_source), ("special-source", special_source),
            ("pre-existing-destination", destination_existing),
            ("destination-appearance-race", destination_race),
            ("source-mutation-race", source_race), ("source-interruption", interrupted),
            ("allocation-ceiling", ceiling), ("cancellation-boundary", cancelled),
            ("incompatible-provider", incompatible_provider)):
        try:
            run_negative(campaign, bundle, name, action)
        except BaseException as exc:
            campaign.exception(f"{name}-harness", exc)
            campaign.row(f"{name}-detection", "FAIL", required=True,
                         evidence_class="simulated_constraint", error=_error(exc))


class _FakeSession:
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs
        self.error = error

    def run(self, _names, _inputs):
        if self.error:
            raise self.error
        return self.outputs


class _FakeOrt:
    __version__ = "1.17.0"

    def __init__(self, session):
        self.session = session

    def get_available_providers(self):
        return ["CPUExecutionProvider"]

    def InferenceSession(self, _graph, providers=None):
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("provider was not pinned to CPU")
        if isinstance(self.session, BaseException):
            raise self.session
        return self.session


def run_lifecycle_failures(campaign, bundle, materialized):
    # Engine initialization failure, with no initialization/ready event.
    record = _new_record("consumer-init-failure")
    error = None
    fake_module = _FakeOrt(RuntimeError("engine init injected"))
    try:
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_module}):
            OnnxCPUConsumer(materialized, observer=record).initialize()
    except BaseException as exc:
        error = exc
        record.mark("release", owner="consumer", cleanup="complete")
    path, data = campaign.record(record)
    campaign.row("consumer-init-failure", "FAIL" if error else "PASS", required=False,
                 expected_failure=True, evidence_class="simulated_constraint",
                 record=path, error=_error(error),
                 consumer_initialized=data["events"].get("consumer_initialized"),
                 consumer_ready=data["events"].get("consumer_ready"),
                 release=data["events"].get("release"))
    campaign.row("consumer-init-failure-detection", "PASS" if error else "FAIL",
                 required=True, evidence_class="simulated_constraint",
                 expected_failure_detection=True,
                 error_code=getattr(error, "code", None),
                 missing_ready=data["events"].get("consumer_ready") is None,
                 release_recorded=data["events"].get("release") is not None)

    # A caller rejection must not become first-valid or ready.
    record = _new_record("consumer-output-rejection")
    error = None
    fake_module = _FakeOrt(_FakeSession([[3.0]]))
    consumer = OnnxCPUConsumer(materialized, observer=record,
                               validity_hook=lambda _outputs: False)
    try:
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_module}):
            consumer.initialize()
            consumer.run({"input": [2.0]})
    except BaseException as exc:
        error = exc
    finally:
        consumer.release()
    path, data = campaign.record(record)
    campaign.row("consumer-output-rejection", "FAIL" if error else "PASS", required=False,
                 expected_failure=True, evidence_class="simulated_constraint",
                 record=path, error=_error(error),
                 consumer_initialized=data["events"].get("consumer_initialized"),
                 first_valid=data["events"].get("consumer_first_valid_output"),
                 ready=data["events"].get("consumer_ready"),
                 release=data["events"].get("release"))
    campaign.row("consumer-output-rejection-detection", "PASS" if error else "FAIL",
                 required=True, evidence_class="simulated_constraint",
                 expected_failure_detection=True,
                 error_code=getattr(error, "code", None),
                 no_false_ready=data["events"].get("consumer_ready") is None,
                 release_recorded=data["events"].get("release") is not None)

    # Cancellation after session initialization releases the consumer-owned
    # session and never emits a ready event.
    record = _new_record("consumer-cancellation")
    calls = [0]

    def cancellation():
        calls[0] += 1
        return calls[0] >= 2

    consumer = OnnxCPUConsumer(materialized, observer=record,
                               cancellation=cancellation)
    fake_module = _FakeOrt(_FakeSession([[3.0]]))
    error = None
    try:
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_module}):
            consumer.initialize()
    except BaseException as exc:
        error = exc
    finally:
        consumer.release()
    path, data = campaign.record(record)
    campaign.row("consumer-cancellation", "FAIL" if error else "PASS",
                 required=False, expected_failure=True,
                 evidence_class="simulated_constraint", record=path,
                 error=_error(error), events=data["events"],
                 lifecycle=data["lifecycle"], cleanup=data["execution"]["cleanup"])
    campaign.row("consumer-cancellation-detection",
                 "PASS" if isinstance(error, B.BundleCancelled) else "FAIL",
                 required=True, evidence_class="simulated_constraint",
                 expected_failure_detection=True,
                 error_code=getattr(error, "code", None),
                 boundary=(getattr(error, "details", {}) or {}).get("boundary"),
                 no_false_ready=data["events"].get("consumer_ready") is None,
                 release_recorded=data["events"].get("release") is not None)

    # Simulated successful consumer lifecycle proves that ready follows the
    # caller validity hook, while the real ORT capability remains independent.
    record = _new_record("consumer-valid-output")
    fake_module = _FakeOrt(_FakeSession([[3.0]]))
    consumer = OnnxCPUConsumer(materialized, observer=record,
                               validity_hook=lambda outputs: outputs == [[3.0]])
    error = None
    try:
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_module}):
            consumer.initialize()
            outputs = consumer.run({"input": [2.0]})
        if outputs != [[3.0]]:
            raise AssertionError("fake consumer output changed")
    except BaseException as exc:
        error = exc
    finally:
        consumer.release()
    path, data = campaign.record(record)
    ordered = [data["events"][name]["at_ns"] for name in (
        "consumer_initialized", "consumer_first_valid_output", "consumer_ready", "release")
               if data["events"].get(name) is not None]
    ok = (error is None and ordered == sorted(ordered) and
          not data["lifecycle"]["order_violations"])
    campaign.row("consumer-valid-output", "PASS" if ok else "FAIL", required=True,
                 evidence_class="simulated_constraint", record=path,
                 output=[[3.0]] if error is None else None,
                 events=data["events"], lifecycle=data["lifecycle"], error=_error(error),
                 note="simulated adapter lifecycle; no real ONNX Runtime quality claim")


def run_optional_ort(campaign, bundle, materialized):
    cap = ORT.capability()
    if not cap.get("available"):
        campaign.row("onnxruntime-cpu", "NOT_RUN", required=False,
                     evidence_class="optional_backend", availability="unavailable",
                     capability=cap,
                     reason=cap.get("reason"),
                     claim="no direct-vs-lmsluice graph equivalence claim")
        return
    try:
        import numpy as np
    except Exception as exc:
        campaign.row("onnxruntime-cpu", "NOT_RUN", required=False,
                     evidence_class="optional_backend", availability="incompatible",
                     capability=cap,
                     reason=f"numpy input dependency unavailable: {type(exc).__name__}: {exc}")
        return
    record = _new_record("onnxruntime-cpu-direct-and-plain", sampled=True)
    error = None
    direct = plain = None
    try:
        import onnxruntime as ort
        direct_session = ort.InferenceSession(
            bundle["graph_path"], providers=["CPUExecutionProvider"])
        ort_input = {"input": np.asarray([2.0], dtype=np.float32)}
        direct = direct_session.run(None, ort_input)
        consumer = OnnxCPUConsumer(materialized, observer=record,
                                   validity_hook=lambda outputs: len(outputs) == len(direct) and
                                   all(np.array_equal(left, right)
                                       for left, right in zip(outputs, direct)))
        consumer.initialize()
        plain = consumer.run(ort_input)
        consumer.release()
        if not (len(plain) == len(direct) and all(np.array_equal(left, right)
                                                  for left, right in zip(plain, direct))):
            raise AssertionError("direct and lmsluice plain outputs differ")
    except BaseException as exc:
        error = exc
        record.failure_event(exc, phase="onnxruntime-comparison")
    path, data = campaign.record(record)
    campaign.row("onnxruntime-cpu", "PASS" if error is None else "FAIL", required=True,
                 evidence_class="optional_backend", availability="configured",
                 capability=cap, record=path, direct_output=direct,
                 plain_output=plain, output_equivalent=error is None,
                 events=data["events"], resources=data["resources"], error=_error(error),
                 input_contract=bundle["input"], provider="CPUExecutionProvider")


def run_optional_lmz(campaign, bundle):
    cap = B.LmzBundleProvider().capability()
    if not cap.get("available"):
        campaign.row("lmz-complete-bundle", "NOT_RUN", required=False,
                     evidence_class="optional_backend", availability="unavailable",
                     capability=cap,
                     reason=cap.get("reason"),
                     note="accepted local lmz provider is optional and absent; plain route is independent")
        return
    provider = B.LmzBundleProvider()
    archive = os.path.join(campaign.out, "optional-lmz-bundle.lmz")
    destination = os.path.join(campaign.out, "optional-lmz-materialized")
    error = None
    inventory = None
    try:
        with open(bundle["manifest_path"], encoding="utf-8") as fh:
            manifest = json.load(fh)
        provider.create(bundle["source_root"], archive, manifest=manifest)
        inventory = provider.inventory(archive, strict=True)
        provider.materialize(archive, destination)
    except BaseException as exc:
        error = exc
    campaign.row("lmz-complete-bundle", "PASS" if error is None else "FAIL",
                 required=True, evidence_class="optional_backend", availability="configured",
                 capability=cap, archive=_digest(archive) if os.path.exists(archive) else None,
                 inventory=inventory, destination=destination if os.path.exists(destination) else None,
                 error=_error(error), sibling_mutation="none; run-directory output only")


def run_claim_boundaries(campaign):
    # These are deliberately independent product questions.  Recording them
    # keeps a successful structural fixture from being misread as evidence for
    # quality, affordability or a chosen specialist market.
    campaign.row(
        "real-multimodal-quality", "INCONCLUSIVE", required=False,
        evidence_class="claim_boundary", availability="unavailable",
        reason="no licensed ASR/TTS/vision/language model, audio, image, prompt or task set",
        result="no speech/vision/language quality claim",
    )
    campaign.row(
        "SLUICE-A1-affordability", "INCONCLUSIVE", required=False,
        evidence_class="customer_or_affordability", availability="unavailable",
        reason="no named host/device, BOM, power/thermal, memory/storage price, volume or customer evidence",
        result="complete-system affordability remains open",
    )
    campaign.row(
        "SLUICE-B1-specialized-deployment", "INCONCLUSIVE", required=False,
        evidence_class="customer_or_affordability", availability="unavailable",
        reason="technical fixture mechanics do not establish buyer, workflow, willingness-to-pay or support evidence",
        result="no specialist niche selected",
    )


def run_cache_preservation(campaign):
    # The corrected zero-open/zero-hash cache path is covered by the accepted
    # baseline tests. Keep an explicit additive row in this unit so the new
    # bundle work cannot be mistaken for a cache rewrite.
    source = os.path.join(campaign.repo_root, "lmsluice", "cache.py")
    try:
        with open(source, encoding="utf-8") as fh:
            body = fh.read()
        ok = ("def find(path: str)" in body and
              "_source_sha256(path)" not in body[body.index("def find(path: str)"):body.index("@dataclass")])
    except BaseException as exc:
        campaign.row("cache-fastpath-preserved", "FAIL", required=True,
                     evidence_class="synthetic_transport", error=_error(exc))
        return
    campaign.row("cache-fastpath-preserved", "PASS" if ok else "FAIL", required=True,
                 evidence_class="synthetic_transport", zero_open_zero_hash="retained",
                 note="prior accepted cache evidence and TestCache remain the regression gate")


def run_regression(campaign):
    log_path = os.path.join(campaign.out, "full-regression.log")
    command = [sys.executable, "-m", "unittest", "discover", "-v",
               "-s", "tests", "-p", "test*.py"]
    attempts = []
    for number in (1, 2):
        result = subprocess.run(command, cwd=campaign.repo_root,
                                capture_output=True, text=True, timeout=240,
                                env={**os.environ, "PYTHONPATH": campaign.repo_root})
        attempts.append(result)
        with open(os.path.join(campaign.out, f"full-regression.attempt{number}.log"),
                  "w", encoding="utf-8") as fh:
            fh.write(result.stdout)
            fh.write(result.stderr)
        if result.returncode == 0:
            break
    final = attempts[-1]
    with open(log_path, "w", encoding="utf-8") as fh:
        for number, result in enumerate(attempts, 1):
            fh.write(f"=== regression attempt {number} ===\n")
            fh.write(result.stdout)
            fh.write(result.stderr)
    output = final.stdout + final.stderr
    import re
    total = re.search(r"Ran (\d+) tests", output)
    skipped = re.search(r"skipped=(\d+)", output)
    skip_categories = {}
    for reason in re.findall(r"skipped ['\"]([^'\"]+)['\"]", output):
        skip_categories[reason] = skip_categories.get(reason, 0) + 1
    warnings = [line for line in output.splitlines()
                if "ResourceWarning" in line or "warning" in line.lower()]
    campaign.row("full-regression", "PASS" if final.returncode == 0 else "FAIL",
                 required=True, evidence_class="compatibility", returncode=final.returncode,
                 tests=int(total.group(1)) if total else None,
                 skipped=int(skipped.group(1)) if skipped else 0,
                 skip_categories=skip_categories, warnings=warnings,
                 attempts=len(attempts), log=os.path.basename(log_path),
                 initial_returncode=attempts[0].returncode,
                 core_dependency_policy="no mandatory lmz/onnxruntime/torch/site package")


def run_stdlib_core_check(campaign):
    """Import the core with ``-S`` so site-packages are not on sys.path."""
    code = (
        "import sys; "
        f"sys.path.insert(0, {campaign.repo_root!r}); "
        "import lmsluice; "
        "from lmsluice.bundle import BundleEntry, BundleRequest; "
        "assert not hasattr(lmsluice, 'onnxruntime'); "
        "print(lmsluice.__version__, BundleEntry.__name__, BundleRequest.__name__)"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code],
                            cwd=campaign.repo_root, capture_output=True,
                            text=True, timeout=30)
    log_name = "stdlib-core-check.log"
    with open(os.path.join(campaign.out, log_name), "w", encoding="utf-8") as fh:
        fh.write(result.stdout)
        fh.write(result.stderr)
    campaign.row("stdlib-only-core", "PASS" if result.returncode == 0 else "FAIL",
                 required=True, evidence_class="compatibility",
                 returncode=result.returncode, log=log_name,
                 output=result.stdout.strip(), error=result.stderr.strip(),
                 command="python -S -c import lmsluice and bundle models",
                 optional_imports="lmz, onnxruntime, onnx and torch absent from core path")


def run_campaign(out, *, repo_root=None, regression=True):
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    repo_root = os.path.abspath(repo_root or os.path.join(
        os.path.dirname(__file__), "..", ".."))
    fixture_root = os.path.join(out, "fixtures")
    bundle = generate_bundle(fixture_root)
    capabilities = _capabilities(repo_root, bundle)
    baseline = {
        "base_commit": "809a7172f03288e4fe9496545d3f35a66cb5ed65",
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "primary_checkout_policy": "preserved outside this worktree",
        "siblings": "read-only; no lmz files are edited",
    }
    campaign = Campaign(out, repo_root)
    # Set an immutable fixture pointer before cases begin so records can carry
    # the same manifest identity even when a negative source is mutated.
    _write_json(os.path.join(out, "fixture.json"), bundle)
    try:
        run_fixture_audit(campaign, bundle)
    except BaseException as exc:
        campaign.exception("fixture-audit-stage", exc)
        campaign.row("fixture-audit-stage", "FAIL", required=True, error=_error(exc))
    try:
        run_contract(campaign, bundle)
    except BaseException as exc:
        campaign.exception("bundle-contract-stage", exc)
        campaign.row("bundle-contract-stage", "FAIL", required=True, error=_error(exc))
    materialized = None
    try:
        materialized = run_plain_success(campaign, bundle)
    except BaseException as exc:
        campaign.exception("plain-stage", exc)
        campaign.row("plain-stage", "FAIL", required=True, error=_error(exc))
    try:
        run_failures(campaign, bundle)
    except BaseException as exc:
        campaign.exception("failure-matrix-stage", exc)
        campaign.row("failure-matrix-stage", "FAIL", required=True, error=_error(exc))
    if materialized is not None:
        try:
            run_lifecycle_failures(campaign, bundle, materialized)
        except BaseException as exc:
            campaign.exception("consumer-lifecycle-stage", exc)
            campaign.row("consumer-lifecycle-stage", "FAIL", required=True, error=_error(exc))
        try:
            run_optional_ort(campaign, bundle, materialized)
        except BaseException as exc:
            campaign.exception("onnxruntime-stage", exc)
            campaign.row("onnxruntime-stage", "FAIL", required=True, error=_error(exc))
    else:
        campaign.row("consumer-lifecycle-stage", "NOT_RUN", required=False,
                     reason="plain materialization prerequisite failed")
        campaign.row("onnxruntime-cpu", "NOT_RUN", required=False,
                     reason="plain materialization prerequisite failed")
    try:
        run_optional_lmz(campaign, bundle)
    except BaseException as exc:
        campaign.exception("lmz-stage", exc)
        campaign.row("lmz-complete-bundle", "FAIL", required=True,
                     evidence_class="optional_backend", error=_error(exc))
    run_claim_boundaries(campaign)
    run_cache_preservation(campaign)
    try:
        run_stdlib_core_check(campaign)
    except BaseException as exc:
        campaign.exception("stdlib-core-stage", exc)
        campaign.row("stdlib-only-core", "FAIL", required=True, error=_error(exc))
    if regression:
        try:
            run_regression(campaign)
        except BaseException as exc:
            campaign.exception("full-regression-stage", exc)
            campaign.row("full-regression", "FAIL", required=True, error=_error(exc))
    else:
        campaign.row("full-regression", "NOT_RUN", required=False,
                     reason="--no-regression requested")
    summary = campaign.write(fixture=bundle, capabilities=capabilities,
                             baseline=baseline)
    print(json.dumps({
        "status": summary["status"],
        "engineering_status": summary["engineering_status"],
        "run": out,
        "summary": os.path.join(out, "summary.json"),
        "results": os.path.join(out, "results.jsonl"),
        "records": os.path.join(out, "records"),
        "errors": os.path.join(out, "errors"),
    }, sort_keys=True))
    return 0 if campaign.required_ok else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--no-regression", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="lmsluice-mm-sluice-01-")
    return run_campaign(out, repo_root=args.repo_root,
                        regression=not args.no_regression)


if __name__ == "__main__":
    raise SystemExit(main())
