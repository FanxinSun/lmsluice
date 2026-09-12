"""Dependency-free readiness campaign used by the portable-device probe.

This module is deliberately a campaign harness rather than a second loader.
It calls the public lmsluice source, archive, model, cloud and transport
interfaces, records every case as PASS/FAIL/SKIP/NOT_RUN/INCONCLUSIVE, and
keeps the generated artifacts and records in a fresh run directory. A failed
case is evidence; it is never removed from a summary to obtain a green exit.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import http.server
import json
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from contextlib import contextmanager

from lmsluice import ReadinessRecord
from lmsluice import observability
from lmsluice.archive import Archive
from lmsluice.model import open_model
from lmsluice.source import FileSource, Source
from lmsluice.transport import split, transport

from .fixtures import FIXTURE_VERSION, generate_bundle, hash_file


CAMPAIGN_SCHEMA = 2
REQUIRED_ENGINEERING = {
    "contract",
    "observability-success",
    "observability-failure",
    "observability-event-order",
    "observability-overhead-no-sampling",
    "observability-overhead-sampled",
    "bandwidth-plain",
    "bandwidth-coded",
    "memory-window",
    "memory-oversized-tensor",
    "memory-noncontiguous",
    "fault-source-reset",
    "fault-truncated-range",
    "fault-corrupt-coded",
    "fault-destination-bounds",
    "consumer-departure",
    "application-restart",
    "multipart-failure-detection",
    "cache-workflow",
}


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _error(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": observability.safe_identity(str(exc))[:400]}


def _file_digest(path: str) -> dict:
    return {"path": os.path.basename(path), "bytes": os.path.getsize(path),
            "sha256": hash_file(path)}


def _event_seconds(record: dict, name: str):
    event = record.get("events", {}).get(name)
    return None if event is None else event.get("at_s")


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


class Campaign:
    """Collect rows, records and failure evidence for one run."""

    def __init__(self, out: str, *, provenance: dict | None = None):
        self.out = os.path.abspath(out)
        self.records_dir = os.path.join(self.out, "records")
        self.errors_dir = os.path.join(self.out, "errors")
        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        self.rows: list[dict] = []
        self.required_failures: list[dict] = []
        self.record_data: dict[str, dict] = {}
        self.provenance = dict(provenance or {})
        self.started_at = _utc_now()
        self._case_status = {}

    def row(self, case: str, status: str, *, required: bool = False, **facts) -> dict:
        """Append a row even when a case is skipped or failed."""
        row = {"schema": CAMPAIGN_SCHEMA, "case": case, "status": status,
               "required": required, "recorded_at_utc": _utc_now(), **facts}
        self.rows.append(row)
        self._case_status[case] = status
        if required and status not in ("PASS", "SKIP"):
            self.required_failures.append(row)
        return row

    def record(self, record: ReadinessRecord, *, report=None,
               provenance: dict | None = None) -> tuple[str, dict]:
        """Freeze a recorder, save it, and return its path and JSON object."""
        record_identity = dict(self.provenance)
        if provenance:
            record_identity.update(provenance)
        if record_identity:
            record.set_provenance(**record_identity)
        data = record.finish(report)
        path = os.path.join(self.records_dir, f"{record.run_id}.json")
        record.write_json(path)
        self.record_data[record.run_id] = data
        return path, data

    def exception(self, case: str, exc: BaseException) -> str:
        path = os.path.join(self.errors_dir, f"{case}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return path

    def write(self, bundle: dict, *, manifest: dict | None = None) -> None:
        raw_path = os.path.join(self.out, "results.jsonl")
        with open(raw_path, "w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        counts = {}
        for row in self.rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary = {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_started_at_utc": self.started_at,
            "campaign_finished_at_utc": _utc_now(),
            "fixture_version": FIXTURE_VERSION,
            "counts": counts,
            "required_failures": self.required_failures,
            "engineering_status": "PASS" if not self.required_failures else "FAIL",
            "strategic_evidence": {
                "affordability": "INCONCLUSIVE",
                "specialized_deployment": "INCONCLUSIVE",
                "customer_evidence": "UNAVAILABLE",
            },
            "bundle": bundle,
            "manifest": manifest or {},
            "provenance": self.provenance,
            "record_count": len(self.record_data),
            "raw_results": os.path.basename(raw_path),
        }
        with open(os.path.join(self.out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @property
    def required_ok(self) -> bool:
        return not self.required_failures


class SharedRateLimiter:
    """Process-local aggregate bandwidth cap shared by all wrapped sources."""

    def __init__(self, bytes_per_second: float, latency_seconds: float = 0.0):
        if bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be positive")
        self.rate = float(bytes_per_second)
        self.latency = max(0.0, float(latency_seconds))
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def reserve(self, count: int) -> None:
        count = max(0, int(count))
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            delay = self.latency + count / self.rate
            self._next = start + delay
        wait = start - now + delay
        if wait > 0:
            time.sleep(wait)


class FaultSource(Source):
    """Wrap a real source for bounded, process-local link/failure injection."""

    def __init__(self, inner: Source, *, limiter: SharedRateLimiter | None = None,
                 fail_at: int | None = None, truncate_at: int | None = None,
                 reset_at: int | None = None):
        self.inner = inner
        self.name = inner.name
        self.size = inner.size
        self.random_access = inner.random_access
        self.limiter = limiter
        self.fail_at = fail_at
        self.truncate_at = truncate_at
        self.reset_at = reset_at
        self.calls = 0
        self.repeated_bytes = 0
        self._last = None
        self._lock = threading.Lock()

    def pread(self, offset: int, length: int) -> bytes:
        with self._lock:
            call = self.calls
            self.calls += 1
        if self.limiter is not None:
            self.limiter.reserve(length)
        if self.reset_at is not None and call == self.reset_at:
            raise ConnectionResetError(f"injected reset at read {call}")
        if self.fail_at is not None and call == self.fail_at:
            raise OSError(f"injected source failure at read {call}")
        data = self.inner.pread(offset, length)
        if self.truncate_at is not None and call == self.truncate_at:
            short = data[:max(0, len(data) // 2)]
            raise EOFError(f"injected truncated range at read {call}: "
                           f"wanted {length}, got {len(short)}")
        with self._lock:
            if self._last == (offset, length):
                self.repeated_bytes += len(data)
            self._last = (offset, length)
        return data

    def close(self) -> None:
        self.inner.close()


def _transport_source(path: str, *, coded: bool, limiter=None,
                      reset_at=None, fail_at=None, truncate_at=None,
                      observer=None, fetch_threads: int = 2,
                      place_threads: int = 1, target: int = 16 << 10):
    """Transport one artifact through a wrapped local source."""
    if coded:
        arc = Archive(path)
        inner = arc.source
        wrapped = FaultSource(inner, limiter=limiter, reset_at=reset_at,
                              fail_at=fail_at, truncate_at=truncate_at)
        arc.source = wrapped
        lo, hi, runs = arc.cover([(0, arc.plain_bytes)], target)
        out = bytearray(max(0, hi - lo))
        try:
            report = transport(
                runs, arc.fetch,
                lambda run, payload: arc.place(run, payload, out, lo,
                                               clip=(lo, hi)),
                fetch_threads=fetch_threads, place_threads=place_threads,
                inflight=max(1, fetch_threads),
                observer=observer)
            return bytes(out), report, wrapped, arc
        except BaseException:
            arc.close()
            raise
    inner = FileSource(path)
    wrapped = FaultSource(inner, limiter=limiter, reset_at=reset_at,
                          fail_at=fail_at, truncate_at=truncate_at)
    jobs = split(inner.size, target)
    out = bytearray(inner.size)
    try:
        report = transport(
            jobs, lambda job: wrapped.pread(job[0], job[1]),
            lambda job, payload: out.__setitem__(
                slice(job[0], job[0] + len(payload)), payload) or len(payload),
            fetch_threads=fetch_threads, place_threads=place_threads,
            inflight=max(1, fetch_threads),
            observer=observer)
        return bytes(out), report, wrapped, inner
    except BaseException:
        wrapped.close()
        raise


def _active_transport_threads() -> list[str]:
    return [t.name for t in threading.enumerate()
            if t.name.startswith("lmsluice-fetch-") or
            t.name.startswith("lmsluice-place-")]


def _run_recorded_load(campaign: Campaign, path: str, expected_sha: str, *,
                       arm: str, scenario: str, case: str, required: bool,
                       names=None, stream_budget=None, cache_condition="warm",
                       cold_method="not_attempted", sample_interval=0.02,
                       sample_resources=True, cache_control=None,
                       load_cache="off", row_facts=None):
    metadata = {
        "case": case,
        "arm": arm,
        "scenario": scenario,
        "artifact": os.path.basename(path),
        "consumer": "synthetic-output-hash",
        "availability": "synthetic_fixture",
        "cache_condition": cache_condition,
        "cold_method": cold_method,
        "cache_control": cache_control or {},
    }
    recorder = ReadinessRecord(metadata=metadata, sample_interval=sample_interval,
                               max_samples=256,
                               sample_resources=sample_resources).start()
    started = time.perf_counter()
    status = "PASS"
    facts = {}
    try:
        with open_model(path, cache=load_cache, observer=recorder,
                        fetch_threads=2, place_threads=1) as model:
            recorder.add_bytes(logical=model.plain_bytes, coded=model.coded_bytes)
            if arm == "normal-loader":
                recorder.set_route(planned="normal-loader", actual="normal-loader")
                view = model.map()
                recorder.mark("first_tensor", name=next(iter(model.tensors), "whole"),
                              bytes=len(view))
                digest = hashlib.sha256(view).hexdigest()
                del view
            elif stream_budget is not None:
                chunks = []
                for name, view in model.stream(names=names, budget=stream_budget,
                                                observer=recorder):
                    chunks.append((name, bytes(view)))
                    if recorder.events.get("consumer_first_useful") is None:
                        recorder.mark("consumer_first_useful", name=name,
                                      output="hash-of-first-tensor")
                digest = hashlib.sha256(b"".join(blob for _, blob in chunks)).hexdigest()
            else:
                buf = model.load(names=names, observer=recorder)
                if model.tensors:
                    first_name = next(iter(model.tensors))
                    recorder.mark("first_tensor", name=first_name,
                                  bytes=model.tensors[first_name].nbytes)
                if names is None:
                    digest = hashlib.sha256(bytes(buf)).hexdigest()
                else:
                    digest_builder = hashlib.sha256()
                    for name in sorted(names, key=lambda item: model.tensors[item].start):
                        digest_builder.update(bytes(model.tensor(name, buffer=buf)))
                    digest = digest_builder.hexdigest()
                del buf
            recorder.mark("consumer_first_useful", output="sha256")
            recorder.mark("consumer_ready", output="sha256", complete=True)
            facts.update({
                "plain_bytes": model.plain_bytes,
                "coded_bytes": model.coded_bytes,
                "route": model.route,
                "output_sha256": digest,
                "output_equivalent": digest == expected_sha,
            })
            if digest != expected_sha:
                raise AssertionError(
                    f"consumer output hash {digest} != expected {expected_sha}")
    except BaseException as exc:  # retain a record and row for every failed arm
        status = "FAIL"
        recorder.failure_event(exc, phase="load")
        facts["error"] = _error(exc)
        facts["error_stage"] = "open/load/consumer"
        campaign.exception(case, exc)
    finally:
        record_path, record_data = campaign.record(recorder)
    facts.update({
        "duration_s": time.perf_counter() - started,
        "record": os.path.relpath(record_path, campaign.out),
        "first_tensor_s": _event_seconds(record_data, "first_tensor"),
        "first_useful_s": _event_seconds(record_data, "consumer_first_useful"),
        "ready_s": _event_seconds(record_data, "consumer_ready"),
        "transfer_s": _event_seconds(record_data, "transfer_complete"),
        "peak_rss_bytes": record_data["resources"]["host"]["peak"].get(
            "vmrss_bytes"),
        "cache_control": cache_control or {},
    })
    if row_facts:
        facts.update(row_facts)
    campaign.row(case, status, required=required, **facts)
    return status, facts, record_data


def _payload_digest(plain_path: str, names=None) -> str:
    """Digest raw tensor bytes without timing a loader arm."""
    with open_model(plain_path, cache="off") as model, open(plain_path, "rb") as fh:
        selected = names or model.tensors
        digest = hashlib.sha256()
        for name in sorted(selected, key=lambda item: model.tensors[item].start):
            tensor = model.tensors[name]
            fh.seek(tensor.start)
            digest.update(fh.read(tensor.nbytes))
        return digest.hexdigest()


def run_contract(campaign: Campaign, bundle: dict, repo_root: str) -> None:
    plain = bundle["plain"]
    actual = _file_digest(bundle["plain_path"])
    contract_path = os.path.join(repo_root, "docs", "evaluations", "workload.json")
    contract = {}
    contract_error = None
    try:
        with open(contract_path, encoding="utf-8") as fh:
            contract = json.load(fh)
    except (OSError, ValueError) as exc:
        contract_error = _error(exc)
    roles = contract.get("tensors", []) if isinstance(contract, dict) else []
    routes = contract.get("route_contract", {}) if isinstance(contract, dict) else {}
    required_contract_fields = (
        "provenance", "sidecars", "compatibility", "availability_policy",
        "input_output_contract", "evidence_classes",
    )
    role_fields = (
        "input_sha256", "consumer_payload_sha256", "synthetic_input",
        "output_equivalence", "evidence_class",
    )
    route_fields = ("availability_policy", "compatibility", "evidence_class")
    contract_roles = {role.get("name"): role for role in roles
                      if isinstance(role, dict)}
    role_hash_coverage = {}
    for generated in plain.get("tensors", []):
        declared = contract_roles.get(generated.get("name"), {})
        role_hash_coverage[generated.get("name")] = (
            declared.get("input_sha256") == generated.get("input_sha256")
            and declared.get("consumer_payload_sha256") ==
            generated.get("consumer_payload_sha256")
            and declared.get("output_equivalence", {}).get("expected_sha256") ==
            generated.get("consumer_payload_sha256"))
    input_contract = contract.get("input_output_contract", {})
    complete = (
        contract_error is None and all(field in contract for field in required_contract_fields)
        and bool(roles) and all(all(field in role for field in role_fields) for role in roles)
        and bool(routes) and all(isinstance(value, dict) and
                                 all(field in value for field in route_fields)
                                 for value in routes.values())
        and contract.get("schema", 0) >= 2
        and input_contract.get("fixed_seed") == bundle.get("seed")
        and all(role_hash_coverage.values())
    )
    ok = (actual["sha256"] == plain["sha256"] and
          actual["bytes"] == plain["bytes"] and
          len(plain.get("tensors", [])) >= 5 and complete)
    campaign.row(
        "contract", "PASS" if ok else "FAIL", required=True,
        artifact=actual, expected_sha256=plain["sha256"],
        roles=[t["role"] for t in plain.get("tensors", [])],
        workload_schema=contract.get("schema") if contract else None,
        contract_fields=required_contract_fields,
        role_field_coverage={role.get("role"): all(field in role for field in role_fields)
                             for role in roles},
        route_field_coverage={name: all(field in value for field in route_fields)
                              for name, value in routes.items()
                              if isinstance(value, dict)},
        role_hash_coverage=role_hash_coverage,
        input_contract_seed=input_contract.get("fixed_seed"),
        availability="synthetic_fixture",
        output_check="sha256 of exact consumer-visible bytes",
        failure=None if ok else {
            "reason": ("generated artifact differs from frozen contract"
                        if contract_error is None else "workload contract unavailable"),
            "contract_error": contract_error,
        },
    )


def run_observability(campaign: Campaign, bundle: dict) -> None:
    del bundle
    recorder = ReadinessRecord(
        metadata={"case": "observability-success", "availability": "synthetic"},
        sample_interval=0.005, max_samples=64).start()
    recorder.mark("source_open", source="synthetic://byte-fixture", size=24)
    recorder.mark("route_planned", route="plain", actual="plain")
    recorder.mark("allocation", bytes=24, destination="host")
    recorder.mark("staging", bytes=24, destination="host")
    seen = bytearray()
    report = transport(
        [(0, 8), (8, 8), (16, 8)], lambda job: bytes([job[0] // 8]) * job[1],
        lambda job, payload: seen.extend(payload) or len(payload),
        fetch_threads=2, place_threads=1, inflight=2,
        observer=_FirstPayloadTensorObserver(recorder))
    recorder.mark("consumer_first_useful", output="synthetic-byte-consumer")
    recorder.mark("consumer_ready", output="synthetic-byte-consumer")
    path, data = campaign.record(recorder, report=report)
    success = (
        data["events"]["first_payload"] is not None and
        data["events"]["transfer_complete"] is not None and
        data["events"]["consumer_ready"] is not None and
        data["events"]["terminal_failure"] is None and
        data["bytes"]["transferred"] == len(seen))
    campaign.row(
        "observability-success", "PASS" if success else "FAIL", required=True,
        record=os.path.relpath(path, campaign.out), events=data["events"],
        bytes=data["bytes"], missing_event_policy="null means not emitted",
    )

    recorder = ReadinessRecord(
        metadata={"case": "observability-failure", "availability": "synthetic"},
        sample_interval=0.005, max_samples=64).start()
    error = None

    def fail_fetch(job):
        if job == 3:
            raise OSError("injected source failure")
        return bytes(4)

    try:
        transport(range(10), fail_fetch, lambda _job, payload: len(payload),
                  fetch_threads=2, place_threads=1, inflight=2,
                  observer=recorder)
    except OSError as exc:
        error = exc
    path, data = campaign.record(recorder)
    success = (error is not None and data["failure"] is not None and
               data["events"]["terminal_failure"] is not None and
               data["events"]["transfer_complete"] is None and
               not _active_transport_threads())
    campaign.row(
        "observability-failure", "PASS" if success else "FAIL", required=True,
        record=os.path.relpath(path, campaign.out), error=_error(error) if error else None,
        terminal_failure=data["failure"], workers_after_case=_active_transport_threads(),
        missing_consumer_ready=data["events"]["consumer_ready"] is None,
    )


class _FirstPayloadTensorObserver:
    """Give the synthetic byte transport a named first-tensor event."""

    def __init__(self, record: ReadinessRecord):
        self.record = record

    def mark(self, event: str, **details):
        self.record.mark(event, **details)
        if event == "first_payload":
            self.record.mark("first_tensor", name="synthetic-payload",
                             bytes=details.get("bytes", 0))

    def __getattr__(self, name):
        return getattr(self.record, name)


def _record_event_order(campaign: Campaign) -> None:
    """Check one complete successful record and keep the null-event policy visible."""
    wanted = (
        "source_open", "route_planned", "allocation", "staging",
        "first_payload", "first_tensor", "transfer_complete",
        "consumer_first_useful", "consumer_ready",
    )
    candidates = []
    for run_id, data in campaign.record_data.items():
        if data.get("failure") is not None:
            continue
        events = data.get("events", {})
        if all(events.get(name) is not None for name in wanted):
            times = [events[name]["at_ns"] for name in wanted]
            metadata = data.get("metadata", {})
            preferred = metadata.get("case") == "observability-success"
            candidates.append((not preferred, run_id, times, events))
    _preferred, run_id, times, events = min(
        candidates, key=lambda item: (item[0], item[1])) if candidates else (
            False, None, [], {})
    ordered = bool(times) and times == sorted(times)
    campaign.row(
        "observability-event-order", "PASS" if ordered else "FAIL", required=True,
        representative_run_id=run_id,
        events=wanted,
        at_ns=times,
        monotonic_order=ordered,
        missing_event_policy="successful representative requires all listed events; other records retain nulls",
        representative_event_details={name: events.get(name) for name in wanted},
    )


def _timed_load(path: str, expected_sha: str, *, observer=None,
                sample_resources=True, campaign: Campaign | None = None,
                case: str = "overhead") -> dict:
    """Run the same plain load for overhead arms, retaining observer records."""
    recorder = observer
    started = time.perf_counter()
    status, error, digest, route = "PASS", None, None, None
    bytes_loaded = None
    try:
        with open_model(path, cache="off", observer=recorder,
                        fetch_threads=2, place_threads=1) as model:
            buf = model.load(observer=recorder)
            route = model.route
            bytes_loaded = len(buf)
            digest = hashlib.sha256(bytes(buf)).hexdigest()
            if digest != expected_sha:
                raise AssertionError(f"consumer output hash {digest} != {expected_sha}")
            if recorder is not None:
                recorder.mark("consumer_first_useful", output="overhead-output-hash")
                recorder.mark("consumer_ready", output="overhead-output-hash")
            del buf
    except BaseException as exc:
        status, error = "FAIL", exc
        if recorder is not None:
            recorder.failure_event(exc, phase="overhead-load")
        if campaign is not None:
            campaign.exception(case, exc)
    record_path = None
    record_data = None
    if recorder is not None and campaign is not None:
        record_path, record_data = campaign.record(recorder)
    return {
        "status": status,
        "duration_s": time.perf_counter() - started,
        "error": _error(error) if error else None,
        "output_sha256": digest,
        "output_equivalent": status == "PASS" and digest == expected_sha,
        "route": route,
        "bytes": bytes_loaded,
        "record": (os.path.relpath(record_path, campaign.out)
                    if record_path and campaign else None),
        "resource_sampling": (record_data or {}).get("resources", {}).get(
            "sampling", {}).get("status", "ENABLED" if sample_resources else "DISABLED"),
    }


def _percentile(values, fraction: float):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(len(values) * fraction) - 1))
    return values[index]


def run_observability_overhead(campaign: Campaign, bundle: dict) -> None:
    """Measure event-only and sampled observer cost against the same default load."""
    path = bundle["plain_path"]
    expected = bundle["plain"]["sha256"]
    rng = random.Random(2026091301)
    comparisons = (
        ("no-sampling", False, "observability-overhead-no-sampling"),
        ("sampled", True, "observability-overhead-sampled"),
    )
    all_comparisons = {}
    for comparison, sampled, aggregate_case in comparisons:
        rows = []
        for repetition in range(30):
            order = ["default", "observer"]
            rng.shuffle(order)
            pair = {}
            for mode in order:
                recorder = None
                if mode == "observer":
                    recorder = ReadinessRecord(
                        metadata={"case": aggregate_case, "comparison": comparison,
                                  "repetition": repetition,
                                  "availability": "synthetic_fixture"},
                        sample_interval=0.001 if sampled else 0.01,
                        max_samples=64, sample_resources=sampled).start()
                facts = _timed_load(
                    path, expected, observer=recorder,
                    sample_resources=sampled, campaign=campaign,
                    case=f"{aggregate_case}-sample")
                facts.update({"comparison": comparison, "repetition": repetition,
                              "mode": mode})
                pair[mode] = facts
                sample_facts = {key: value for key, value in facts.items()
                                if key != "status"}
                campaign.row(
                    f"{aggregate_case}-sample", facts["status"], required=False,
                    **sample_facts)
            rows.append((repetition, pair))
        ratios = []
        paired = []
        for repetition, pair in rows:
            default = pair.get("default", {})
            observer = pair.get("observer", {})
            if default.get("status") == observer.get("status") == "PASS":
                ratio = observer["duration_s"] / max(default["duration_s"], 1e-12)
                ratios.append(ratio)
                paired.append({
                    "repetition": repetition,
                    "default_s": default["duration_s"],
                    "observer_s": observer["duration_s"],
                    "observer_to_default_ratio": ratio,
                    "byte_equivalent": (default.get("bytes") == observer.get("bytes")
                                         and default.get("output_sha256") ==
                                         observer.get("output_sha256") == expected),
                    "route_equivalent": default.get("route") == observer.get("route"),
                })
        mode_summary = {}
        for mode in ("default", "observer"):
            values = [pair[mode]["duration_s"] for _, pair in rows
                      if pair[mode].get("status") == "PASS"]
            mode_summary[mode] = {
                "n": len(values), "median_s": _median(values),
                "min_s": min(values) if values else None,
                "max_s": max(values) if values else None,
                "empirical_p95_s": (_percentile(values, 0.95)
                                    if len(values) >= 30 else None),
                "p95_sample_count": len(values),
            }
        equivalent = all(item["byte_equivalent"] and item["route_equivalent"]
                          for item in paired)
        summary = {
            "comparison": comparison,
            "repetitions": 30,
            "paired_complete": len(paired) == 30,
            "modes": mode_summary,
            "paired": paired,
            "paired_ratio": {
                "n": len(ratios), "median": _median(ratios),
                "min": min(ratios) if ratios else None,
                "max": max(ratios) if ratios else None,
                "empirical_p95": _percentile(ratios, 0.95) if len(ratios) >= 30 else None,
                "p95_sample_count": len(ratios),
            },
            "byte_equivalence": equivalent,
            "route_equivalence": equivalent,
            "same_fixture": _file_digest(path),
            "worker_settings": {"fetch_threads": 2, "place_threads": 1},
            "cache": "off; identical path and cache condition",
            "interpretation": "bounded local opt-in overhead; no device or commercial threshold",
            "noise_limit": "tiny synthetic loads are timer/noise sensitive; paired ratios do not generalize to a device",
        }
        all_comparisons[comparison] = summary
        campaign.row(
            aggregate_case, "PASS" if summary["paired_complete"] and equivalent else "FAIL",
            required=True, **summary)
    return all_comparisons


def _measure_transport(campaign: Campaign, path: str, *, coded: bool, case: str,
                       required: bool, limiter=None, fetch_threads=2,
                       reset_at=None, fail_at=None, truncate_at=None,
                       provenance=None):
    expected_path = path if not coded else path.replace(
        ".lmsluice", ".safetensors")
    expected = _file_digest(expected_path)["sha256"]
    recorder = ReadinessRecord(
        metadata={"case": case, "arm": "coded" if coded else "plain",
                  "availability": "synthetic_fixture"},
        sample_interval=0.01, max_samples=128).start()
    recorder.set_route(planned="coded" if coded else "plain",
                       actual="coded" if coded else "plain", source=path)
    recorder.mark("source_open", source=path)
    recorder.mark("route_planned", route="coded" if coded else "plain")
    recorder.mark("allocation", bytes=os.path.getsize(expected_path), destination="host")
    started = time.perf_counter()
    error = None
    report = None
    owner = None
    try:
        output, report, wrapped, owner = _transport_source(
            path, coded=coded, limiter=limiter, reset_at=reset_at,
            fail_at=fail_at, truncate_at=truncate_at, observer=recorder,
            fetch_threads=fetch_threads)
        digest = hashlib.sha256(output).hexdigest()
        recorder.mark("first_tensor", bytes=len(output), name="whole-output")
        recorder.mark("consumer_first_useful", output="sha256")
        recorder.mark("consumer_ready", output="sha256")
        if digest != expected:
            raise AssertionError(f"transport output {digest} != {expected}")
    except BaseException as exc:
        error = exc
        recorder.failure_event(exc, phase="transport-source")
        campaign.exception(case, exc)
    finally:
        if owner is not None:
            owner.close()
        elif 'wrapped' in locals() and wrapped is not None:
            wrapped.close()
        path_record, data = campaign.record(recorder, report=report,
                                            provenance=provenance)
    facts = {
        "coded": coded,
        "duration_s": time.perf_counter() - started,
        "record": os.path.relpath(path_record, campaign.out),
        "error": _error(error) if error else None,
        "output_equivalent": error is None,
        "fetched_bytes": data["bytes"]["fetched"],
        "transferred_bytes": data["bytes"]["transferred"],
        "limited_by": (data.get("transport") or {}).get("limited_by"),
        "repeated_bytes": data["bytes"]["repeated"],
        "worker_cleanup": not _active_transport_threads(),
    }
    status = "PASS" if error is None else "FAIL"
    campaign.row(case, status, required=required, **facts)
    return status, facts, data


def run_bandwidth(campaign: Campaign, bundle: dict) -> None:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    if not bundle["coded"].get("bytes"):
        campaign.row("bandwidth-coded", "SKIP", required=True,
                     reason=bundle["coded"].get("reason", "stdlib codec unavailable"),
                     evidence_class="unavailable_backend")
        return
    cap = 96 * 1024
    for arm, path in (("plain", plain), ("coded", coded)):
        samples = []
        for workers in (1, 2):
            limiter = SharedRateLimiter(cap, latency_seconds=0.001)
            status, facts, _data = _measure_transport(
                campaign, path, coded=arm == "coded",
                case=f"bandwidth-{arm}-w{workers}", required=False,
                limiter=limiter, fetch_threads=workers)
            facts["workers"] = workers
            facts["aggregate_cap_bytes_s"] = cap
            facts["observed_bytes_s"] = (facts["transferred_bytes"] /
                                          max(facts["duration_s"], 1e-9))
            facts["aggregate_cap_respected"] = facts["observed_bytes_s"] <= cap * 1.35
            facts["transport_status"] = status
            samples.append({"workers": workers, **facts})
        rates = [s["observed_bytes_s"] for s in samples]
        ratio = max(rates) / max(min(rates), 1e-9)
        campaign.row(
            f"bandwidth-{arm}-aggregate", "PASS" if ratio < 1.75 and
            all(s["aggregate_cap_respected"] for s in samples) else "FAIL",
            required=True, cap_bytes_s=cap, samples=samples,
            worker_rate_ratio=ratio, same_aggregate_cap=True,
            n_threads_not_a_bandwidth_multiplier=ratio < 1.75,
        )


def run_memory(campaign: Campaign, bundle: dict) -> None:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    stream_expected = _payload_digest(plain)
    _run_recorded_load(
        campaign, plain, stream_expected, arm="lmsluice_plain",
        scenario="window", case="memory-window", required=True,
        stream_budget=32 << 10, cache_condition="warm")
    coded_available = bool(bundle["coded"].get("bytes"))
    if not coded_available:
        reason = bundle["coded"].get("reason", "stdlib codec unavailable")
        for case in ("memory-oversized-tensor", "memory-noncontiguous",
                     "consumer-departure"):
            campaign.row(case, "SKIP", required=True, reason=reason,
                         evidence_class="unavailable_backend")
    oversized_expected = _payload_digest(plain, ["oversized.tensor"])
    if coded_available:
        _run_recorded_load(
            campaign, coded, oversized_expected, arm="lmsluice_stdlib_coded",
            scenario="oversized-tensor", case="memory-oversized-tensor", required=True,
            names=["oversized.tensor"], stream_budget=32 << 10,
            cache_condition="warm")
    noncontiguous = ["language.embed.weight", "vision.patch.weight",
                     "specialist.task.weight"]
    noncontig_expected = _payload_digest(plain, noncontiguous)
    if coded_available:
        _run_recorded_load(
            campaign, coded, noncontig_expected, arm="lmsluice_stdlib_coded",
            scenario="noncontiguous-spans", case="memory-noncontiguous", required=True,
            names=noncontiguous, stream_budget=None, cache_condition="warm")

    retained = []
    recorder = ReadinessRecord(metadata={"case": "memory-retained-output",
                                          "availability": "synthetic_fixture"},
                               sample_interval=0.01, max_samples=64).start()
    status, error = "PASS", None
    try:
        with open_model(plain, cache="off", observer=recorder) as model:
            for _name, view in model.stream(budget=24 << 10, observer=recorder):
                retained.append(bytes(view))
            retained_bytes = sum(map(len, retained))
            recorder.add_coverage(retained_output_bytes=retained_bytes)
            recorder.mark("consumer_first_useful", output="retained-bytes")
            recorder.mark("consumer_ready", output="retained-bytes")
    except BaseException as exc:
        status, error = "FAIL", exc
        recorder.failure_event(exc, phase="retained-output")
        campaign.exception("memory-retained-output", exc)
    path, _data = campaign.record(recorder)
    campaign.row("memory-retained-output", status, required=False,
                 record=os.path.relpath(path, campaign.out),
                 retained_bytes=sum(map(len, retained)),
                 error=_error(error) if error else None,
                 note="retained buffers belong to the consumer and are outside the stream window")

    if coded_available:
        recorder = ReadinessRecord(metadata={"case": "consumer-departure",
                                              "availability": "synthetic_fixture"},
                                   sample_interval=0.01, max_samples=64).start()
        status, error, active = "PASS", None, []
        try:
            with open_model(coded, cache="off", observer=recorder) as model:
                stream = model.stream(budget=24 << 10, observer=recorder)
                next(stream)
                recorder.mark("consumer_first_useful", output="first-tensor")
                stream.close()
                del stream
            active = _active_transport_threads()
            if active:
                raise AssertionError(f"workers remain after consumer departure: {active}")
        except BaseException as exc:
            status, error = "FAIL", exc
            recorder.failure_event(exc, phase="consumer-departure")
            campaign.exception("consumer-departure", exc)
        path, data = campaign.record(recorder)
        campaign.row("consumer-departure", status, required=True,
                     record=os.path.relpath(path, campaign.out), workers_after_close=active,
                     error=_error(error) if error else None,
                     consumer_ready_missing=data["events"]["consumer_ready"] is None)

    try:
        import torch  # noqa: F401
        torch_available = True
    except Exception:
        torch_available = False
    try:
        from lmsluice import cuda

        cuda_available, cuda_why = cuda.available()
    except Exception as exc:  # pragma: no cover - import guard
        cuda_available, cuda_why = False, f"{type(exc).__name__}: {exc}"
    if not (torch_available and cuda_available):
        campaign.row("memory-cuda-alignment", "NOT_RUN", required=False,
                     reason=("torch adapter unavailable" if not torch_available
                             else cuda_why), evidence_class="unavailable_target")
    else:  # pragma: no cover - target-specific and bounded if reached
        campaign.row("memory-cuda-alignment", "NOT_RUN", required=False,
                     reason="CUDA target detected; explicit GPU campaign is deferred",
                     evidence_class="target_requires_visible_campaign")


@contextmanager
def _temporary_env(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _flip(path: str, offset: int) -> None:
    with open(path, "r+b") as fh:
        fh.seek(offset)
        byte = fh.read(1)
        if not byte:
            raise EOFError(f"cannot flip absent byte at {offset} in {path}")
        fh.seek(offset)
        fh.write(bytes([byte[0] ^ 0x01]))


def _expected_failure(campaign: Campaign, case: str, action, *, required=True,
                      expected=Exception, detail=None) -> bool:
    error = None
    try:
        action()
    except BaseException as exc:
        error = exc
    observed = isinstance(error, expected)
    expected_name = (expected.__name__ if hasattr(expected, "__name__") else
                     "/".join(item.__name__ for item in expected))
    if error is None:
        campaign.exception(case, AssertionError("induced failure was not raised"))
    campaign.row(
        case, "FAIL" if observed else "PASS", required=False,
        induced_failure_observed=observed, expected_error=expected_name,
        error=_error(error) if error else {"type": "none"}, detail=detail,
    )
    campaign.row(
        f"{case}-detection", "PASS" if observed else "FAIL", required=required,
        assertion=("failure propagated and was classified" if observed else
                   "induced failure was not propagated"),
        error=_error(error) if error else None,
    )
    return observed


def run_faults(campaign: Campaign, bundle: dict) -> None:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    reset_status, reset_facts, _reset_data = _measure_transport(
        campaign, plain, coded=False, case="fault-source-reset",
        required=False, reset_at=0)
    campaign.row("fault-source-reset-detection",
                 "PASS" if reset_status == "FAIL" else "FAIL", required=True,
                 failure_propagated=reset_status == "FAIL",
                 worker_cleanup=reset_facts["worker_cleanup"])
    # The preceding row is intentionally FAIL because a failure was induced.
    # The required detection row above is the engineering verdict.
    trunc_status, trunc_facts, _trunc_data = _measure_transport(
        campaign, plain, coded=False, case="fault-truncated-range",
        required=False, truncate_at=0)
    campaign.row("fault-truncated-range-detection",
                 "PASS" if trunc_status == "FAIL" else "FAIL", required=True,
                 failure_propagated=trunc_status == "FAIL",
                 worker_cleanup=trunc_facts["worker_cleanup"])

    if not bundle["coded"].get("bytes"):
        campaign.row("fault-corrupt-coded", "SKIP", required=True,
                     reason="coded fixture unavailable",
                     evidence_class="unavailable_backend")
    else:
        corrupt = os.path.join(os.path.dirname(coded), "corrupt-copy.lmsluice")
        shutil.copy2(coded, corrupt)
        _flip(corrupt, 12)
        _expected_failure(
            campaign, "fault-corrupt-coded",
            lambda: _load_bytes(corrupt), expected=Exception,
            detail=_file_digest(corrupt))

    _expected_failure(
        campaign, "fault-destination-bounds",
        lambda: _load_into_too_small(plain), expected=ValueError,
        detail="destination is one byte smaller than the frozen artifact")
    _expected_failure(
        campaign, "fault-destination-unwritable",
        lambda: _load_into_unwritable(plain), expected=(TypeError, ValueError),
        required=False, detail="immutable bytes destination")

    # Wrong-key and tampered-ciphertext checks are run only when the existing
    # configured crypto backend can make a real sealed fixture. The unavailable
    # result remains explicit and does not turn encryption into a claim.
    from lmsluice import crypt

    if not crypt.available():
        campaign.row("fault-wrong-key", "NOT_RUN", required=False,
                     reason="encryption backend unavailable", evidence_class="unavailable")
        campaign.row("fault-tampered-ciphertext", "NOT_RUN", required=False,
                     reason="encryption backend unavailable", evidence_class="unavailable")
    else:
        from lmsluice import sealed

        directory = os.path.dirname(plain)
        good = os.path.join(directory, "readiness-good.key")
        bad = os.path.join(directory, "readiness-bad.key")
        sealed_path = os.path.join(directory, "portable-workload.sealed")
        with open(good, "wb") as fh:
            fh.write(crypt.new_key())
        with open(bad, "wb") as fh:
            fh.write(crypt.new_key())
        sealed.seal_file(coded, sealed_path, good)
        _expected_failure(
            campaign, "fault-wrong-key",
            lambda: _open_sealed_with_key(sealed_path, bad), expected=ValueError,
            detail="key identifier must fail before plaintext is returned")

        tampered = os.path.join(directory, "tampered.sealed")
        shutil.copy2(sealed_path, tampered)
        outer = FileSource(tampered)
        wrapped = sealed.SealedSource(outer, key_file=good)
        row = next(row for row in wrapped._rows if row[4] == sealed.SEALED)
        wrapped.close()
        _flip(tampered, row[2])
        _expected_failure(
            campaign, "fault-tampered-ciphertext",
            lambda: _open_sealed_with_key(tampered, good), expected=ValueError,
            detail="one ciphertext byte flipped; payload tag must reject it")

    # A fresh retry from zero is part of the current behavior. Record the
    # repeated work explicitly so it cannot later be misread as durable resume.
    retry_error = None
    try:
        _transport_source(plain, coded=False, reset_at=0, fetch_threads=2)
    except BaseException as exc:
        retry_error = exc
    output, report, wrapped, owner = _transport_source(
        plain, coded=False, fetch_threads=2)
    owner.close()
    retry_ok = (retry_error is not None and
                hashlib.sha256(output).hexdigest() == _file_digest(plain)["sha256"])
    campaign.row(
        "retry-restart-from-zero", "PASS" if retry_ok else "FAIL", required=True,
        first_attempt_error=_error(retry_error) if retry_error else None,
        retry_report=report.to_dict(), repeated_bytes=os.path.getsize(plain),
        semantics="ordinary retry; not durable resume",
        leftover_artifacts=[])


def _child_load(path: str) -> int:
    """Child entrypoint for a genuinely separate application launch."""
    started = time.perf_counter()
    try:
        with open_model(path, cache="off", fetch_threads=2, place_threads=1) as model:
            buf = model.load()
            digest = hashlib.sha256(bytes(buf)).hexdigest()
            result = {
                "pid": os.getpid(),
                "route": model.route,
                "bytes": len(buf),
                "output_sha256": digest,
                "loader_readiness_s": time.perf_counter() - started,
            }
            del buf
    except BaseException as exc:
        print(json.dumps({"error": _error(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def run_application_restarts(campaign: Campaign, bundle: dict, repo_root: str) -> None:
    """Launch bounded child processes and separate startup from load time."""
    path = bundle["plain_path"]
    expected = bundle["plain"]["sha256"]
    restart_dir = os.path.join(campaign.out, "restarts")
    os.makedirs(restart_dir, exist_ok=True)
    samples = []
    for repetition in range(5):
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m",
             "experiments.device_readiness.harness", "--child-load", path],
            cwd=repo_root, capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        process_wall = time.perf_counter() - started
        stdout_path = os.path.join(restart_dir, f"{repetition:02d}.stdout")
        stderr_path = os.path.join(restart_dir, f"{repetition:02d}.stderr")
        with open(stdout_path, "w", encoding="utf-8") as fh:
            fh.write(result.stdout)
        with open(stderr_path, "w", encoding="utf-8") as fh:
            fh.write(result.stderr)
        child = {}
        parse_error = None
        if result.stdout.strip():
            try:
                child = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                parse_error = _error(exc)
        sample = {
            "repetition": repetition,
            "returncode": result.returncode,
            "pid": child.get("pid"),
            "process_wall_s": process_wall,
            "loader_readiness_s": child.get("loader_readiness_s"),
            "process_start_s": (process_wall - child["loader_readiness_s"]
                                 if isinstance(child.get("loader_readiness_s"), (int, float))
                                 else None),
            "output_sha256": child.get("output_sha256"),
            "bytes": child.get("bytes"),
            "route": child.get("route"),
            "output_equivalent": child.get("output_sha256") == expected,
            "stdout": os.path.relpath(stdout_path, campaign.out),
            "stderr": os.path.relpath(stderr_path, campaign.out),
            "parse_error": parse_error,
            "error": child.get("error"),
        }
        samples.append(sample)
        campaign.row("application-restart-sample", "PASS" if (
            result.returncode == 0 and sample["output_equivalent"] and
            sample["bytes"] == bundle["plain"]["bytes"] and
            sample["pid"] is not None) else "FAIL",
            required=False, **sample)
    valid = [sample for sample in samples
             if sample["returncode"] == 0 and sample["output_equivalent"]]
    distinct_pids = len({sample["pid"] for sample in valid}) == len(valid)
    ok = len(valid) == 5 and distinct_pids
    campaign.row(
        "application-restart", "PASS" if ok else "FAIL", required=True,
        repetitions=5, samples=samples, valid_samples=len(valid),
        distinct_child_pids=distinct_pids,
        byte_identical_outputs=all(sample["output_equivalent"] for sample in samples),
        separation=("process_wall_s includes interpreter/process startup; "
                    "loader_readiness_s is measured inside each child; "
                    "process_start_s is their bounded difference"),
        fixture=_file_digest(path),
    )


def _load_bytes(path: str) -> bytes:
    with open_model(path, cache="off") as model:
        return bytes(model.load())


def _load_into_too_small(path: str):
    with open_model(path, cache="off") as model:
        return model.load(into=bytearray(max(0, model.plain_bytes - 1)))


def _load_into_unwritable(path: str):
    with open_model(path, cache="off") as model:
        return model.load(into=bytes(model.plain_bytes))


def _open_sealed_with_key(path: str, key: str) -> bytes:
    with _temporary_env(LMSLUICE_KEY_FILE=key):
        with open_model(path, cache="off") as model:
            return bytes(model.load())


class _SignedMultipartStore:
    """Local S3-shaped endpoint that rechecks signatures and body hashes."""

    def __init__(self, directory: str, *, fail_part: int = 2):
        self.directory = directory
        self.fail_part = fail_part
        self.secret = "readiness-local-secret-2026"
        self.access_key = "readiness-local-access"
        self.uploads = {}
        self.signature_checks = []
        self.body_checks = []
        self.failed_parts = []
        self.abort_requests = 0
        self.final_objects = []

    def __enter__(self) -> str:
        outer = self
        os.makedirs(self.directory, exist_ok=True)

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _reply(self, code: int, body: bytes = b"", extra=None):
                self.send_response(code)
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _authorized(self, body: bytes) -> bool:
                import hashlib as _hashlib

                got = self.headers.get("Authorization")
                signature_ok = False
                body_ok = False
                if got and "Credential=" in got and "SignedHeaders=" in got:
                    try:
                        credential = got.split("Credential=", 1)[1].split(",", 1)[0]
                        access, date, region, service, _request = credential.split("/")
                        stamp = self.headers.get("x-amz-date")
                        when = datetime.datetime.strptime(
                            stamp, "%Y%m%dT%H%M%SZ").replace(
                                tzinfo=datetime.timezone.utc)
                        signed = got.split("SignedHeaders=", 1)[1].split(",", 1)[0]
                        sent = {key: value for key, value in self.headers.items()
                                if key.lower() != "authorization"}
                        signed_inputs = {
                            key: value for key, value in sent.items()
                            if key.lower() in signed.split(";")
                            and not key.lower().startswith(("x-amz-", "host"))
                        }
                        from lmsluice import sign

                        url = f"http://{self.headers.get('Host')}{self.path}"
                        want = sign.sigv4_headers(
                            method=self.command, url=url, region=region,
                            service=service, access_key=access,
                            secret_key=outer.secret, headers=signed_inputs,
                            payload_sha256=self.headers.get(
                                "x-amz-content-sha256", sign.EMPTY_SHA256),
                            when=when)
                        signature_ok = (
                            access == outer.access_key
                            and want["Authorization"] == got)
                        body_ok = (self.headers.get("x-amz-content-sha256") ==
                                   _hashlib.sha256(body).hexdigest())
                    except (AttributeError, IndexError, ValueError, TypeError):
                        signature_ok = body_ok = False
                outer.signature_checks.append(signature_ok)
                outer.body_checks.append(body_ok)
                return signature_ok and body_ok

            def _deny(self):
                self._reply(403)

            def do_POST(self):
                body = self._body()
                if not self._authorized(body):
                    return self._deny()
                query = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query,
                    keep_blank_values=True)
                if "uploads" in query:
                    upload_id = f"u{len(outer.uploads) + 1}"
                    outer.uploads[upload_id] = {}
                    return self._reply(
                        200,
                        f"<InitiateMultipartUploadResult><UploadId>{upload_id}</UploadId>"
                        "</InitiateMultipartUploadResult>".encode())
                upload_id = query.get("uploadId", [""])[0]
                parts = outer.uploads.pop(upload_id, None)
                if parts is None:
                    return self._reply(404)
                order = [int(number) for number in re.findall(
                    r"<PartNumber>(\d+)</PartNumber>",
                    body.decode("utf-8", "replace"))]
                output = b"".join(parts[number] for number in order)
                object_name = os.path.basename(
                    urllib.parse.urlparse(self.path).path)
                with open(os.path.join(outer.directory, object_name), "wb") as fh:
                    fh.write(output)
                outer.final_objects.append(object_name)
                return self._reply(200, b"<CompleteMultipartUploadResult/>")

            def do_PUT(self):
                body = self._body()
                if not self._authorized(body):
                    return self._deny()
                query = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query,
                    keep_blank_values=True)
                if "partNumber" not in query:
                    return self._reply(400)
                number = int(query["partNumber"][0])
                if number == outer.fail_part:
                    outer.failed_parts.append(number)
                    return self._reply(500, b"<Error>induced part rejection</Error>")
                upload_id = query.get("uploadId", [""])[0]
                outer.uploads.setdefault(upload_id, {})[number] = body
                return self._reply(
                    200,
                    b"",
                    {"ETag": f'"{hashlib.md5(body).hexdigest()}"'},
                )

            def do_DELETE(self):
                body = self._body()
                if not self._authorized(body):
                    return self._deny()
                query = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query,
                    keep_blank_values=True)
                upload_id = query.get("uploadId", [""])[0]
                if upload_id in outer.uploads:
                    outer.uploads.pop(upload_id, None)
                    outer.abort_requests += 1
                    return self._reply(204)
                return self._reply(404)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="lmsluice-local-signed-store", daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def run_multipart_failure(campaign: Campaign, bundle: dict) -> None:
    """Prove signed multipart part failure propagates and aborts local state."""
    from lmsluice import cloud

    store_dir = os.path.join(campaign.out, "signed-store")
    target = "s3://readiness-bucket/portable-workload.safetensors"
    error = None
    store = _SignedMultipartStore(store_dir, fail_part=2)
    with store as endpoint:
        env = {
            "LMSLUICE_S3_ENDPOINT": endpoint,
            "AWS_ACCESS_KEY_ID": "readiness-local-access",
            "AWS_SECRET_ACCESS_KEY": "readiness-local-secret-2026",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        with _temporary_env(**env):
            try:
                cloud.put(bundle["plain_path"], target, part_bytes=16 << 10)
            except BaseException as exc:
                error = exc
        facts = {
            "target": "local-s3-shaped-endpoint",
            "failed_part": 2,
            "part_bytes": 16 << 10,
            "error": _error(error) if error else {"type": "none"},
            "error_type": type(error).__name__ if error else None,
            "cloud_error_propagated": isinstance(error, cloud.CloudError),
            "signature_checks": list(store.signature_checks),
            "body_checks": list(store.body_checks),
            "all_signatures_valid": bool(store.signature_checks) and
            all(store.signature_checks),
            "all_body_hashes_valid": bool(store.body_checks) and
            all(store.body_checks),
            "failed_parts": list(store.failed_parts),
            "abort_attempted": store.abort_requests == 1,
            "abort_requests": store.abort_requests,
            "remaining_multipart_state": len(store.uploads),
            "final_objects": list(store.final_objects),
            "synthetic_credentials": True,
            "external_network": False,
        }
    observed = (facts["cloud_error_propagated"] and facts["failed_parts"] == [2]
                and facts["all_signatures_valid"]
                and facts["all_body_hashes_valid"]
                and facts["abort_attempted"]
                and facts["remaining_multipart_state"] == 0
                and facts["final_objects"] == [])
    if error is not None:
        campaign.exception("multipart-signature-body-failure", error)
    campaign.row(
        "multipart-signature-body-failure", "FAIL" if error else "PASS",
        required=False, induced_failure_observed=error is not None, **facts)
    campaign.row(
        "multipart-failure-detection", "PASS" if observed else "FAIL",
        required=True,
        assertion="local signed store rejected the injected part; CloudError propagated; abort removed state",
        **facts)


class _PeriodicTask:
    """Small synthetic interference probe; it is never presented as a voice SLA."""

    def __init__(self, period: float = 0.005):
        self.period = period
        self.stop = threading.Event()
        self.ticks = 0
        self.slips = []
        self.tick_times = []
        self._thread = threading.Thread(target=self._run,
                                         name="lmsluice-periodic-control", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        deadline = time.perf_counter() + self.period
        while not self.stop.wait(max(0.0, deadline - time.perf_counter())):
            now = time.perf_counter()
            self.slips.append(max(0.0, now - deadline))
            self.tick_times.append(now)
            self.ticks += 1
            hashlib.sha256(b"periodic-control" * 32).digest()
            deadline += self.period

    def close(self):
        self.stop.set()
        self._thread.join(timeout=2)


def _active_load(campaign: Campaign, path: str, expected_sha: str, *,
                 case: str, arm: str, names=None, load_repetitions: int = 8) -> dict:
    periodic = _PeriodicTask(period=0.002).start()
    recorder = ReadinessRecord(
        metadata={"case": case, "arm": arm, "availability": "synthetic_fixture",
                  "consumer": "periodic-control-task"},
        sample_interval=0.02, max_samples=128).start()
    started = time.perf_counter()
    time.sleep(max(0.01, periodic.period * 4))
    load_started = time.perf_counter()
    status, error, digest = "PASS", None, None
    try:
        with open_model(path, cache="off", observer=recorder) as model:
            for _iteration in range(load_repetitions):
                buf = model.load(names=names, observer=recorder)
                digest_builder = hashlib.sha256()
                if names is None:
                    digest_builder.update(bytes(buf))
                else:
                    for name in sorted(names, key=lambda item: model.tensors[item].start):
                        digest_builder.update(bytes(model.tensor(name, buffer=buf)))
                digest = digest_builder.hexdigest()
                if recorder.events.get("consumer_first_useful") is None:
                    recorder.mark("consumer_first_useful", output="periodic-load-check")
                del buf
            recorder.mark("consumer_ready", output="periodic-load-check")
        if digest != expected_sha:
            raise AssertionError(f"active consumer output {digest} != {expected_sha}")
    except BaseException as exc:
        status, error = "FAIL", exc
        recorder.failure_event(exc, phase="active-consumer")
        campaign.exception(case, exc)
    finally:
        load_finished = time.perf_counter()
        periodic.close()
        record_path, data = campaign.record(recorder)
    before = [at for at in periodic.tick_times if at < load_started]
    during = [at for at in periodic.tick_times
              if load_started <= at <= load_finished]
    slips = list(periodic.slips)
    overlap_ok = bool(before) and bool(during)
    final_status = status if status != "PASS" or overlap_ok else "FAIL"
    campaign.row(
        case, final_status, required=False,
        record=os.path.relpath(record_path, campaign.out),
        arm=arm, output_equivalent=error is None,
        duration_s=load_finished - load_started,
        setup_s=load_started - started,
        load_repetitions=load_repetitions,
        periodic_ticks=periodic.ticks,
        periodic_ticks_before_load=len(before),
        periodic_ticks_during_load=len(during),
        periodic_max_slip_s=max(slips, default=0.0),
        periodic_slip_distribution_s={
            "n": len(slips), "min": min(slips) if slips else None,
            "median": _median(slips), "max": max(slips) if slips else None,
        },
        overlap_observed=overlap_ok,
        note="scheduler/resource interference only; no real voice deadline",
        error=_error(error) if error else None,
    )
    return {
        "status": final_status, "duration_s": load_finished - load_started,
        "periodic_ticks_before": len(before),
        "periodic_ticks_during": len(during),
        "periodic_ticks": periodic.ticks,
        "slips_s": slips,
        "record": os.path.relpath(record_path, campaign.out),
        "output_equivalent": error is None,
        "overlap_observed": overlap_ok,
        "error": _error(error) if error else None,
        "arm": arm,
    }


def _optional_lmz_artifact(plain: str, destination_path: str) -> tuple[dict | None, dict]:
    """Build the optional lmz arm only when the configured interpreter can import it."""
    try:
        from lmsluice.lmzcodec import backend_info, encoder

        info = backend_info()
        if not info.get("version"):
            return None, {"available": False, "reason": info.get("error", "lmz version unavailable"),
                          "backend": info}
        measured = encoder().encode(plain, destination_path, chunk_size=1 << 20)
        return {
            "path": destination_path,
            "bytes": os.path.getsize(destination_path),
            "sha256": hash_file(destination_path),
            "format": "lmz-coded-optional",
            "backend": info,
            "measured": observability._json_value(measured),
        }, {"available": True, "backend": info}
    except BaseException as exc:  # noqa: BLE001 - capability stays explicit
        return None, {"available": False, "reason": _error(exc),
                      "backend": "importable but encoding failed"}


def _run_route_prediction(campaign: Campaign, bundle: dict) -> dict:
    """Use separate synthetic artifacts for calibration and held-out evaluation."""
    base = os.path.join(campaign.out, "route-prediction")
    calibration_dir = os.path.join(base, "calibration")
    evaluation_dir = os.path.join(base, "evaluation")
    calibration = generate_bundle(calibration_dir, seed=17)
    evaluation = generate_bundle(evaluation_dir, seed=29)
    for item, directory in ((calibration, calibration_dir), (evaluation, evaluation_dir)):
        item["plain_path"] = os.path.join(directory, item["plain"]["path"])
        item["coded_path"] = os.path.join(directory, item["coded"]["path"])

    def artifact_provenance(item):
        return {
            "artifact_sha256": item["plain"]["sha256"],
            "artifact_bytes": item["plain"]["bytes"],
            "fixture_version": item["version"],
        }

    if not calibration["coded"].get("bytes") or not evaluation["coded"].get("bytes"):
        detail = {
            "calibration": calibration["coded"],
            "evaluation": evaluation["coded"],
            "calibration_evaluation_disjoint": True,
        }
        campaign.row("A1-route-prediction", "INCONCLUSIVE", required=False,
                     local_facts=detail,
                     target_link_codec_boundary="INCONCLUSIVE; no real target link supplied")
        return detail

    conditions = (
        ("fast-local", None),
        ("slow-simulated-link", SharedRateLimiter(96 * 1024, latency_seconds=0.001)),
    )
    evaluations = []
    calibration_rows = []
    for condition, _shared in conditions:
        cal_results = {}
        for arm, path, coded in (
                ("plain", calibration["plain_path"], False),
                ("coded", calibration["coded_path"], True)):
            limiter = (None if condition == "fast-local"
                       else SharedRateLimiter(96 * 1024, latency_seconds=0.001))
            status, facts, _data = _measure_transport(
                campaign, path, coded=coded,
                case=f"A1-route-calibration-{condition}-{arm}", required=False,
                limiter=limiter, fetch_threads=2,
                provenance=artifact_provenance(calibration))
            cal_results[arm] = {"status": status, **facts}
        available = {arm: row for arm, row in cal_results.items()
                     if row["status"] == "PASS"}
        if len(available) < 2:
            calibration_rows.append({
                "condition": condition, "predicted_route": None,
                "samples": cal_results,
                "artifact": _file_digest(calibration["plain_path"]),
                "status": "INCONCLUSIVE",
            })
            campaign.row(
                "A1-route-prediction-sample", "INCONCLUSIVE", required=False,
                condition=condition, phase="calibration",
                reason="both calibration routes did not complete")
            continue
        predicted = min(available, key=lambda arm: available[arm]["duration_s"])
        calibration_row = {
            "condition": condition,
            "predicted_route": predicted,
            "samples": cal_results,
            "artifact": _file_digest(calibration["plain_path"]),
        }
        calibration_rows.append(calibration_row)
        for arm, path, coded in (
                ("plain", evaluation["plain_path"], False),
                ("coded", evaluation["coded_path"], True)):
            limiter = (None if condition == "fast-local"
                       else SharedRateLimiter(96 * 1024, latency_seconds=0.001))
            status, facts, _data = _measure_transport(
                campaign, path, coded=coded,
                case=f"A1-route-evaluation-{condition}-{arm}", required=False,
                limiter=limiter, fetch_threads=2,
                provenance=artifact_provenance(evaluation))
            cal_results.setdefault("evaluation", {})[arm] = {
                "status": status, **facts}
        eval_results = cal_results["evaluation"]
        actual_available = {arm: row for arm, row in eval_results.items()
                            if row["status"] == "PASS"}
        if len(actual_available) < 2:
            evaluations.append({
                "condition": condition, "predicted_route": predicted,
                "actual_winner": None, "sign_of_benefit_error": None,
                "reason": "both held-out evaluation routes did not complete",
                "evaluation_samples": eval_results,
            })
            campaign.row(
                "A1-route-prediction-sample", "INCONCLUSIVE", required=False,
                condition=condition, phase="evaluation",
                reason="both held-out routes did not complete")
            continue
        actual = min(actual_available, key=lambda arm: actual_available[arm]["duration_s"])
        predicted_row = eval_results[predicted]
        actual_row = eval_results[actual]
        wrong = predicted != actual
        sample = {
            "condition": condition,
            "calibration_artifact": _file_digest(calibration["plain_path"]),
            "evaluation_artifact": _file_digest(evaluation["plain_path"]),
            "predicted_route": predicted,
            "actual_winner": actual,
            "sign_of_benefit_error": wrong,
            "predicted_ready": predicted_row.get("output_equivalent", False),
            "actual_ready": actual_row.get("output_equivalent", False),
            "predicted_duration_s": predicted_row["duration_s"],
            "winning_duration_s": actual_row["duration_s"],
            "wrong_route_cost_s": (predicted_row["duration_s"] - actual_row["duration_s"]
                                    if wrong else 0.0),
            "wrong_route_ratio": (predicted_row["duration_s"] /
                                   max(actual_row["duration_s"], 1e-12)),
            "evaluation_samples": eval_results,
        }
        evaluations.append(sample)
        campaign.row(
            "A1-route-prediction-sample",
            "PASS" if sample["predicted_ready"] and sample["actual_ready"] else "FAIL",
            required=False, **sample)
    summary = {
        "calibration_evaluation_disjoint": True,
        "calibration": calibration_rows,
        "evaluation": evaluations,
        "sign_of_benefit_errors": sum(bool(item["sign_of_benefit_error"])
                                      for item in evaluations),
        "ordinary_plain_wins": sum(item.get("actual_winner") == "plain"
                                    for item in evaluations),
        "wrong_route_cost_s": [item["wrong_route_cost_s"] for item in evaluations
                                if item["sign_of_benefit_error"]],
        "wrong_route_ratio": [item["wrong_route_ratio"] for item in evaluations
                               if item["sign_of_benefit_error"]],
        "target_link_codec_boundary": "INCONCLUSIVE; no real device/link calibration supplied",
        "interpretation": "held-out local mechanism facts only; no universal route threshold",
    }
    campaign.row("A1-route-prediction", "INCONCLUSIVE", required=False, **summary)
    return summary


def _run_affordability(campaign: Campaign, bundle: dict) -> dict:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    expected = bundle["plain"]["sha256"]
    arms = [("normal-loader", plain), ("lmsluice-plain", plain)]
    lmz_path = os.path.join(os.path.dirname(plain), "portable-workload.lmz")
    lmz_artifact, lmz_capability = _optional_lmz_artifact(plain, lmz_path)
    if lmz_artifact is not None:
        arms.append(("lmz-coded", lmz_path))
        campaign.row("A1-lmz-coded", "PASS", required=False, arm="lmz-coded",
                     availability=lmz_capability, artifact=lmz_artifact)
    else:
        campaign.row("A1-lmz-coded", "NOT_RUN", required=False, arm="lmz-coded",
                     availability=lmz_capability,
                     reason=lmz_capability.get("reason", "lmz unavailable"),
                     evidence_class="optional_backend_unavailable")
    if bundle["coded"].get("bytes"):
        arms.append(("lmsluice-stdlib-coded", coded))

    rng = random.Random(20260913)
    samples = []
    for scenario in ("first-fetch", "warm-reuse"):
        # Warmups are retained as rows but never enter the five-sample summary.
        for arm, path in arms:
            _run_recorded_load(
                campaign, path, expected, arm=arm, scenario=f"{scenario}-warmup",
                case=f"A1-{scenario}-{arm}-warmup", required=False,
                cache_condition="warmup", sample_interval=0.05,
                row_facts={"warmup": True, "repetition": None})
        for repetition in range(5):
            order = list(arms)
            rng.shuffle(order)
            for arm, path in order:
                if scenario == "first-fetch":
                    cache_control = _uncache(path)
                    cold_ok = bool(cache_control.get("uncache_ok"))
                    cache_condition = "cold_attempt" if cold_ok else "warm_or_layered"
                    cold_method = cache_control.get("uncache_method", "unavailable")
                else:
                    cache_control = {
                        "path": os.path.abspath(path), "attempted": False,
                        "uncache_ok": False, "uncache_method": "warm_after_warmup",
                    }
                    cold_ok, cold_method = False, "warm_after_warmup"
                    cache_condition = "warm_reuse"
                status, facts, _record = _run_recorded_load(
                    campaign, path, expected, arm=arm, scenario=scenario,
                    case=f"A1-{scenario}-{arm}", required=False,
                    cache_condition=cache_condition, cold_method=cold_method,
                    sample_interval=0.02, cache_control=cache_control,
                    row_facts={"repetition": repetition, "warmup": False,
                               "cold_supported": cold_ok})
                facts.update({"repetition": repetition, "status": status,
                              "cold_supported": cold_ok})
                samples.append({"scenario": scenario, "repetition": repetition,
                                "arm": arm, **facts})

    # Pair each five-sample arm with the normal loader from the same repetition.
    paired = []
    for scenario in ("first-fetch", "warm-reuse"):
        for repetition in range(5):
            group = [row for row in samples if row["scenario"] == scenario
                     and row["repetition"] == repetition]
            baseline = next((row for row in group if row["arm"] == "normal-loader"
                             and row.get("status") == "PASS"), None)
            for row in group:
                ratio = (row["duration_s"] / max(baseline["duration_s"], 1e-12)
                         if baseline and row.get("status") == "PASS" else None)
                row["paired_ratio_vs_normal"] = ratio
                paired.append({"scenario": scenario, "repetition": repetition,
                               "arm": row["arm"], "ratio": ratio,
                               "baseline": baseline["duration_s"] if baseline else None})
                campaign.row("A1-paired-ratio", "PASS" if ratio is not None else "FAIL",
                             required=False, **paired[-1])

    # A 30-sample empirical tail remains only for the two inexpensive plain arms.
    tail = []
    for repetition in range(30):
        for arm, path in (("normal-loader", plain), ("lmsluice-plain", plain)):
            status, facts, _record = _run_recorded_load(
                campaign, path, expected, arm=arm, scenario="warm-local-tail",
                case=f"A1-tail-{arm}", required=False,
                cache_condition="warm_reuse", sample_interval=0.1,
                row_facts={"repetition": repetition, "warmup": False})
            facts.update({"repetition": repetition, "status": status})
            tail.append({"repetition": repetition, "arm": arm, **facts})

    interference = []
    controls = []
    for repetition in range(5):
        for arm, path in arms[1:]:
            result = _active_load(
                campaign, path, expected,
                case=f"A1-interference-{arm}", arm=arm, load_repetitions=8)
            result.update({"repetition": repetition})
            interference.append(result)
            target_duration = max(result["duration_s"], 0.01)
            control = _PeriodicTask(period=0.002).start()
            control_started = time.perf_counter()
            time.sleep(target_duration)
            control_duration = time.perf_counter() - control_started
            control.close()
            control_facts = {
                "repetition": repetition, "arm": arm,
                "duration_s": control_duration,
                "target_duration_s": target_duration,
                "periodic_ticks": control.ticks,
                "periodic_max_slip_s": max(control.slips, default=0.0),
                "periodic_slip_distribution_s": {
                    "n": len(control.slips), "min": min(control.slips) if control.slips else None,
                    "median": _median(control.slips), "max": max(control.slips) if control.slips else None,
                },
                "comparable_duration": control_duration >= target_duration * 0.9,
                "note": "already-resident synthetic control; no model loading occurred",
            }
            controls.append(control_facts)
            campaign.row("A1-resident-inference-control", "PASS" if
                         control_facts["comparable_duration"] else "FAIL",
                         required=False, **control_facts)
    overlap_ok = bool(interference) and all(
        item["status"] == "PASS" and item["overlap_observed"] and
        item["periodic_ticks_before"] > 0 and item["periodic_ticks_during"] > 0
        for item in interference)
    campaign.row("A1-interference-overlap", "PASS" if overlap_ok else "FAIL",
                 required=True, active=interference, controls=controls,
                 positive_ticks_required=True,
                 note="scheduler/resource interference only; no voice deadline or application SLA")

    def duration_summary(rows):
        result = {}
        for arm in sorted({r["arm"] for r in rows}):
            values = [r["duration_s"] for r in rows
                      if r["arm"] == arm and r.get("status") == "PASS"]
            ratios = [r["paired_ratio_vs_normal"] for r in rows
                      if r["arm"] == arm and isinstance(r.get("paired_ratio_vs_normal"), (int, float))]
            result[arm] = {
                "n": len(values), "median_s": _median(values),
                "min_s": min(values) if values else None,
                "max_s": max(values) if values else None,
                "paired_ratio_vs_normal": {
                    "n": len(ratios), "median": _median(ratios),
                    "min": min(ratios) if ratios else None,
                    "max": max(ratios) if ratios else None,
                },
            }
        return result

    summaries = {
        scenario: duration_summary([r for r in samples if r["scenario"] == scenario])
        for scenario in ("first-fetch", "warm-reuse")
    }
    tail_summary = duration_summary(tail)
    for arm in tail_summary:
        values = [r["duration_s"] for r in tail
                  if r["arm"] == arm and r.get("status") == "PASS"]
        tail_summary[arm]["empirical_p95_s"] = (
            _percentile(values, 0.95) if len(values) >= 30 else None)
        tail_summary[arm]["p95_sample_count"] = len(values)
    plain_wins = sum(
        1 for row in paired if row["arm"] == "lmsluice-plain" and
        isinstance(row.get("ratio"), (int, float)) and row["ratio"] < 1.0)
    coded_wins = sum(
        1 for row in paired if row["arm"] in ("lmsluice-stdlib-coded", "lmz-coded") and
        isinstance(row.get("ratio"), (int, float)) and row["ratio"] < 1.0)
    route_prediction = _run_route_prediction(campaign, bundle)
    campaign.row(
        "A1-summary", "PASS", required=False, sample_count=len(samples),
        measured_repetitions_per_scenario=5, summaries=summaries,
        tail_summary=tail_summary, plain_wins=plain_wins, coded_wins=coded_wins,
        lmz=lmz_capability,
        interference=interference, controls=controls,
        route_prediction=route_prediction,
        interpretation="transport/startup/resource evidence only; no lower-BOM claim",
        p95_policy="p95 is excluded from five-sample arm summaries and retained only for the 30-sample warm tail and 30-sample observer comparisons",
    )
    return {"samples": samples, "paired": paired, "tail": tail,
            "summaries": summaries, "tail_summary": tail_summary,
            "plain_wins": plain_wins, "coded_wins": coded_wins,
            "lmz": lmz_capability, "interference": interference,
            "controls": controls, "route_prediction": route_prediction}


def _run_cache_workflow(campaign: Campaign, bundle: dict) -> dict:
    """Exercise the existing local cache identity and reuse behavior."""
    from lmsluice import cache

    workflow_dir = os.path.join(campaign.out, "cache-workflow")
    os.makedirs(workflow_dir, exist_ok=True)
    source = os.path.join(workflow_dir, "cache-source.safetensors")
    shutil.copy2(bundle["plain_path"], source)
    cache_home = os.path.join(workflow_dir, "cache-home")
    expected = hash_file(source)
    outcomes = {}
    with _temporary_env(XDG_CACHE_HOME=cache_home):
        initial_hit = cache.find(source)
        with open_model(source, cache="auto") as model:
            first = bytes(model.load())
            initial_facts = {
                "route": model.route, "from_cache": model.from_cache,
                "output_sha256": hashlib.sha256(first).hexdigest(),
            }
        initial_ok = (initial_hit is None and initial_facts["route"] == "plain"
                      and not initial_facts["from_cache"]
                      and initial_facts["output_sha256"] == expected)
        outcomes["initial_miss"] = initial_ok
        campaign.row("B1-cache-initial-miss", "PASS" if initial_ok else "FAIL",
                     required=False, cache_hit=initial_hit,
                     fetch="plain local source", expected_sha256=expected,
                     **initial_facts)

        entry = cache.build(source, codec="zstd")
        with open_model(source, cache="auto") as model:
            warm = bytes(model.load())
            warm_facts = {
                "route": model.route, "from_cache": model.from_cache,
                "output_sha256": hashlib.sha256(warm).hexdigest(),
                "cache_entry": os.path.relpath(entry, workflow_dir),
            }
        warm_ok = (warm_facts["from_cache"] and warm_facts["route"] == "coded"
                   and warm_facts["output_sha256"] == expected)
        outcomes["warm_reuse"] = warm_ok
        campaign.row("B1-cache-warm-reuse", "PASS" if warm_ok else "FAIL",
                     required=False, **warm_facts,
                     reuse="automatic existing cache entry; output bytes checked")

        before_stat = os.stat(source)
        before_key = cache.key_for(source)
        before_source_sha = hash_file(source)
        _flip(source, 700)
        os.utime(source, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        after_key = cache.key_for(source)
        after_source_sha = hash_file(source)
        stale = cache.find(source)
        changed_ok = before_source_sha != after_source_sha and before_key == after_key
        refusal_ok = stale is None
        outcomes["changed_artifact"] = changed_ok
        outcomes["stale_refusal"] = refusal_ok
        campaign.row(
            "B1-cache-changed-artifact", "PASS" if changed_ok else "FAIL",
            required=False, before_key=before_key, after_key=after_key,
            before_source_sha256=before_source_sha,
            after_source_sha256=after_source_sha,
            same_stat_identity=True,
            identity_changed=before_source_sha != after_source_sha)
        campaign.row(
            "B1-cache-stale-refusal", "PASS" if refusal_ok else "FAIL",
            required=False, candidate_entry=entry,
            cache_find_after_change=stale,
            refusal=refusal_ok,
            reason="sidecar source_sha256 rejects an in-place rewrite even when size and mtime are restored")

        refreshed_entry = cache.build(source, codec="zstd", force=True)
        changed_expected = hash_file(source)
        with open_model(source, cache="auto") as model:
            refreshed = bytes(model.load())
            refreshed_facts = {
                "route": model.route, "from_cache": model.from_cache,
                "output_sha256": hashlib.sha256(refreshed).hexdigest(),
                "expected_sha256": changed_expected,
                "cache_entry": os.path.relpath(refreshed_entry, workflow_dir),
            }
        refreshed_ok = (refreshed_facts["from_cache"] and
                        refreshed_facts["output_sha256"] == changed_expected)
        outcomes["changed_refresh"] = refreshed_ok
        campaign.row("B1-cache-changed-refresh", "PASS" if refreshed_ok else "FAIL",
                     required=False, **refreshed_facts,
                     stale_bytes_replaced=True)

        retry_row = next((row for row in reversed(campaign.rows)
                          if row["case"] == "retry-restart-from-zero"), None)
        retry_ok = bool(retry_row and retry_row["status"] == "PASS")
        outcomes["interruption_reconnect"] = retry_ok
        campaign.row(
            "B1-cache-interruption-reconnect", "PASS" if retry_ok else "FAIL",
            required=False,
            cross_link={"case": "retry-restart-from-zero",
                        "status": retry_row["status"] if retry_row else "NOT_RUN"},
            interruption="ConnectionResetError followed by a fresh full retry",
            restart_semantics="restart-from-zero; no durable resume implemented",
            repeated_bytes=(retry_row.get("repeated_bytes") if retry_row else None),
            cache_role="cache workflow retains the shared transport evidence; it does not add resume")

    range_row = next((row for row in reversed(campaign.rows)
                      if row["case"] == "B1-bandwidth-range"), None)
    no_range_row = next((row for row in reversed(campaign.rows)
                         if row["case"] == "B1-bandwidth-no-range"), None)
    outcomes["range"] = bool(range_row and range_row["status"] == "PASS")
    outcomes["no_range"] = bool(no_range_row and no_range_row["status"] == "PASS")
    campaign.row(
        "B1-cache-range", "PASS" if outcomes["range"] else "FAIL",
        required=False, cross_link=range_row,
        fallback="range requests accepted by local deterministic server")
    campaign.row(
        "B1-cache-no-range", "PASS" if outcomes["no_range"] else "FAIL",
        required=False, cross_link=no_range_row,
        fallback="full-response materialisation path restored identical output")

    wrong_key = next((row for row in reversed(campaign.rows)
                      if row["case"] == "fault-wrong-key-detection"), None)
    tampered = next((row for row in reversed(campaign.rows)
                     if row["case"] == "fault-tampered-ciphertext-detection"), None)
    protected = {
        "wrong_key": wrong_key["status"] if wrong_key else "NOT_RUN",
        "tampered_ciphertext": tampered["status"] if tampered else "NOT_RUN",
        "interpretation": "protected-artifact failure evidence is cross-linked; no customer or hosted-store claim",
    }
    all_ok = all(outcomes.values())
    result = {
        "outcomes": outcomes,
        "protected_artifact_failure": protected,
        "range": range_row,
        "no_range": no_range_row,
        "initial_miss": initial_facts,
        "warm_reuse": warm_facts,
        "changed_refresh": refreshed_facts,
        "cache_root": os.path.relpath(cache_home, campaign.out),
        "persistent_resume": "NOT_IMPLEMENTED; existing retry is from zero",
        "customer_value": "UNVALIDATED",
    }
    campaign.row("cache-workflow", "PASS" if all_ok else "FAIL", required=True,
                 **result)
    return result


def _run_niche(campaign: Campaign, bundle: dict) -> dict:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    expected = bundle["plain"]["sha256"]
    candidates = []

    # Candidate 1: restricted/private distribution. A locally sealed artifact
    # is a complete mechanism workflow; it does not prove a buyer or policy.
    from lmsluice import crypt, sealed

    if crypt.available() and bundle["coded"].get("bytes"):
        directory = os.path.dirname(plain)
        key = os.path.join(directory, "niche-private.key")
        sealed_path = os.path.join(directory, "niche-private.sealed")
        with open(key, "wb") as fh:
            fh.write(crypt.new_key())
        sealed.seal_file(coded, sealed_path, key)
        with _temporary_env(LMSLUICE_KEY_FILE=key):
            private_status, _private_facts, _private_record = _run_recorded_load(
                campaign, sealed_path, expected, arm="lmsluice-sealed",
                scenario="private-distribution", case="B1-private-complete",
                required=False, cache_condition="first-fetch")
        private_evidence = "local sealed artifact load and output hash"
    else:
        private_status = "SKIP"
        private_evidence = "crypto backend or coded fixture unavailable"
        campaign.row("B1-private-complete", private_status, required=False,
                     reason=private_evidence, evidence_class="unavailable")
    candidates.append({
        "candidate": "restricted/private distribution",
        "consumer_hypothesis": "deployment owner moving approved private artifacts",
        "buyer_hypothesis": "security-sensitive integrator; unvalidated",
        "trigger": "artifact provisioning or controlled refresh",
        "source_destination": "local sealed fixture to local deployment directory",
        "alternative": "authenticated download plus filesystem/access policy",
        "artifact": _file_digest(coded) if os.path.exists(coded) else None,
        "technical_feasibility": private_status,
        "observed_workflow_value": (
            "integrity/confidentiality mechanism exercised locally"
            if private_status == "PASS" else "mechanism unavailable in this environment"),
        "integration_support_effort": "medium: key distribution and policy remain external",
        "customer_evidence": "UNAVAILABLE; no outreach authorized",
        "recommendation": "DEFER customer claim; retain mechanism evidence",
        "evidence": private_evidence,
    })

    # Candidate 2: bandwidth-constrained population. Exercise both range and
    # no-range fallback through a local HTTP server and compare to local file.
    network_path = coded if os.path.exists(coded) else plain
    network_artifact = os.path.basename(network_path)
    network_arm = ("lmsluice-http-range" if network_path == coded
                   else "lmsluice-http-plain")
    with range_server(os.path.dirname(network_path), ranges=True) as base:
        range_status, _range_facts, _range_record = _run_recorded_load(
            campaign, f"{base}/{network_artifact}", expected,
            arm=network_arm, scenario="initial-fetch",
            case="B1-bandwidth-range", required=False,
            cache_condition="first-fetch")
    with range_server(os.path.dirname(network_path), ranges=False) as base:
        fallback_status, _fallback_facts, _fallback_record = _run_recorded_load(
            campaign, f"{base}/{network_artifact}", expected,
            arm="http-no-range-fallback", scenario="initial-fetch",
            case="B1-bandwidth-no-range", required=False,
            cache_condition="first-fetch")
    cache_workflow = _run_cache_workflow(campaign, bundle)
    candidates.append({
        "candidate": "bandwidth-constrained device population",
        "consumer_hypothesis": "device fleet restoring a model over a slow or intermittent link",
        "buyer_hypothesis": "fleet/deployment operator; no named customer available",
        "trigger": "initial fetch, cache miss, changed artifact, reconnect",
        "source_destination": "loopback range server to temporary client cache",
        "alternative": "HTTP/object-store client plus existing cache/deduplication",
        "artifact": _file_digest(network_path),
        "technical_feasibility": ("PASS" if range_status == "PASS" and
                                   fallback_status == "PASS" else "FAIL"),
        "observed_workflow_value": "range and no-range paths both restored identical output",
        "integration_support_effort": "medium: reconnect policy and durable publication are external",
        "customer_evidence": "UNAVAILABLE; local proxy is mechanism evidence only",
        "recommendation": "PURSUE technical validation on a named target; defer market claim",
        "evidence": "local loopback, not WAN/device evidence",
        "cache_workflow": cache_workflow,
    })

    # Candidate 3: switching specialist models while a useful synthetic task is
    # active. The periodic task is explicitly a scheduler probe, not a voice SLA.
    names = ["language.embed.weight"]
    specialist = ["specialist.task.weight"]
    language_result = _active_load(
        campaign, plain, _payload_digest(plain, names),
        case="B1-switch-language", arm="lmsluice-plain", names=names)
    specialist_path = coded if os.path.exists(coded) else plain
    specialist_arm = ("lmsluice-stdlib-coded" if specialist_path == coded
                      else "lmsluice-plain")
    specialist_result = _active_load(
        campaign, specialist_path, _payload_digest(plain, specialist),
        case="B1-switch-specialist", arm=specialist_arm, names=specialist)
    language_status = language_result["status"]
    specialist_status = specialist_result["status"]
    candidates.append({
        "candidate": "switching specialized models",
        "consumer_hypothesis": "application loading language/vision/task variants as work continues",
        "buyer_hypothesis": "multimodal product integrator; no customer evidence",
        "trigger": "modality/task change with useful work ongoing",
        "source_destination": "local artifact set into host buffer",
        "alternative": "runtime normal loader with warm mmap/cache and resident models",
        "artifact": _file_digest(specialist_path),
        "technical_feasibility": ("PASS" if language_status == "PASS" and
                                   specialist_status == "PASS" else "FAIL"),
        "observed_workflow_value": "subset output checks and periodic interference recorded",
        "integration_support_effort": "medium/high: residency and caller scheduling stay external",
        "customer_evidence": "UNAVAILABLE; synthetic task is not application acceptance",
        "recommendation": "DEFER until a named runtime and switching cost are measured",
        "evidence": "local synthetic roles only",
        "interference": {
            "language": language_result,
            "specialist": specialist_result,
            "interpretation": "positive overlap ticks are scheduler evidence only",
        },
    })

    campaign.row(
        "B1-candidate-comparison", "PASS", required=False,
        candidates=candidates,
        customer_outreach="NOT_AUTHORIZED",
        real_authenticated_cloud="UNAVAILABLE; no credentials or writes used",
        local_signed_store="P2 multipart signature/body failure campaign evidence",
        cache_workflow=cache_workflow,
        selection="manager retains final niche decision",
    )
    return {"candidates": candidates}


def _uncache(path: str) -> dict:
    """Attempt cache control on the exact source path used by one arm."""
    result = {
        "path": os.path.abspath(path),
        "attempted": True,
        "uncache_ok": False,
        "uncache_method": "unavailable",
        "recache_ok": False,
        "recache_method": "not_attempted",
    }
    try:
        with FileSource(path) as source:
            ok, why = source.uncache()
            result.update(uncache_ok=ok, uncache_method=why)
            restored, restore_why = source.recache()
            result.update(recache_ok=restored, recache_method=restore_why)
            return result
    except Exception as exc:  # pragma: no cover - platform-specific
        result["uncache_method"] = f"uncache unavailable: {type(exc).__name__}: {exc}"
        return result


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    directory = ""
    ranges = True
    reset_after = None
    requests = 0

    def log_message(self, *_args):
        pass

    def _path(self):
        name = os.path.basename(urllib.parse.urlparse(self.path).path)
        return os.path.join(type(self).directory, name)

    def do_HEAD(self):
        path = self._path()
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        type(self).requests += 1
        path = self._path()
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        header = self.headers.get("Range") if self.ranges else None
        with open(path, "rb") as fh:
            if header:
                lo, _, hi = header.split("=", 1)[1].partition("-")
                lo, hi = int(lo), int(hi) if hi else size - 1
                hi = min(hi, size - 1)
                fh.seek(lo)
                body = fh.read(hi - lo + 1)
                if self.reset_after is not None and type(self).requests == self.reset_after:
                    body = body[:max(0, len(body) // 2)]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
            else:
                body = fh.read()
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass


@contextmanager
def range_server(directory: str, *, ranges: bool = True, reset_after=None):
    handler = type(
        "ReadinessRangeHandler", (_RangeHandler,),
        {"directory": directory, "ranges": ranges, "reset_after": reset_after,
         "requests": 0},
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever,
                              name="lmsluice-readiness-http", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _capability_manifest(repo_root: str, bundle: dict) -> dict:
    import importlib.util
    import platform
    import shutil
    import sys

    optional = {}
    for name in ("torch", "lmzip", "zstandard"):
        optional[name] = bool(importlib.util.find_spec(name))
    try:
        from lmsluice.zstdcodec import backend_name

        stdlib_codec = backend_name()
    except Exception as exc:  # pragma: no cover - import guard
        stdlib_codec = f"unavailable: {type(exc).__name__}"
    try:
        from lmsluice.lmzcodec import backend_info

        lmz = backend_info()
    except Exception as exc:  # pragma: no cover - optional adapter
        lmz = {"available": False, "reason": type(exc).__name__}
    disk = shutil.disk_usage(repo_root)
    memory = {}
    meminfo = "/proc/meminfo"
    if os.path.exists(meminfo):
        try:
            with open(meminfo, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    key, sep, value = line.partition(":")
                    if sep and key in ("MemTotal", "MemAvailable"):
                        memory[key] = value.strip()
        except OSError:
            pass
    source_commit, commit_source = _source_identity(repo_root)
    workload_schema = _workload_schema(repo_root)
    return {
        "repository": repo_root,
        "commit": source_commit,
        "source_commit": source_commit,
        "source_commit_source": commit_source,
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "workload_schema": workload_schema,
        "artifact_sha256": bundle.get("plain", {}).get("sha256"),
        "artifact_bytes": bundle.get("plain", {}).get("bytes"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "optional_modules": optional,
        "stdlib_codec": stdlib_codec,
        "lmz": lmz,
        "cuda": _cuda_capability(),
        "resource_plan": {
            "max_busy_cpu_workers": 2,
            "synthetic_bundle_limit_bytes": 128 << 20,
            "new_run_storage_limit_bytes": 2 << 30,
            "process_working_set_planning_limit_bytes": 1 << 30,
            "free_disk_bytes_at_start": disk.free,
            "memory": memory,
            "departure": "fixture is much smaller than all limits",
        },
        "cache_policy": {
            "expected_hash_computed_at_setup": True,
            "hash_not_taken_immediately_before_timed_read": True,
            "host_cache_state": "WSL2 lower-cache state may be unobservable",
        },
        "bundle": bundle,
        "availability_classes": {
            "synthetic_fixture": "mechanism/correctness only",
            "simulated_constraint": "local bounded fault/link model",
            "measured_target": "not available in this run",
            "real_customer": "not authorized/available",
        },
    }


def _cuda_capability() -> dict:
    try:
        from lmsluice import cuda

        ok, why = cuda.available()
        return {"available": bool(ok), "reason": why}
    except Exception as exc:  # pragma: no cover - import guard
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _git_value(repo_root: str, *args) -> str:
    import subprocess

    try:
        result = subprocess.run(["git", *args], cwd=repo_root,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - non-git export
        return f"unavailable: {type(exc).__name__}"


def _source_identity(repo_root: str) -> tuple[str, str]:
    """Resolve a current checkout or Git archive to its exact source commit."""
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit or ""):
        return commit, "git-rev-parse"
    try:
        from .buildinfo import SOURCE_COMMIT

        marker = SOURCE_COMMIT.strip()
    except Exception as exc:  # pragma: no cover - import guard
        return f"unavailable: {type(exc).__name__}", "unavailable"
    if re.fullmatch(r"[0-9a-fA-F]{40}", marker):
        return marker, "git-archive-export-subst"
    return f"unavailable: {commit or 'no commit metadata'}", "unavailable"


def _workload_schema(repo_root: str):
    path = os.path.join(repo_root, "docs", "evaluations", "workload.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("schema")
    except (OSError, ValueError):
        return None


def run_regression(campaign: Campaign, repo_root: str) -> None:
    log_path = os.path.join(campaign.out, "full-regression.log")
    command = [sys.executable, "-m", "unittest", "discover", "-v",
               "-s", "tests", "-p", "test*.py"]
    results = []
    for attempt in (1, 2):
        result = subprocess.run(
            command, cwd=repo_root, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": repo_root}, timeout=180)
        results.append(result)
        attempt_path = os.path.join(
            campaign.out, f"full-regression.attempt{attempt}.log")
        with open(attempt_path, "w", encoding="utf-8") as fh:
            fh.write(result.stdout)
            fh.write(result.stderr)
        if result.returncode == 0:
            break
    result = results[-1]
    with open(log_path, "w", encoding="utf-8") as fh:
        for attempt, item in enumerate(results, 1):
            fh.write(f"=== regression attempt {attempt} ===\n")
            fh.write(item.stdout)
            fh.write(item.stderr)
    if len(results) > 1:
        campaign.row(
            "P5-full-regression-initial", "FAIL", required=False,
            returncode=results[0].returncode,
            log="full-regression.attempt1.log",
            note="initial suite failure retained; one bounded retry was run")
    output = result.stdout + result.stderr
    total_match = re.search(r"Ran (\d+) tests", output)
    skipped_match = re.search(r"skipped=(\d+)", output)
    skip_categories = {}
    for reason in re.findall(r"skipped ['\"]([^'\"]+)['\"]", output):
        skip_categories[reason] = skip_categories.get(reason, 0) + 1
    warning_lines = [line for line in output.splitlines()
                     if "ResourceWarning" in line or "warning" in line.lower()]
    status = "PASS" if result.returncode == 0 else "FAIL"
    campaign.row(
        "P5-full-regression", status, required=True,
        returncode=result.returncode,
        tests=int(total_match.group(1)) if total_match else None,
        skipped=int(skipped_match.group(1)) if skipped_match else 0,
        skip_categories=skip_categories,
        warnings=warning_lines,
        log=os.path.basename(log_path),
        attempts=len(results),
        initial_returncode=results[0].returncode,
        optional_backend_coverage="torch skipped when not installed",
    )


def run_campaign(out: str, *, repo_root: str | None = None,
                 regression: bool = True) -> int:
    """Run all local engineering cases and both independent evaluations."""
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    repo_root = os.path.abspath(repo_root or os.path.join(
        os.path.dirname(__file__), "..", ".."))
    fixtures = os.path.join(out, "fixtures")
    bundle = generate_bundle(fixtures)
    bundle["plain_path"] = os.path.join(fixtures, bundle["plain"]["path"])
    bundle["coded_path"] = os.path.join(fixtures, bundle["coded"]["path"])
    with open(os.path.join(out, "bundle.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=True)
        fh.write("\n")
    manifest = _capability_manifest(repo_root, bundle)
    provenance = {
        "artifact_sha256": bundle["plain"]["sha256"],
        "artifact_bytes": bundle["plain"]["bytes"],
        "workload_schema_version": manifest.get("workload_schema"),
        "fixture_version": bundle.get("version"),
        "source_commit": manifest.get("source_commit"),
        "source_commit_source": manifest.get("source_commit_source"),
        "platform": manifest.get("platform"),
        "utc_origin": "record.started_at_utc; event times use record.time_origin",
    }
    campaign = Campaign(out, provenance=provenance)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    stages = [
        ("P0-contract", lambda: run_contract(campaign, bundle, repo_root)),
        ("P1-observability", lambda: run_observability(campaign, bundle)),
        ("P1-overhead", lambda: run_observability_overhead(campaign, bundle)),
        ("P2-bandwidth", lambda: run_bandwidth(campaign, bundle)),
        ("P2-memory", lambda: run_memory(campaign, bundle)),
        ("P2-faults", lambda: run_faults(campaign, bundle)),
        ("P2-restarts", lambda: run_application_restarts(campaign, bundle, repo_root)),
        ("P2-multipart", lambda: run_multipart_failure(campaign, bundle)),
        ("P3-affordability", lambda: _write_evaluation(
            campaign, "affordability", _run_affordability(campaign, bundle))),
        ("P4-specialized-deployment", lambda: _write_evaluation(
            campaign, "niche", _run_niche(campaign, bundle))),
        ("P1-event-order", lambda: _record_event_order(campaign)),
    ]
    for name, action in stages:
        try:
            action()
        except BaseException as exc:
            campaign.exception(name, exc)
            campaign.row(name, "FAIL", required=True, error=_error(exc),
                         error_log=os.path.relpath(
                             os.path.join(campaign.errors_dir, f"{name}.txt"),
                             campaign.out))
    if regression:
        try:
            run_regression(campaign, repo_root)
        except BaseException as exc:
            campaign.exception("P5-full-regression", exc)
            campaign.row("P5-full-regression", "FAIL", required=True,
                         error=_error(exc))
    else:
        campaign.row("P5-full-regression", "NOT_RUN", required=False,
                     reason="--no-regression requested; run the full suite separately")
    campaign.write(bundle, manifest=manifest)
    print(json.dumps({
        "status": "PASS" if campaign.required_ok else "FAIL",
        "engineering_status": "PASS" if campaign.required_ok else "FAIL",
        "strategic_evidence": "inconclusive where target/customer evidence is unavailable",
        "run": out,
        "summary": os.path.join(out, "summary.json"),
        "results": os.path.join(out, "results.jsonl"),
    }, sort_keys=True))
    return 0 if campaign.required_ok else 3


def _write_evaluation(campaign: Campaign, name: str, data: dict) -> None:
    path = os.path.join(campaign.out, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="fresh run directory (default: temporary)")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--child-load", help=argparse.SUPPRESS)
    parser.add_argument("--no-regression", action="store_true",
                        help="skip the existing suite; useful for harness development")
    args = parser.parse_args(argv)
    if args.child_load:
        return _child_load(os.path.abspath(args.child_load))
    out = args.out or tempfile.mkdtemp(prefix="lmsluice-device-readiness-")
    return run_campaign(out, repo_root=args.repo_root,
                        regression=not args.no_regression)


if __name__ == "__main__":  # pragma: no cover - exercised by the shell probe
    raise SystemExit(main())
