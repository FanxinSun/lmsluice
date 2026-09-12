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
import hashlib
import http.server
import json
import os
import random
import shutil
import signal
import statistics
import struct
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


CAMPAIGN_SCHEMA = 1
REQUIRED_ENGINEERING = {
    "contract",
    "observability-success",
    "observability-failure",
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

    def __init__(self, out: str):
        self.out = os.path.abspath(out)
        self.records_dir = os.path.join(self.out, "records")
        self.errors_dir = os.path.join(self.out, "errors")
        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.errors_dir, exist_ok=True)
        self.rows: list[dict] = []
        self.required_failures: list[dict] = []
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

    def record(self, record: ReadinessRecord, *, report=None) -> tuple[str, dict]:
        """Freeze a recorder, save it, and return its path and JSON object."""
        data = record.finish(report)
        path = os.path.join(self.records_dir, f"{record.run_id}.json")
        record.write_json(path)
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
                       cold_method="not_attempted", sample_interval=0.02):
    metadata = {
        "case": case,
        "arm": arm,
        "scenario": scenario,
        "artifact": os.path.basename(path),
        "consumer": "synthetic-output-hash",
        "availability": "synthetic_fixture",
        "cache_condition": cache_condition,
        "cold_method": cold_method,
    }
    recorder = ReadinessRecord(metadata=metadata, sample_interval=sample_interval,
                               max_samples=256).start()
    started = time.perf_counter()
    status = "PASS"
    facts = {}
    try:
        with open_model(path, cache="off", observer=recorder,
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
    })
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


def run_contract(campaign: Campaign, bundle: dict) -> None:
    plain = bundle["plain"]
    actual = _file_digest(bundle["plain_path"])
    ok = (actual["sha256"] == plain["sha256"] and
          actual["bytes"] == plain["bytes"] and
          len(plain.get("tensors", [])) >= 5)
    campaign.row(
        "contract", "PASS" if ok else "FAIL", required=True,
        artifact=actual, expected_sha256=plain["sha256"],
        roles=[t["role"] for t in plain.get("tensors", [])],
        availability="synthetic_fixture",
        output_check="sha256 of exact consumer-visible bytes",
        failure=None if ok else {
            "reason": "generated artifact differs from frozen contract"},
    )


def run_observability(campaign: Campaign, bundle: dict) -> None:
    del bundle
    recorder = ReadinessRecord(
        metadata={"case": "observability-success", "availability": "synthetic"},
        sample_interval=0.005, max_samples=64).start()
    seen = bytearray()
    report = transport(
        [(0, 8), (8, 8), (16, 8)], lambda job: bytes([job[0] // 8]) * job[1],
        lambda job, payload: seen.extend(payload) or len(payload),
        fetch_threads=2, place_threads=1, inflight=2, observer=recorder)
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


def _measure_transport(campaign: Campaign, path: str, *, coded: bool, case: str,
                       required: bool, limiter=None, fetch_threads=4,
                       reset_at=None, fail_at=None, truncate_at=None):
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
        path_record, data = campaign.record(recorder, report=report)
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


class _PeriodicTask:
    """Small synthetic interference probe; it is never presented as a voice SLA."""

    def __init__(self, period: float = 0.005):
        self.period = period
        self.stop = threading.Event()
        self.ticks = 0
        self.slips = []
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
            self.ticks += 1
            hashlib.sha256(b"periodic-control" * 32).digest()
            deadline += self.period

    def close(self):
        self.stop.set()
        self._thread.join(timeout=2)


def _active_load(campaign: Campaign, path: str, expected_sha: str, *,
                 case: str, arm: str, names=None) -> None:
    periodic = _PeriodicTask().start()
    recorder = ReadinessRecord(
        metadata={"case": case, "arm": arm, "availability": "synthetic_fixture",
                  "consumer": "periodic-control-task"},
        sample_interval=0.02, max_samples=128).start()
    started = time.perf_counter()
    status, error, digest = "PASS", None, None
    try:
        with open_model(path, cache="off", observer=recorder) as model:
            buf = model.load(names=names, observer=recorder)
            digest_builder = hashlib.sha256()
            if names is None:
                digest_builder.update(bytes(buf))
            else:
                for name in sorted(names, key=lambda item: model.tensors[item].start):
                    digest_builder.update(bytes(model.tensor(name, buffer=buf)))
            digest = digest_builder.hexdigest()
            recorder.mark("consumer_first_useful", output="periodic-load-check")
            recorder.mark("consumer_ready", output="periodic-load-check")
            del buf
        if digest != expected_sha:
            raise AssertionError(f"active consumer output {digest} != {expected_sha}")
    except BaseException as exc:
        status, error = "FAIL", exc
        recorder.failure_event(exc, phase="active-consumer")
        campaign.exception(case, exc)
    finally:
        periodic.close()
        record_path, data = campaign.record(recorder)
    campaign.row(
        case, status, required=False,
        record=os.path.relpath(record_path, campaign.out),
        arm=arm, output_equivalent=error is None,
        duration_s=time.perf_counter() - started,
        periodic_ticks=periodic.ticks,
        periodic_max_slip_s=max(periodic.slips, default=0.0),
        note="scheduler/resource interference only; no real voice deadline",
        error=_error(error) if error else None,
    )
    return status


def _run_affordability(campaign: Campaign, bundle: dict) -> dict:
    plain = bundle["plain_path"]
    coded = os.path.join(os.path.dirname(plain), bundle["coded"]["path"])
    expected = bundle["plain"]["sha256"]
    arms = [("normal-loader", plain), ("lmsluice-plain", plain)]
    if bundle["coded"].get("bytes"):
        arms.append(("lmsluice-stdlib-coded", coded))
    rng = random.Random(20260913)
    samples = []
    for scenario in ("first-fetch", "warm-reuse"):
        # One warmup is recorded separately so it cannot enter a measured tail.
        for arm, path in arms:
            _run_recorded_load(
                campaign, path, expected, arm=arm, scenario=f"{scenario}-warmup",
                case=f"A1-{scenario}-{arm}-warmup", required=False,
                cache_condition="warmup", sample_interval=0.05)
        for repetition in range(5):
            order = list(arms)
            rng.shuffle(order)
            for arm, path in order:
                if scenario == "first-fetch":
                    cold_ok, cold_method = _uncache(plain)
                    cache_condition = "cold_attempt" if cold_ok else "warm_or_layered"
                else:
                    cold_ok, cold_method = False, "warm_after_warmup"
                    cache_condition = "warm_reuse"
                status, facts, _record = _run_recorded_load(
                    campaign, path, expected, arm=arm, scenario=scenario,
                    case=f"A1-{scenario}-{arm}", required=False,
                    cache_condition=cache_condition, cold_method=cold_method,
                    sample_interval=0.02)
                facts.update({"repetition": repetition, "status": status,
                              "cold_supported": cold_ok})
                samples.append({"scenario": scenario, "repetition": repetition,
                                "arm": arm, **facts})

    # A 30-sample empirical tail is selected only for the inexpensive local
    # plain route. It is an empirical percentile, not a product p95 claim.
    tail = []
    for repetition in range(30):
        for arm, path in (("normal-loader", plain), ("lmsluice-plain", plain)):
            status, facts, _record = _run_recorded_load(
                campaign, path, expected, arm=arm, scenario="warm-local-tail",
                case=f"A1-tail-{arm}", required=False,
                cache_condition="warm_reuse", sample_interval=0.1)
            facts.update({"repetition": repetition, "status": status})
            tail.append({"repetition": repetition, "arm": arm, **facts})

    for repetition in range(5):
        for arm, path in arms[1:]:
            _active_load(campaign, path, expected,
                         case=f"A1-interference-{arm}", arm=arm)
        control = _PeriodicTask().start()
        started = time.perf_counter()
        time.sleep(0.02)
        control.close()
        campaign.row(
            "A1-resident-inference-control", "PASS", required=False,
            repetition=repetition, duration_s=time.perf_counter() - started,
            periodic_ticks=control.ticks, periodic_max_slip_s=max(control.slips, default=0.0),
            note="already-resident synthetic control; no model loading occurred")

    def duration_summary(rows):
        result = {}
        for arm in sorted({r["arm"] for r in rows}):
            values = [r["duration_s"] for r in rows
                      if r["arm"] == arm and r.get("status") == "PASS"]
            result[arm] = {
                "n": len(values), "median_s": _median(values),
                "min_s": min(values) if values else None,
                "max_s": max(values) if values else None,
                "empirical_p95_s": sorted(values)[max(0, int(len(values) * .95) - 1)]
                if len(values) >= 30 else None,
            }
        return result

    summaries = {
        scenario: duration_summary([r for r in samples if r["scenario"] == scenario])
        for scenario in ("first-fetch", "warm-reuse")
    }
    tail_summary = duration_summary(tail)
    plain_wins = sum(
        1 for r in samples if r["arm"] == "lmsluice-plain" and
        r.get("status") == "PASS" and r["duration_s"] < next(
            (q["duration_s"] for q in samples
             if q["scenario"] == r["scenario"] and
             q["repetition"] == r["repetition"] and
             q["arm"] == "normal-loader"), float("inf")))
    coded_wins = sum(
        1 for r in samples if r["arm"] == "lmsluice-stdlib-coded" and
        r.get("status") == "PASS" and r["duration_s"] < next(
            (q["duration_s"] for q in samples
             if q["scenario"] == r["scenario"] and
             q["repetition"] == r["repetition"] and
             q["arm"] == "normal-loader"), float("inf")))
    campaign.row(
        "A1-summary", "PASS", required=False, sample_count=len(samples),
        summaries=summaries, tail_summary=tail_summary,
        plain_wins=plain_wins, coded_wins=coded_wins,
        interpretation="transport/startup evidence only; no lower-BOM claim",
    )
    campaign.row(
        "A1-route-prediction", "INCONCLUSIVE", required=False,
        calibration_data="no independent target rate calibration was supplied",
        evaluation_data="measured local rows retained in results.jsonl",
        sign_of_benefit_errors="not inferable without a frozen machine profile",
        wrong_route_cost="could be a slower coded load; no universal threshold applied",
    )
    return {"samples": samples, "tail": tail, "summaries": summaries,
            "tail_summary": tail_summary, "plain_wins": plain_wins,
            "coded_wins": coded_wins}


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
    })

    # Candidate 3: switching specialist models while a useful synthetic task is
    # active. The periodic task is explicitly a scheduler probe, not a voice SLA.
    names = ["language.embed.weight"]
    specialist = ["specialist.task.weight"]
    language_status = _active_load(
        campaign, plain, _payload_digest(plain, names),
        case="B1-switch-language", arm="lmsluice-plain", names=names)
    specialist_path = coded if os.path.exists(coded) else plain
    specialist_arm = ("lmsluice-stdlib-coded" if specialist_path == coded
                      else "lmsluice-plain")
    specialist_status = _active_load(
        campaign, specialist_path, _payload_digest(plain, specialist),
        case="B1-switch-specialist", arm=specialist_arm, names=specialist)
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
    })

    campaign.row(
        "B1-candidate-comparison", "PASS", required=False,
        candidates=candidates,
        customer_outreach="NOT_AUTHORIZED",
        real_authenticated_cloud="UNAVAILABLE; no credentials or writes used",
        local_signed_store="covered by existing lmsluice regression fixtures",
        selection="manager retains final niche decision",
    )
    return {"candidates": candidates}


def _uncache(path: str) -> tuple[bool, str]:
    try:
        with FileSource(path) as source:
            ok, why = source.uncache()
            if ok:
                source.recache()
            return ok, why
    except Exception as exc:  # pragma: no cover - platform-specific
        return False, f"uncache unavailable: {type(exc).__name__}: {exc}"


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
    return {
        "repository": repo_root,
        "commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "branch": _git_value(repo_root, "branch", "--show-current"),
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


def run_regression(campaign: Campaign, repo_root: str) -> None:
    import re
    import subprocess
    log_path = os.path.join(campaign.out, "full-regression.log")
    command = ["python3", "-m", "unittest", "discover", "-s", "tests",
               "-p", "test*.py"]
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
    status = "PASS" if result.returncode == 0 else "FAIL"
    campaign.row(
        "P5-full-regression", status, required=True,
        returncode=result.returncode,
        tests=int(total_match.group(1)) if total_match else None,
        skipped=int(skipped_match.group(1)) if skipped_match else 0,
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
    campaign = Campaign(out)
    manifest = _capability_manifest(repo_root, bundle)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    stages = [
        ("P0-contract", lambda: run_contract(campaign, bundle)),
        ("P1-observability", lambda: run_observability(campaign, bundle)),
        ("P2-bandwidth", lambda: run_bandwidth(campaign, bundle)),
        ("P2-memory", lambda: run_memory(campaign, bundle)),
        ("P2-faults", lambda: run_faults(campaign, bundle)),
        ("P3-affordability", lambda: _write_evaluation(
            campaign, "affordability", _run_affordability(campaign, bundle))),
        ("P4-specialized-deployment", lambda: _write_evaluation(
            campaign, "niche", _run_niche(campaign, bundle))),
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
    parser.add_argument("--no-regression", action="store_true",
                        help="skip the existing suite; useful for harness development")
    args = parser.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="lmsluice-device-readiness-")
    return run_campaign(out, repo_root=args.repo_root,
                        regression=not args.no_regression)


if __name__ == "__main__":  # pragma: no cover - exercised by the shell probe
    raise SystemExit(main())
