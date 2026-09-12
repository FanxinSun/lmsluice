"""Opt-in readiness and resource records for model transport.

The transport package owns movement of bytes. A caller owns the meaning of
"useful" and "ready", so the recorder deliberately exposes those as events the
caller must mark. It never turns transport completion into application
readiness and it never imports an inference framework.

The record is intentionally small and JSON-native. Event times are monotonic
nanoseconds from one request origin; the UTC value is only a human correlation
hint. Resource sampling is bounded and measures this process. It does not add
shared pages from children, device counters, or energy estimates when those
measurements are unavailable.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import platform
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass

try:  # ``resource`` is unavailable on native Windows Python.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on native Windows.
    _resource = None


SCHEMA = 1
UNMEASURED = "UNMEASURED"
EVENTS = (
    "source_open",
    "route_planned",
    "allocation",
    "staging",
    "first_payload",
    "first_tensor",
    "transfer_complete",
    "consumer_first_useful",
    "consumer_ready",
    "terminal_failure",
)

_ERROR_QUERY = re.compile(
    r"([?&](?:[A-Za-z0-9_.-]*(?:sig|token|credential|signature|key|secret)"
    r"[A-Za-z0-9_.-]*)=)[^&\s]+",
    re.IGNORECASE,
)


def safe_identity(value: object) -> str:
    """Return a source identity without query credentials or long payloads."""
    text = str(value)
    text = text.split("#", 1)[0]
    if "?" in text:
        text = text.split("?", 1)[0]
    return text[:240]


def _error_text(exc: BaseException) -> str:
    text = _ERROR_QUERY.sub(r"\1<redacted>", str(exc))
    return text[:400]


def _json_value(value: object, depth: int = 0):
    """Keep caller details bounded and serialisable."""
    if depth > 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return value[:240]
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "bytes": len(value)}
    if isinstance(value, dict):
        return {str(k)[:80]: _json_value(v, depth + 1)
                for k, v in list(value.items())[:24]}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, depth + 1) for v in value[:24]]
    if is_dataclass(value):
        return _json_value(asdict(value), depth + 1)
    return str(value)[:240]


class ReadinessRecord:
    """A machine-readable record for one transport/application request.

    The class is a passive observer. Call :meth:`start` before the operation,
    pass it as ``observer=`` to the additive loader/transport APIs, mark
    consumer events from the real caller, then call :meth:`finish`.
    """

    def __init__(self, *, run_id: str | None = None, metadata: dict | None = None,
                 sample_interval: float = 0.05, max_samples: int = 512,
                 device_probe=None, sample_resources: bool = True,
                 provenance: dict | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self._origin_ns = time.monotonic_ns()
        self._started_utc = _datetime.datetime.now(
            _datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        self._lock = threading.RLock()
        self.events = {name: None for name in EVENTS}
        self.route = {
            "planned": None,
            "actual": None,
            "fallback_reason": None,
            "codec": None,
            "cipher": None,
            "credential_mode": None,
            "source": None,
        }
        self.bytes = {
            "logical": 0,
            "coded": 0,
            "transferred": 0,
            "fetched": 0,
            "placed": 0,
            "repeated": 0,
        }
        self.coverage = {
            "requested_spans": 0,
            "covered_spans": 0,
            "requested_bytes": 0,
            "covered_bytes": 0,
            "tensor_window_bytes": 0,
            "largest_tensor_bytes": 0,
            "alignment_copy_bytes": 0,
            "retained_output_bytes": 0,
        }
        self.execution = {
            "fetch_threads": None,
            "place_threads": None,
            "inflight": None,
            "retries": 0,
            "cpu_seconds": None,
            "wall_seconds": None,
            "thread_settings": {},
            "limits": {},
            "contention": {},
        }
        self.failure = None
        self.metadata = _json_value(metadata or {})
        self.provenance = _json_value(provenance or {})
        self.platform = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        }
        self._sampler = ResourceSampler(
            self, interval=sample_interval, max_samples=max_samples,
            device_probe=device_probe, enabled=sample_resources)
        self._started = False
        self._finished = False

    @property
    def origin_ns(self) -> int:
        return self._origin_ns

    def start(self) -> "ReadinessRecord":
        """Start bounded process sampling and return this record."""
        with self._lock:
            if not self._started:
                self._started = True
                self._sampler.start()
        return self

    def mark(self, event: str, **details) -> None:
        """Record the first occurrence of an event, preserving missing events."""
        if event not in EVENTS:
            raise ValueError(f"unknown readiness event {event!r}")
        with self._lock:
            if self.events[event] is not None:
                return
            at_ns = time.monotonic_ns() - self._origin_ns
            self.events[event] = {
                "at_ns": at_ns,
                "at_s": at_ns / 1_000_000_000,
                "details": _json_value(details),
            }

    def set_route(self, *, planned=None, actual=None, fallback_reason=None,
                  codec=None, cipher=None, credential_mode=None, source=None) -> None:
        """Set additive route facts without storing credentials."""
        values = {
            "planned": planned,
            "actual": actual,
            "fallback_reason": fallback_reason,
            "codec": codec,
            "cipher": cipher,
            "credential_mode": credential_mode,
            "source": safe_identity(source) if source is not None else None,
        }
        with self._lock:
            for key, value in values.items():
                if value is not None:
                    self.route[key] = _json_value(value)

    def add_bytes(self, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                if key not in self.bytes:
                    continue
                try:
                    amount = int(value)
                except (TypeError, ValueError):
                    continue
                if amount >= 0:
                    self.bytes[key] += amount

    def set_execution(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                if key in self.execution:
                    self.execution[key] = _json_value(value)

    def set_provenance(self, **values) -> None:
        """Attach immutable campaign identity without changing the load path."""
        with self._lock:
            current = dict(self.provenance) if isinstance(self.provenance, dict) else {}
            current.update(_json_value(values))
            self.provenance = current

    def add_coverage(self, **values: int) -> None:
        with self._lock:
            for key, value in values.items():
                if key not in self.coverage:
                    continue
                try:
                    amount = int(value)
                except (TypeError, ValueError):
                    continue
                if amount >= 0:
                    self.coverage[key] += amount

    def failure_event(self, exc: BaseException, *, phase: str = "") -> None:
        """Record a terminal failure without leaking credential-bearing text."""
        self.failure = {
            "type": type(exc).__name__,
            "message": _error_text(exc),
            "phase": phase[:80],
        }
        self.mark("terminal_failure", type=type(exc).__name__, phase=phase)

    def attach_report(self, report) -> None:
        """Attach the existing transport ``Report`` as an additive summary."""
        with self._lock:
            self._transport = (report.to_dict() if hasattr(report, "to_dict")
                               else _json_value(report))
            # ``transport`` may attach once at completion and a caller may
            # attach the same report again while freezing the record. These
            # are absolute counters, so assignment/max avoids double counting.
            for key, value in (
                    ("fetched", getattr(report, "fetched_bytes", 0)),
                    ("placed", getattr(report, "placed_bytes", 0)),
                    ("transferred", getattr(report, "fetched_bytes", 0))):
                try:
                    self.bytes[key] = max(self.bytes[key], int(value))
                except (KeyError, TypeError, ValueError):
                    pass
        self.set_execution(
            fetch_threads=getattr(report, "fetch_threads", None),
            place_threads=getattr(report, "place_threads", None),
            inflight=getattr(report, "inflight", None),
            wall_seconds=getattr(report, "seconds", None),
            cpu_seconds=(getattr(report, "fetch_seconds", 0.0) or 0.0)
            + (getattr(report, "place_seconds", 0.0) or 0.0),
        )

    def finish(self, report=None) -> dict:
        """Stop sampling, attach an optional transport report and freeze output."""
        if report is not None:
            self.attach_report(report)
        with self._lock:
            if self._started and not self._finished:
                self._sampler.stop()
            self._finished = True
        return self.to_dict()

    def to_dict(self) -> dict:
        with self._lock:
            out = {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "started_at_utc": self._started_utc,
                "time_origin": {
                    "clock": "time.monotonic_ns",
                    "origin_ns": self._origin_ns,
                    "units": "ns",
                },
                "events": _json_value(self.events),
                "route": _json_value(self.route),
                "bytes": _json_value(self.bytes),
                "coverage": _json_value(self.coverage),
                "execution": _json_value(self.execution),
                "platform": _json_value(self.platform),
                "resources": self._sampler.to_dict(),
                "failure": _json_value(self.failure),
                "metadata": _json_value(self.metadata),
                "provenance": _json_value(self.provenance),
            }
            if hasattr(self, "_transport"):
                out["transport"] = _json_value(self._transport)
            return out

    def write_json(self, path: str) -> str:
        """Write one record atomically, creating only its parent directory."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        temp = f"{path}.{os.getpid()}.tmp"
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temp, path)
        return path

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, _tb):
        if exc is not None:
            self.failure_event(exc, phase="context")
        self.finish()
        return False


class ResourceSampler:
    """Bounded current-process RSS/PSS sampler used by ``ReadinessRecord``."""

    def __init__(self, record: ReadinessRecord, *, interval: float,
                 max_samples: int, device_probe=None, enabled: bool = True):
        self.record = record
        self.interval = max(0.001, float(interval))
        self.max_samples = max(1, int(max_samples))
        self.device_probe = device_probe
        self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._samples = 0
        self._baseline = None
        self._peak = None
        self._last = None
        self._device = {"status": UNMEASURED, "reason": "no probe supplied"}

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._sample()
            self._thread = threading.Thread(
                target=self._run, name="lmsluice-resource-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, self.interval * 4))
        with self._lock:
            self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._samples >= self.max_samples:
                    return
                self._sample()

    def _sample(self) -> None:
        if self._samples >= self.max_samples:
            return
        current = _memory_sample()
        self._samples += 1
        self._last = current
        if self._baseline is None:
            self._baseline = current
        if self._peak is None:
            self._peak = dict(current)
        for key, value in current.items():
            if isinstance(value, int) and isinstance(self._peak.get(key), int):
                self._peak[key] = max(self._peak[key], value)
        if self.device_probe is not None:
            try:
                value = self.device_probe()
                self._device = {"status": "MEASURED", **_json_value(value)} \
                    if isinstance(value, dict) else {
                        "status": "MEASURED", "value": _json_value(value)}
            except Exception as exc:  # noqa: BLE001 - observation must not fail work
                self._device = {"status": UNMEASURED,
                                "reason": type(exc).__name__}

    def to_dict(self) -> dict:
        with self._lock:
            if not self.enabled:
                return {
                    "sampling": {
                        "status": "DISABLED",
                        "interval_s": self.interval,
                        "max_samples": self.max_samples,
                        "samples": 0,
                        "scope": "current_process",
                        "child_accounting": "excluded; shared pages not summed",
                        "coverage": "disabled_by_caller",
                    },
                    "host": {"baseline": {}, "peak": {}, "last": {}},
                    "allocator": {
                        "status": UNMEASURED,
                        "reason": "no allocator-specific hook supplied",
                    },
                    "device": {"status": UNMEASURED, "reason": "sampling disabled"},
                    "energy": {
                        "status": UNMEASURED,
                        "reason": "no reliable energy sensor supplied",
                    },
                }
            baseline = self._baseline or {}
            peak = self._peak or {}
            return {
                "sampling": {
                    "interval_s": self.interval,
                    "max_samples": self.max_samples,
                    "samples": self._samples,
                    "scope": "current_process",
                    "child_accounting": "excluded; shared pages not summed",
                    "coverage": "best_effort",
                },
                "host": {
                    "baseline": _json_value(baseline),
                    "peak": _json_value(peak),
                    "last": _json_value(self._last or {}),
                },
                "allocator": {
                    "status": UNMEASURED,
                    "reason": "no allocator-specific hook supplied",
                },
                "device": _json_value(self._device),
                "energy": {
                    "status": UNMEASURED,
                    "reason": "no reliable energy sensor supplied",
                },
            }


def _memory_sample() -> dict:
    out = {}
    status_path = "/proc/self/status"
    if os.path.exists(status_path):
        try:
            with open(status_path, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    key, sep, value = line.partition(":")
                    if sep and key in ("VmRSS", "VmHWM"):
                        parts = value.strip().split()
                        if parts and parts[0].isdigit():
                            multiplier = 1024 if len(parts) == 1 or parts[1] == "kB" else 1
                            out[key.lower() + "_bytes"] = int(parts[0]) * multiplier
        except OSError:
            pass
    pss_path = "/proc/self/smaps_rollup"
    if os.path.exists(pss_path):
        try:
            with open(pss_path, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    key, sep, value = line.partition(":")
                    if sep and key == "Pss":
                        parts = value.strip().split()
                        if parts and parts[0].isdigit():
                            out["pss_bytes"] = int(parts[0]) * 1024
                        break
        except OSError:
            pass
    if _resource is not None:
        try:
            hwm = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
            # Linux reports KiB, macOS reports bytes.
            out.setdefault("ru_maxrss_bytes", hwm * (1024 if sys.platform != "darwin" else 1))
        except (AttributeError, OSError, ValueError):
            pass
    return out


__all__ = ["EVENTS", "ReadinessRecord", "ResourceSampler", "SCHEMA",
           "UNMEASURED", "safe_identity"]
