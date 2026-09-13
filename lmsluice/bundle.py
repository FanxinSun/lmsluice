"""Verified, format-neutral model bundle delivery.

This module owns the part between an artifact provider and a consumer.  A
bundle is a small manifest plus a complete set of immutable files; the
manifest is deliberately independent of any archive codec or inference
framework.  The plain provider uses :mod:`lmsluice.source` and
:mod:`lmsluice.transport` to copy files through an invocation-owned staging
tree.  Optional providers live behind separate lazy adapters and never enter
the core import path.

The plain source format is a directory containing ``bundle.json`` (or
``manifest.json``) and the relative files named by its ``entries`` list.  A
manifest may also be supplied as a JSON path.  A wrapper with a top-level
``bundle`` object is accepted because that is the extension point used by the
accepted lmz producer; the object itself is the canonical identity payload.

Hashes establish equality with an expected value.  They do not establish who
published the bytes, so the inventory always reports authentication as false
unless an independent authenticated mechanism is supplied by a provider.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass, field

from .source import Source
from .transport import Report, transport


BUNDLE_SCHEMA_MAJOR = 1
BUNDLE_SCHEMA_MINOR = 0
SUPPORTED_ROLES = frozenset({
    "graph", "weights", "config", "preprocess", "vocabulary",
    "calibration", "opaque",
})
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
DRIVE_PATH = re.compile(r"^[A-Za-z]:")
COPY_CHUNK = 1 << 20
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def _entry_id(path: str) -> str:
    return "entry-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:32]


class BundleError(RuntimeError):
    """A structured, fail-closed bundle error."""

    def __init__(self, message: str, *, code: str = "bundle_error", **details):
        super().__init__(message)
        self.code = str(code)
        self.details = {str(k): _json_value(v) for k, v in details.items()}

    def as_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


class BundleSourceChanged(BundleError):
    """The source generation changed during resolution or materialization."""

    def __init__(self, message: str = "bundle source generation changed", **details):
        super().__init__(message, code="source_changed", **details)


class BundleCancelled(BundleError):
    """Cooperative cancellation at a supported bundle boundary."""

    def __init__(self, message: str = "bundle operation cancelled", **details):
        super().__init__(message, code="cancelled", **details)


@dataclass(frozen=True)
class BundleDependency:
    """A dependency edge and, when declared, the range consumed from it."""

    path: str
    offset: int | None = None
    length: int | None = None
    entry_id: str | None = None

    def to_dict(self) -> dict:
        out = {"path": self.path}
        if self.entry_id is not None:
            out["id"] = self.entry_id
        if self.offset is not None:
            out["offset"] = self.offset
        if self.length is not None:
            out["length"] = self.length
        return out


@dataclass(frozen=True)
class BundleEntry:
    """One complete, content-addressed bundle file."""

    path: str
    role: str
    length: int
    sha256: str
    dependencies: tuple[BundleDependency, ...] = ()
    entry_id: str | None = None
    representation: object = "plain"
    stored_length: int | None = None
    consumer: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "path": self.path,
            "role": self.role,
            "length": self.length,
            "sha256": self.sha256,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "representation": _json_value(self.representation),
        }
        if self.entry_id is not None:
            out["id"] = self.entry_id
        if self.stored_length is not None:
            out["stored_length"] = self.stored_length
        if self.consumer:
            out["consumer"] = _json_value(self.consumer)
        if self.metadata:
            out.update(_json_value(self.metadata))
        return out


@dataclass(frozen=True)
class BundleDescriptor:
    """Resolved manifest and immutable source generation."""

    schema: str
    manifest: dict
    entries: tuple[BundleEntry, ...]
    identity: str
    source_root: str | None = None
    manifest_path: str | None = None
    source_generation: dict | None = None
    entry_point: str | None = None
    entry_point_details: dict = field(default_factory=dict)
    consumer: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(entry.length for entry in self.entries)

    @property
    def graph(self) -> BundleEntry | None:
        if self.entry_point:
            for entry in self.entries:
                if entry.path == self.entry_point:
                    return entry
        return next((entry for entry in self.entries if entry.role == "graph"), None)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "manifest_sha256": self.identity,
            "entry_point": self.entry_point,
            "entry_point_details": _json_value(self.entry_point_details),
            "consumer": _json_value(self.consumer),
            "resources": _json_value(self.resources),
            "entries": [entry.to_dict() for entry in self.entries],
            "source_root": self.source_root,
            "manifest_path": self.manifest_path,
            "source_generation": _json_value(self.source_generation),
        }


@dataclass(frozen=True)
class BundleRequest:
    """Immutable caller request and transport/materialization ceilings."""

    expected_manifest_sha256: str | None = None
    expected_bundle_sha256: str | None = None
    source_generation: dict | None = None
    consumer: str | None = None
    entry_point: str | None = None
    max_bytes: int | None = None
    max_staging_bytes: int | None = None
    max_transferred_bytes: int | None = None
    cancellation: object | None = None
    route: str = "plain"

    @classmethod
    def from_value(cls, value=None, **overrides) -> "BundleRequest":
        if value is None:
            data = {}
        elif isinstance(value, cls):
            data = {name: getattr(value, name) for name in cls.__dataclass_fields__}
        elif isinstance(value, dict):
            data = dict(value)
        else:
            raise TypeError("request must be BundleRequest, dict or None")
        aliases = {
            "manifest_sha256": "expected_manifest_sha256",
            "bundle_sha256": "expected_bundle_sha256",
            "max_materialized_bytes": "max_bytes",
            "max_transfer_bytes": "max_transferred_bytes",
        }
        for key, target in aliases.items():
            if key in data and target not in data:
                data[target] = data.pop(key)
        data.update(overrides)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise TypeError(f"unknown bundle request field(s): {', '.join(unknown)}")
        return cls(**data)

    def limits(self) -> dict:
        return {
            key: value for key, value in (
                ("max_bytes", self.max_bytes),
                ("max_staging_bytes", self.max_staging_bytes),
                ("max_transferred_bytes", self.max_transferred_bytes),
            ) if value is not None
        }


@dataclass(frozen=True)
class BundleResult:
    """Verified result of a successful plain materialization."""

    destination: str
    manifest_sha256: str
    bundle_sha256: str
    source_generation: dict
    entries: tuple[dict, ...]
    materialized_bytes: int
    transferred_bytes: int | None
    route: str = "plain"
    authenticated: bool = False
    entry_point: str | None = None
    entry_point_details: dict = field(default_factory=dict)
    consumer: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "destination": self.destination,
            "manifest_sha256": self.manifest_sha256,
            "bundle_sha256": self.bundle_sha256,
            "source_generation": _json_value(self.source_generation),
            "entries": _json_value(self.entries),
            "materialized_bytes": self.materialized_bytes,
            "transferred_bytes": self.transferred_bytes,
            "route": self.route,
            "authenticated": self.authenticated,
            "publisher_authenticated": self.authenticated,
            "entry_point": self.entry_point,
            "entry_point_details": _json_value(self.entry_point_details),
            "consumer": _json_value(self.consumer),
        }

    def __fspath__(self):
        return self.destination


class BundleProvider:
    """Provider protocol implemented by plain and optional coded routes."""

    name = "provider"

    def capability(self) -> dict:
        return {"provider": self.name, "available": True}

    def validate(self, source, **kwargs):  # pragma: no cover - protocol guard
        raise NotImplementedError

    def inventory(self, source, **kwargs):  # pragma: no cover - protocol guard
        raise NotImplementedError

    def materialize(self, source, destination, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _FdSource(Source):
    """A Source over an already verified descriptor, used by plain transport."""

    def __init__(self, fd: int, name: str, size: int):
        self._fd = fd
        self.name = name
        self.size = int(size)
        self.random_access = True

    def pread(self, offset: int, length: int) -> bytes:
        if hasattr(os, "pread"):
            data = os.pread(self._fd, length, offset)
        else:  # pragma: no cover - native Windows fallback
            os.lseek(self._fd, offset, os.SEEK_SET)
            data = os.read(self._fd, length)
        if len(data) != length:
            raise EOFError(f"{self.name}: wanted {length} at {offset}, got {len(data)}")
        return data

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class PlainBundleProvider(BundleProvider):
    """Directory/manifest provider with verified atomic no-replace publish."""

    name = "plain"

    def validate(self, source, *, request=None, observer=None, **kwargs) -> dict:
        req = BundleRequest.from_value(request, **kwargs)
        descriptor = _resolve_descriptor(source, req, observer=observer)
        _check_limits(descriptor, req)
        content = _verify_source_files(descriptor, req, observer=observer)
        _mark(observer, "bundle_verified", manifest_sha256=descriptor.identity,
              entries=len(descriptor.entries), materialized_bytes=descriptor.total_bytes)
        result = _validation_dict(descriptor, content, req)
        return result

    def inventory(self, source, *, request=None, strict=True, observer=None,
                  **kwargs) -> dict:
        req = BundleRequest.from_value(request, **kwargs)
        descriptor = _resolve_descriptor(source, req, observer=observer)
        _check_limits(descriptor, req)
        if strict:
            content = _verify_source_files(descriptor, req, observer=observer)
        else:
            content = {entry.path: {"length": entry.length, "sha256": entry.sha256,
                                    "verified": False} for entry in descriptor.entries}
        return _inventory_dict(descriptor, content, req, strict=strict)

    def materialize(self, source, destination, *, request=None, observer=None,
                    fetch_threads=1, place_threads=1, inflight=2,
                    chunk_bytes=COPY_CHUNK, **kwargs) -> BundleResult:
        req = BundleRequest.from_value(request, **kwargs)
        started = time.perf_counter()
        parent = None
        parent_fd = -1
        parent_token = None
        stage_name = None
        stage_fd = -1
        stage_token = None
        published = False
        cleanup_state = "removed"
        aggregate = Report(fetch_threads=fetch_threads, place_threads=place_threads,
                           inflight=max(1, int(inflight)))
        try:
            descriptor = _resolve_descriptor(source, req, observer=observer)
            _check_limits(descriptor, req)
            _check_cancel(req.cancellation, "resolve")
            # Verification is completed before any ready/materialization event.
            content = _verify_source_files(descriptor, req, observer=observer)
            _mark(observer, "bundle_verified", manifest_sha256=descriptor.identity,
                  entries=len(descriptor.entries), materialized_bytes=descriptor.total_bytes)
            _check_cancel(req.cancellation, "before_materialization")
            destination_abs, parent, name, parent_fd, parent_token = \
                _prepare_destination_anchor(destination)
            _check_parent_anchor(parent, parent_fd, parent_token, phase="before_staging")
            stage_name, stage_fd, stage_token = _create_owned_stage(parent_fd)
            stage = os.path.join(parent, stage_name)
            _mark(observer, "allocation", bytes=descriptor.total_bytes,
                  destination="staging", ceiling=req.limits())
            _mark(observer, "staging", bytes=descriptor.total_bytes,
                  destination_ownership="invocation-owned", staging=stage)
            _set_execution(
                observer,
                resource_bytes={
                    "declared_materialized": descriptor.total_bytes,
                    "staging_reserved": descriptor.total_bytes,
                },
                resource_peak_bytes={
                    "staging_reserved": descriptor.total_bytes,
                    "transport_inflight_bound": max(1, int(inflight)) *
                    max(1, int(chunk_bytes)),
                },
                measurement_method={
                    "staging_reserved": "manifest_declared_bytes",
                    "transport_inflight_bound": "inflight_times_chunk_ceiling",
                },
            )
            transferred = 0
            materialized = []
            first_payload = False
            for entry in descriptor.entries:
                _check_cancel(req.cancellation, f"before_fetch:{entry.path}")
                _mark(observer, "fetch_started", path=entry.path,
                      route=req.route)
                src_path = _safe_entry_path(descriptor.source_root, entry.path)
                out_fd = None
                src_fd = None
                source_reader = None
                try:
                    src_fd, before = _open_verified_entry(
                        descriptor.source_root, src_path, entry,
                        descriptor.source_generation)
                    out_fd = _open_relative_fd(stage_fd, entry.path, write=True)
                    source_reader = _FdSource(src_fd, src_path, entry.length)
                    src_fd = None
                    jobs = [(offset, min(max(1, int(chunk_bytes)),
                                         entry.length - offset))
                            for offset in range(0, entry.length,
                                                max(1, int(chunk_bytes)))]

                    def fetch(job):
                        _check_cancel(req.cancellation, f"among_fetches:{entry.path}")
                        return source_reader.pread(job[0], job[1])

                    def place(job, payload):
                        _check_cancel(req.cancellation, f"among_places:{entry.path}")
                        _write_at(out_fd, payload, job[0])
                        return len(payload)

                    if jobs:
                        report = transport(
                            jobs, fetch, place,
                            fetch_threads=max(1, min(int(fetch_threads), len(jobs))),
                            place_threads=max(1, min(int(place_threads), len(jobs))),
                            inflight=max(1, int(inflight)),
                        )
                        _add_report(aggregate, report)
                        transferred += report.fetched_bytes
                        if not first_payload and report.fetched_bytes:
                            first_payload = True
                            _mark(observer, "first_payload", bytes=report.fetched_bytes,
                                  path=entry.path, route=req.route)
                        _add_observer_bytes(observer, transferred=report.fetched_bytes,
                                             fetched=report.fetched_bytes,
                                             decoded=report.placed_bytes)
                        if transferred > _limit(req.max_transferred_bytes):
                            raise BundleError("transport ceiling exceeded",
                                              code="resource_limit",
                                              limit=req.max_transferred_bytes,
                                              actual=transferred)
                    _check_cancel(req.cancellation, f"after_fetch:{entry.path}")
                    os.fsync(out_fd)
                    actual = _hash_fd(out_fd)
                    if actual["length"] != entry.length:
                        raise BundleError("materialized entry length mismatch",
                                          code="length_mismatch", path=entry.path,
                                          expected=entry.length, actual=actual["length"])
                    if actual["sha256"] != entry.sha256:
                        raise BundleError("materialized entry digest mismatch",
                                          code="digest_mismatch", path=entry.path,
                                          expected=entry.sha256, actual=actual["sha256"])
                    _check_generation(src_path, descriptor.source_generation, entry.path)
                    materialized.append({
                        "path": entry.path,
                        "id": entry.entry_id,
                        "role": entry.role,
                        "length": actual["length"],
                        "sha256": actual["sha256"],
                        "verified": True,
                    })
                finally:
                    if out_fd is not None:
                        try:
                            os.close(out_fd)
                        except OSError:
                            pass
                    if src_fd is not None:
                        try:
                            os.close(src_fd)
                        except OSError:
                            pass
                    if source_reader is not None:
                        source_reader.close()
                        source_reader = None
            _check_generation_root(descriptor)
            _check_cancel(req.cancellation, "before_publish")
            _check_parent_anchor(parent, parent_fd, parent_token, phase="before_publish")
            _mark(observer, "transfer_complete", bytes=transferred, route="plain")
            _mark(observer, "reconstruction_complete", entries=len(materialized),
                  bytes=descriptor.total_bytes)
            _atomic_publish_fd_no_replace(parent_fd, stage_name, name, destination_abs)
            published = True
            stage_name = None
            _add_observer_bytes(observer, materialized=descriptor.total_bytes,
                                 placed=descriptor.total_bytes)
            _mark(observer, "materialization_complete", destination=destination_abs,
                  bytes=descriptor.total_bytes, owner="caller-destination")
            _attach_report(observer, aggregate)
            bundle_digest = _bundle_content_digest(descriptor.entries)
            return BundleResult(
                destination=destination_abs,
                manifest_sha256=descriptor.identity,
                bundle_sha256=bundle_digest,
                source_generation=descriptor.source_generation or {},
                entries=tuple(materialized),
                materialized_bytes=descriptor.total_bytes,
                transferred_bytes=transferred,
                entry_point=descriptor.entry_point,
                entry_point_details=descriptor.entry_point_details,
                consumer=descriptor.consumer,
            )
        except BaseException as exc:
            if isinstance(exc, BundleCancelled):
                error = exc
            elif isinstance(exc, BundleError):
                error = exc
            else:
                error = BundleError(str(exc) or type(exc).__name__,
                                    code="materialization_failed",
                                    exception=type(exc).__name__)
            if isinstance(error, BundleCancelled):
                _mark(observer, "cancelled", boundary=error.details.get("boundary"))
                _set_execution(observer,
                               cancellation_boundary=error.details.get("boundary"))
            _failure(observer, error, phase="materialize")
            _mark(observer, "release", owner="invocation", cleanup="staging_removed")
            raise error from exc
        finally:
            if stage_fd >= 0 and not published:
                cleanup_state = _remove_owned_stage(parent_fd, stage_name, stage_fd,
                                                     stage_token)
            if stage_fd >= 0:
                os.close(stage_fd)
                stage_fd = -1
            if parent_fd >= 0:
                os.close(parent_fd)
                parent_fd = -1
            _set_execution(observer, cleanup={
                "owned_staging": "published" if published else cleanup_state,
                "staging_exists_after": cleanup_state not in ("removed", "published", "none"),
            })
            _set_execution(observer, wall_seconds=time.perf_counter() - started,
                           limits=req.limits(), destination_ownership=(
                               "published" if published else "staging_removed"))


class LmzBundleProvider(BundleProvider):
    """Read-only adapter for the accepted public lmz bundle API.

    The adapter owns the translation between the lmsluice envelope and lmz's
    direct schema-1 payload.  It calls only lmz's public bundle functions and
    never imports archive internals.  Source files and caller destinations are
    kept outside the sibling operation: lmz receives an owned seven-file
    snapshot and materialization is copied from an owned private result into
    lmsluice's stable destination staging tree.
    """

    name = "lmz"

    def _module(self):
        try:
            module = importlib.import_module("lmz")
        except Exception as exc:
            raise BundleError("lmz bundle provider is unavailable",
                              code="provider_unavailable",
                              provider="lmz", reason=f"{type(exc).__name__}: {exc}") from exc
        return module

    def _function(self, name):
        module = self._module()
        fn = getattr(module, name, None)
        if fn is None:
            try:
                fn = getattr(importlib.import_module("lmz.bundle"), name)
            except Exception as exc:
                raise BundleError(f"lmz does not publish {name}",
                                  code="provider_incompatible", provider="lmz",
                                  function=name) from exc
        return fn

    def capability(self) -> dict:
        try:
            module = self._module()
            functions = [name for name in (
                "create_bundle", "validate_bundle", "inventory_bundle",
                "materialize_bundle") if getattr(module, name, None) is not None]
            return {"provider": self.name, "available": len(functions) == 4,
                    "functions": functions,
                    "reason": None if len(functions) == 4 else "public bundle API incomplete"}
        except BundleError as exc:
            return {"provider": self.name, "available": False,
                    "functions": [], "reason": exc.details.get("reason", str(exc))}

    def create(self, source_root, archive, *, manifest=None, **kwargs):
        observer = kwargs.pop("observer", None)
        request_value = kwargs.pop("request", None)
        request_fields = {}
        for field_name in BundleRequest.__dataclass_fields__:
            value = kwargs.pop(field_name, None)
            if field_name != "route" and value is not None:
                request_fields[field_name] = value
        req = BundleRequest.from_value(request_value, route="lmz", **request_fields)
        started = time.perf_counter()
        _mark(observer, "bundle_requested", route="lmz",
              expected_manifest_sha256=req.expected_manifest_sha256,
              limits=req.limits(), operation="create")
        try:
            supplied = manifest
            if supplied is None:
                supplied, root, _manifest_path = _load_manifest_source(source_root)
            else:
                if isinstance(source_root, dict):
                    root_value = source_root.get("source_root") or source_root.get("root")
                    root = _safe_directory(os.path.abspath(os.fspath(root_value)),
                                           "source root")
                else:
                    root = _safe_directory(os.path.abspath(os.fspath(source_root)),
                                           "source root")
            _mark(observer, "source_open", source=root,
                  source_kind="plain_source_snapshot", provider="lmz")
            _set_route(observer, planned="lmz", actual="lmz", source=root,
                       codec="lmz", provider="lmz")
            descriptor = _resolve_descriptor(
                {"source_root": root, "manifest": supplied}, req)
            content = _verify_source_files(descriptor, req)
            payload = _lmz_payload(supplied, descriptor=descriptor)
            with tempfile.TemporaryDirectory(prefix=".lmsluice-lmz-source-") as snapshot:
                _copy_descriptor_entries(descriptor, snapshot, content)
                fn = self._function("create_bundle")
                args = dict(kwargs)
                args["manifest"] = payload
                _check_cancel(req.cancellation, "before_lmz_call")
                raw = fn(snapshot, archive, **args)
            result = _normalize_lmz_result(raw, operation="create")
            _check_cancel(req.cancellation, "after_lmz_call")
            _check_normalized_limits(result, req)
            result["input_manifest_sha256"] = descriptor.identity
            result["source_generation"] = descriptor.source_generation
            _lmz_observe_result(observer, result, phase="create")
            return result
        except BaseException as exc:
            error = _as_lmz_error(exc, phase="create")
            if isinstance(error, BundleCancelled):
                _mark(observer, "cancelled", boundary=error.details.get("boundary"))
                _set_execution(observer,
                               cancellation_boundary=error.details.get("boundary"))
            _failure(observer, error, phase="lmz_create")
            raise error from exc
        finally:
            _set_execution(observer, wall_seconds=time.perf_counter() - started,
                           destination_ownership="provider-owned")
            _mark(observer, "release", owner="lmz-provider", cleanup="complete")

    def validate(self, source, *, request=None, observer=None, **kwargs):
        req = _lmz_request(request, kwargs)
        started = time.perf_counter()
        try:
            _lmz_mark_start(observer, source, req, "validate")
            fn = self._function("validate_bundle")
            _check_cancel(req.cancellation, "before_lmz_call")
            raw = fn(source, **_lmz_digest_kwargs(req))
            result = _normalize_lmz_result(raw, operation="validate")
            _check_cancel(req.cancellation, "after_lmz_call")
            _check_normalized_limits(result, req)
            _lmz_observe_result(observer, result, phase="validate")
            return result
        except BaseException as exc:
            error = _as_lmz_error(exc, phase="validate")
            if isinstance(error, BundleCancelled):
                _mark(observer, "cancelled", boundary=error.details.get("boundary"))
                _set_execution(observer,
                               cancellation_boundary=error.details.get("boundary"))
            _failure(observer, error, phase="lmz_validate")
            raise error from exc
        finally:
            _set_execution(observer, wall_seconds=time.perf_counter() - started)
            _mark(observer, "release", owner="lmz-provider", cleanup="complete")

    def inventory(self, source, *, request=None, strict=True, observer=None, **kwargs):
        req = _lmz_request(request, kwargs)
        started = time.perf_counter()
        try:
            _lmz_mark_start(observer, source, req, "inventory")
            fn = self._function("inventory_bundle")
            _check_cancel(req.cancellation, "before_lmz_call")
            raw = fn(source, strict=strict, **_lmz_digest_kwargs(req))
            result = _normalize_lmz_result(raw, operation="inventory", strict=strict)
            _check_cancel(req.cancellation, "after_lmz_call")
            _check_normalized_limits(result, req)
            _lmz_observe_result(observer, result, phase="inventory")
            return result
        except BaseException as exc:
            error = _as_lmz_error(exc, phase="inventory")
            if isinstance(error, BundleCancelled):
                _mark(observer, "cancelled", boundary=error.details.get("boundary"))
                _set_execution(observer,
                               cancellation_boundary=error.details.get("boundary"))
            _failure(observer, error, phase="lmz_inventory")
            raise error from exc
        finally:
            _set_execution(observer, wall_seconds=time.perf_counter() - started)
            _mark(observer, "release", owner="lmz-provider", cleanup="complete")

    def materialize(self, source, destination, *, request=None, observer=None, **kwargs):
        req = _lmz_request(request, kwargs)
        started = time.perf_counter()
        parent_fd = -1
        stage_fd = -1
        stage_name = None
        stage_token = None
        parent = None
        parent_token = None
        published = False
        cleanup_state = "removed"
        private_root = None
        try:
            _lmz_mark_start(observer, source, req, "materialize")
            inventory_fn = self._function("inventory_bundle")
            _check_cancel(req.cancellation, "before_inventory_call")
            inventory_raw = inventory_fn(source, strict=True,
                                         **_lmz_digest_kwargs(req))
            inventory = _normalize_lmz_result(inventory_raw, operation="inventory",
                                              strict=True)
            _check_cancel(req.cancellation, "after_inventory_call")
            _lmz_observe_result(observer, inventory, phase="inventory")
            if not inventory.get("valid", True):
                raise BundleError("lmz inventory is not verified", code="provider_failure",
                                  provider="lmz", status=inventory.get("status"))
            _check_normalized_limits(inventory, req)
            _check_cancel(req.cancellation, "before_materialize_call")
            destination_abs, parent, name, parent_fd, parent_token = \
                _prepare_destination_anchor(destination)
            _check_parent_anchor(parent, parent_fd, parent_token, phase="before_lmz_call")
            private_root = tempfile.mkdtemp(prefix=".lmsluice-lmz-operation-")
            provider_destination = os.path.join(private_root, "materialized")
            fn = self._function("materialize_bundle")
            raw = fn(source, provider_destination, **_lmz_digest_kwargs(req))
            result = _normalize_lmz_result(raw, operation="materialize")
            _lmz_observe_result(observer, result, phase="lmz_call")
            _check_cancel(req.cancellation, "after_materialize_call")
            if result.get("status") not in ("complete", "verified"):
                raise BundleError("lmz materialization did not complete",
                                  code="provider_failure", provider="lmz",
                                  status=result.get("status"))
            _check_materialization_matches_inventory(inventory, result)
            _check_normalized_limits(result, req)
            _check_parent_anchor(parent, parent_fd, parent_token, phase="before_staging")
            stage_name, stage_fd, stage_token = _create_owned_stage(parent_fd)
            stage = os.path.join(parent, stage_name)
            declared_bytes = sum(entry["length"] for entry in inventory["entries"])
            _mark(observer, "allocation", bytes=declared_bytes,
                  destination="staging", ceiling=req.limits(), route="lmz")
            _mark(observer, "staging", bytes=declared_bytes,
                  destination_ownership="invocation-owned", staging=stage,
                  route="lmz")
            accounting = result.get("archive_accounting") or \
                inventory.get("archive_accounting") or {}
            _set_execution(
                observer,
                resource_bytes={
                    "declared_materialized": declared_bytes,
                    "staging_reserved": declared_bytes,
                    "provider_archive_bytes": accounting.get("archive_bytes"),
                    "provider_unique_payload_bytes": accounting.get("unique_payload_bytes"),
                    "provider_decoded_bytes": accounting.get("decoded_bytes"),
                },
                resource_peak_bytes={"staging_reserved": declared_bytes},
                measurement_method={
                    "staging_reserved": "normalized_lmz_inventory",
                    "provider_archive_accounting": "lmz_public_inventory",
                    "provider_materialization": "lmz_public_call_boundary",
                    "transferred_bytes": "unavailable_not_fabricated",
                    "cancellation_boundary": (
                        "before_and_after_each_public_lmz_call"),
                },
            )
            entries = result.get("entries", [])
            copied = []
            for entry in entries:
                path = entry["path"]
                src_fd = None
                out_fd = None
                try:
                    src_path = _safe_entry_path(provider_destination, path)
                    src_fd = _open_relative_nofollow(provider_destination, src_path)
                    source_stat = os.fstat(src_fd)
                    if not stat.S_ISREG(source_stat.st_mode):
                        raise BundleError("lmz output entry is not regular",
                                          code="provider_failure", path=path)
                    out_fd = _open_relative_fd(stage_fd, path, write=True)
                    copied_bytes = 0
                    while True:
                        block = os.read(src_fd, COPY_CHUNK)
                        if not block:
                            break
                        _write_at(out_fd, block, copied_bytes)
                        copied_bytes += len(block)
                    os.fsync(out_fd)
                    actual = _hash_fd(out_fd)
                    expected_length = int(entry.get("length", entry.get("decoded_bytes",
                                                                        entry.get("size", 0))))
                    if (actual["length"] != expected_length or
                            actual["sha256"] != entry.get("sha256")):
                        raise BundleError("lmz output entry digest mismatch",
                                          code="digest_mismatch", path=path,
                                          expected=entry.get("sha256"), actual=actual)
                    copied.append({**_json_value(entry), **actual, "verified": True})
                finally:
                    if out_fd is not None:
                        os.close(out_fd)
                    if src_fd is not None:
                        os.close(src_fd)
            _check_parent_anchor(parent, parent_fd, parent_token, phase="before_publish")
            _mark(observer, "reconstruction_complete", entries=len(copied),
                  bytes=sum(item["length"] for item in copied), route="lmz")
            _atomic_publish_fd_no_replace(parent_fd, stage_name, name, destination_abs)
            published = True
            stage_name = None
            materialized_bytes = sum(item["length"] for item in copied)
            _add_observer_bytes(observer, materialized=materialized_bytes,
                                 placed=materialized_bytes)
            _mark(observer, "materialization_complete", destination=destination_abs,
                  bytes=materialized_bytes, owner="caller-destination", route="lmz")
            _set_execution(observer, destination_ownership="caller-destination",
                           cleanup={"provider_private_output": "removed",
                                    "owned_staging": "published"},
                           measurement_method={
                               "staging_reserved": "normalized_lmz_inventory",
                               "provider_archive_accounting": "lmz_public_inventory",
                               "provider_materialization": "lmz_public_call_boundary",
                               "transferred_bytes": "unavailable_not_fabricated",
                               "cancellation_boundary": (
                                   "before_and_after_each_public_lmz_call"),
                           })
            return BundleResult(
                destination=destination_abs,
                manifest_sha256=result["manifest_sha256"],
                bundle_sha256=result["bundle_sha256"],
                source_generation={}, entries=tuple(copied),
                materialized_bytes=materialized_bytes,
                transferred_bytes=None, route="lmz", authenticated=False,
                entry_point=result.get("entry_point"),
                entry_point_details=result.get("entry_point_details", {}),
                consumer=result.get("consumer", {}),
            )
        except BaseException as exc:
            error = _as_lmz_error(exc, phase="materialize")
            if isinstance(error, BundleCancelled):
                _mark(observer, "cancelled", boundary=error.details.get("boundary"))
                _set_execution(observer,
                               cancellation_boundary=error.details.get("boundary"))
            _failure(observer, error, phase="lmz_materialize")
            _mark(observer, "release", owner="lmz-provider", cleanup="staging_removed")
            raise error from exc
        finally:
            if stage_fd >= 0 and not published:
                cleanup_state = _remove_owned_stage(parent_fd, stage_name, stage_fd,
                                                     stage_token)
            if stage_fd >= 0:
                os.close(stage_fd)
            if parent_fd >= 0:
                os.close(parent_fd)
            if private_root is not None:
                shutil.rmtree(private_root, ignore_errors=True)
            _set_execution(observer, cleanup={
                "provider_private_output": "removed" if private_root else "none",
                "owned_staging": "published" if published else cleanup_state,
                "staging_exists_after": cleanup_state not in ("removed", "published", "none"),
            }, wall_seconds=time.perf_counter() - started,
                           limits=req.limits())
            if published:
                _mark(observer, "release", owner="lmz-provider", cleanup="complete")


def provider(name: str = "plain") -> BundleProvider:
    """Return a named provider without importing optional implementations."""
    name = str(name).lower()
    if name == "plain":
        return PlainBundleProvider()
    if name == "lmz":
        return LmzBundleProvider()
    raise BundleError(f"unsupported bundle provider {name!r}",
                      code="provider_incompatible", provider=name)


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Return the deterministic identity encoding of a bundle manifest."""
    payload = _identity_payload(manifest)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def make_manifest(source_root: str, entries: list[dict], *, schema="1.0",
                  entry_point=None, consumer=None, resources=None, **metadata) -> dict:
    """Create a complete manifest using the accepted schema-1 field names.

    The plain provider keeps the small ``{"bundle": payload}`` envelope for
    compatibility with directory manifests.  The payload itself is the
    canonical schema shared with the optional lmz provider.  Legacy aliases
    accepted by older callers are translated here so two unequal producer
    shapes are not presented as one canonical manifest.
    """
    root = os.path.abspath(os.fspath(source_root))
    major, minor = _parse_schema(schema)
    built = []
    for item in entries:
        path = _validate_relative_path(item.get("path"))
        file_path = _safe_entry_path(root, path)
        facts = _hash_regular_file(file_path)
        entry = dict(item)
        entry_id = entry.get("id", entry.get("identity")) or _entry_id(path)
        entry["id"] = str(entry_id)
        entry.update({"path": path, "length": facts["length"],
                      "size": facts["length"],
                      "decoded_bytes": facts["length"],
                      "sha256": facts["sha256"]})
        entry_consumer = entry.get("consumer")
        if entry_consumer is not None:
            if not isinstance(entry_consumer, dict):
                raise BundleError("entry consumer must be an object",
                                  code="manifest_shape", path=path)
            entry_consumer = dict(entry_consumer)
            if "required_operators" not in entry_consumer and \
                    "operators" in entry_consumer:
                entry_consumer["required_operators"] = entry_consumer.pop("operators")
            entry_consumer.setdefault("version_range", "*")
            entry["consumer"] = entry_consumer
        built.append(entry)
    by_path = {entry["path"]: entry for entry in built}
    by_id = {entry["id"]: entry for entry in built}
    for entry in built:
        raw_deps = entry.get("dependencies", entry.get("depends_on", [])) or []
        normal_deps = []
        for raw_dep in raw_deps:
            item = {"path": raw_dep} if isinstance(raw_dep, str) else dict(raw_dep)
            target = item.get("path") or item.get("id")
            target_entry = by_path.get(target) or by_id.get(target)
            if target_entry is None:
                raise BundleError("bundle dependency is missing",
                                  code="missing_dependency", path=entry["path"],
                                  dependency=target)
            item["id"] = target_entry["id"]
            item["path"] = target_entry["path"]
            normal_deps.append(item)
        entry["dependencies"] = normal_deps

    payload_metadata = dict(metadata)
    if "preprocessing" in payload_metadata and "preprocess" not in payload_metadata:
        payload_metadata["preprocess"] = payload_metadata.pop("preprocessing")
    if "state" in payload_metadata and "temporal_state" not in payload_metadata:
        payload_metadata["temporal_state"] = payload_metadata.pop("state")
    payload_metadata.setdefault("id", os.path.basename(root.rstrip(os.sep)) or "bundle")
    payload_metadata.setdefault("version", "1")
    payload_metadata.setdefault("source", {"revision": "unknown"})
    payload_metadata.setdefault("license", {
        "classification": "unknown", "redistribution": "unknown",
    })
    consumer_payload = dict(consumer or {})
    if "required_operators" not in consumer_payload and "operators" in consumer_payload:
        consumer_payload["required_operators"] = consumer_payload.pop("operators")
    consumer_payload.setdefault("engine", "unspecified")
    consumer_payload.setdefault("backend", "unspecified")
    consumer_payload.setdefault("version_range", "*")
    consumer_payload.setdefault("required_operators", [])
    consumer_payload.setdefault("extensions", [])
    resource_payload = dict(resources or {})
    if "decode_workspace_bytes" in resource_payload and "decode_workspace" not in resource_payload:
        legacy = resource_payload.pop("decode_workspace_bytes")
        if isinstance(legacy, dict):
            value = legacy.get("bytes", legacy.get("value", 0))
            resource_payload["decode_workspace"] = {
                "bytes": int(value), "measurement": "estimated",
                "method": legacy.get("method", "legacy declaration"),
            }
        else:
            resource_payload["decode_workspace"] = {
                "bytes": int(legacy), "measurement": "estimated",
                "method": "legacy declaration",
            }
    resource_payload.setdefault("decode_workspace", {
        "bytes": 0, "measurement": "unknown", "method": "not measured",
    })
    point = entry_point
    if isinstance(point, str):
        point = {"path": point}
    if isinstance(point, dict):
        point = dict(point)
        target = by_path.get(point.get("path"))
        if target is not None:
            point.setdefault("kind", target.get("role", "opaque"))
    payload = {
        "schema": {"major": major, "minor": minor},
        **payload_metadata,
        "entries": built,
        "entry_point": point,
        "consumer": consumer_payload,
        "resources": resource_payload,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return {"bundle": payload}


def _lmz_payload(manifest: dict, *, descriptor: BundleDescriptor | None = None) -> dict:
    """Translate an lmsluice manifest into lmz's direct accepted payload."""
    payload = _identity_payload(manifest)
    out = _json_value(payload)
    major, minor = _parse_schema(out.get("schema", out.get("version", "1.0")))
    out["schema"] = {"major": major, "minor": minor}
    out.setdefault("id", "lmsluice-bundle")
    out.setdefault("version", "1")
    out.setdefault("source", {"revision": "unknown"})
    out.setdefault("license", {
        "classification": "unknown", "redistribution": "unknown",
    })
    source = out.get("source")
    if not isinstance(source, dict):
        out["source"] = {"revision": str(source)}
    elif not isinstance(source.get("revision"), str):
        out["source"] = {**source, "revision": "unknown"}
    license_info = out.get("license")
    if not isinstance(license_info, dict):
        out["license"] = {"classification": "unknown",
                           "redistribution": "unknown"}
    else:
        out["license"] = {
            **license_info,
            "classification": str(license_info.get("classification", "unknown")),
            "redistribution": str(license_info.get("redistribution", "unknown")),
        }
    if "preprocess" not in out and "preprocessing" in out:
        out["preprocess"] = out.pop("preprocessing")
    if "temporal_state" not in out and "state" in out:
        out["temporal_state"] = out.pop("state")
    if "consumer" not in out and isinstance(out.get("consumer_constraints"), dict):
        out["consumer"] = out.pop("consumer_constraints")
    consumer = dict(out.get("consumer") or {})
    if "required_operators" not in consumer and "operators" in consumer:
        consumer["required_operators"] = consumer.pop("operators")
    consumer.setdefault("engine", "unspecified")
    consumer.setdefault("backend", "unspecified")
    consumer.setdefault("version_range", "*")
    consumer.setdefault("required_operators", [])
    consumer.setdefault("extensions", [])
    out["consumer"] = consumer
    resources = dict(out.get("resources") or {})
    if "decode_workspace" not in resources and "decode_workspace_bytes" in resources:
        legacy = resources.pop("decode_workspace_bytes")
        if isinstance(legacy, dict):
            resources["decode_workspace"] = {
                "bytes": int(legacy.get("bytes", legacy.get("value", 0))),
                "measurement": "estimated",
                "method": legacy.get("method", "legacy declaration"),
            }
        else:
            resources["decode_workspace"] = {
                "bytes": int(legacy), "measurement": "estimated",
                "method": "legacy declaration",
            }
    resources.setdefault("decode_workspace", {
        "bytes": 0, "measurement": "unknown", "method": "not measured",
    })
    out["resources"] = resources

    descriptor_entries = {entry.path: entry for entry in descriptor.entries} \
        if descriptor is not None else {}
    raw_entries = out.get("entries", out.get("artifacts"))
    if not isinstance(raw_entries, list) or not raw_entries:
        if descriptor is None:
            raise BundleError("bundle entries must be a non-empty list",
                              code="manifest_shape")
        raw_entries = [entry.to_dict() for entry in descriptor.entries]
    translated = []
    by_path = {}
    by_id = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise BundleError("bundle entry must be an object", code="manifest_shape")
        item = dict(raw)
        path = _validate_relative_path(item.get("path", item.get("name")))
        descriptor_entry = descriptor_entries.get(path)
        entry_id = item.get("id", item.get("identity"))
        if descriptor_entry is not None:
            entry_id = descriptor_entry.entry_id
        entry_id = str(entry_id or _entry_id(path))
        item["id"] = entry_id
        item["path"] = path
        item["role"] = str(item.get("role", descriptor_entry.role if descriptor_entry else "opaque"))
        if descriptor_entry is not None:
            item.update({"length": descriptor_entry.length,
                         "size": descriptor_entry.length,
                         "decoded_bytes": descriptor_entry.length,
                         "sha256": descriptor_entry.sha256})
            item["dependencies"] = [dep.to_dict() for dep in descriptor_entry.dependencies]
        else:
            raw_length = item.get("decoded_bytes", item.get("size", item.get("length")))
            if raw_length is not None:
                item["decoded_bytes"] = int(raw_length)
                item["size"] = int(raw_length)
                item.setdefault("length", int(raw_length))
        if isinstance(item.get("consumer"), dict):
            constraints = dict(item["consumer"])
            if "required_operators" not in constraints and "operators" in constraints:
                constraints["required_operators"] = constraints.pop("operators")
            constraints.setdefault("version_range", "*")
            item["consumer"] = constraints
        translated.append(item)
        by_path[path] = item
        by_id[entry_id] = item
    for item in translated:
        normal_deps = []
        for raw_dep in item.get("dependencies", []) or []:
            if isinstance(raw_dep, str):
                dep = {"path": raw_dep}
            elif isinstance(raw_dep, dict):
                dep = dict(raw_dep)
            else:
                raise BundleError("dependency must be a path or object",
                                  code="manifest_shape", path=item["path"])
            target = by_path.get(dep.get("path")) or by_id.get(dep.get("id"))
            if target is None:
                raise BundleError("bundle dependency is missing",
                                  code="missing_dependency", path=item["path"],
                                  dependency=dep.get("path", dep.get("id")))
            dep["id"] = target["id"]
            dep["path"] = target["path"]
            normal_deps.append(dep)
        item["dependencies"] = normal_deps
    point = out.get("entry_point", out.get("graph"))
    if isinstance(point, str):
        point = {"path": point}
    if point is None and descriptor is not None and descriptor.entry_point:
        point = dict(descriptor.entry_point_details or {}, path=descriptor.entry_point)
    if isinstance(point, dict):
        point = dict(point)
        point["path"] = _validate_relative_path(point.get("path"))
        target = by_path.get(point["path"])
        if target is not None:
            point.setdefault("kind", target["role"])
        out["entry_point"] = point
    for key in ("manifest_sha256", "identity", "canonical_sha256", "artifacts"):
        out.pop(key, None)
    out["entries"] = translated
    return out


def _copy_descriptor_entries(descriptor: BundleDescriptor, destination: str, content: dict):
    """Copy only verified declared files into an owned lmz source snapshot."""
    os.makedirs(destination, mode=0o700, exist_ok=True)
    for entry in descriptor.entries:
        source_path = _safe_entry_path(descriptor.source_root, entry.path)
        target = os.path.join(destination, *entry.path.split("/"))
        os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
        src_fd = None
        out_fd = None
        try:
            src_fd, _ = _open_verified_entry(descriptor.source_root, source_path, entry,
                                             descriptor.source_generation)
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | \
                getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            out_fd = os.open(target, flags, 0o600)
            offset = 0
            while True:
                block = os.read(src_fd, COPY_CHUNK)
                if not block:
                    break
                _write_at(out_fd, block, offset)
                offset += len(block)
            os.fsync(out_fd)
            actual = _hash_fd(out_fd)
            expected = content.get(entry.path) or {}
            if (actual.get("length") != expected.get("length") or
                    actual.get("sha256") != expected.get("sha256")):
                raise BundleSourceChanged("source snapshot digest changed", path=entry.path)
        finally:
            if out_fd is not None:
                os.close(out_fd)
            if src_fd is not None:
                os.close(src_fd)


def _lmz_request(request, kwargs) -> BundleRequest:
    values = dict(kwargs)
    values.setdefault("route", "lmz")
    return BundleRequest.from_value(request, **values)


def _as_lmz_error(exc, *, phase: str):
    if isinstance(exc, (BundleError, BundleCancelled)):
        return exc
    code = getattr(exc, "code", None) or "provider_failure"
    details = {"provider": "lmz", "phase": phase,
               "provider_exception": type(exc).__name__}
    if hasattr(exc, "as_dict"):
        details["provider_error"] = exc.as_dict()
    return BundleError(str(exc) or type(exc).__name__, code=str(code), **details)


def _normalize_lmz_entry(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise BundleError("lmz entry result is not an object", code="provider_result")
    path = _validate_relative_path(raw.get("path", raw.get("name")))
    raw_length = raw.get("decoded_bytes", raw.get("size", raw.get("length")))
    try:
        length = int(raw_length)
    except (TypeError, ValueError) as exc:
        raise BundleError("lmz entry result lacks decoded length", code="provider_result",
                          path=path) from exc
    if length < 0:
        raise BundleError("lmz entry result has a negative decoded length",
                          code="provider_result", path=path)
    try:
        size = int(raw.get("size", length))
    except (TypeError, ValueError) as exc:
        raise BundleError("lmz entry result has an invalid size",
                          code="provider_result", path=path) from exc
    if size < 0 or size != length:
        raise BundleError("lmz entry result size disagrees with decoded length",
                          code="provider_result", path=path,
                          size=size, length=length)
    digest = _normalize_digest(raw.get("sha256", raw.get("digest")))
    role = str(raw.get("role", "opaque"))
    if role not in SUPPORTED_ROLES:
        raise BundleError("lmz entry result has an unsupported role",
                          code="provider_result", path=path, role=role)
    result = {
        "path": path,
        "role": role,
        "length": length,
        "size": size,
        "decoded_bytes": length,
        "sha256": digest,
        "dependencies": [],
        "representation": _json_value(raw.get("representation", raw.get("mode", "plain"))),
        "verified": True,
    }
    entry_id = raw.get("id", raw.get("identity"))
    if entry_id is not None:
        result["id"] = str(entry_id)
    for key in ("stored_length", "stored_bytes", "complete_stored_cost",
                "compression_smaller", "mode", "consumer"):
        if key in raw:
            result[key] = _json_value(raw[key])
    for raw_dep in raw.get("dependencies", []) or []:
        dep = dict(raw_dep) if isinstance(raw_dep, dict) else {"path": raw_dep}
        if "path" not in dep and "name" in dep:
            dep["path"] = dep.pop("name")
        dep["path"] = _validate_relative_path(dep.get("path"))
        if "id" in dep:
            dep["id"] = str(dep["id"])
        for key in ("offset", "length"):
            if key in dep:
                try:
                    dep[key] = int(dep[key])
                except (TypeError, ValueError) as exc:
                    raise BundleError("lmz dependency range is invalid",
                                      code="provider_result", path=path,
                                      dependency=dep.get("path")) from exc
                if dep[key] < 0:
                    raise BundleError("lmz dependency range is negative",
                                      code="provider_result", path=path,
                                      dependency=dep.get("path"))
        result["dependencies"].append(dep)
    return result


def _normalize_lmz_result(raw: dict, *, operation: str, strict=True) -> dict:
    if not isinstance(raw, dict):
        raise BundleError("lmz provider returned a non-object result",
                          code="provider_result", operation=operation)
    status = raw.get("status")
    if status == "invalid" or raw.get("valid") is False:
        validation = raw.get("validation") or {}
        return {
            "status": "invalid", "valid": False, "strict": bool(strict),
            "route": "lmz", "provider": "lmz",
            "failure": _json_value(validation.get("reason", validation)),
            "archive": raw.get("archive"),
            "materialization": _json_value(raw.get("materialization", {})),
        }
    bundle = raw.get("bundle")
    if not isinstance(bundle, dict):
        raise BundleError("lmz provider result lacks bundle payload",
                          code="provider_result", operation=operation)
    raw_entries = raw.get("entries", bundle.get("entries", []))
    if not isinstance(raw_entries, list):
        raise BundleError("lmz provider entries are not a list",
                          code="provider_result", operation=operation)
    entries = [_normalize_lmz_entry(entry) for entry in raw_entries]
    by_path = {entry["path"]: entry for entry in entries}
    by_id = {entry.get("id"): entry for entry in entries if entry.get("id")}
    if len(by_path) != len(entries) or len(by_id) != len(
            [entry for entry in entries if entry.get("id")]):
        raise BundleError("lmz provider returned duplicate entry identity",
                          code="provider_result", operation=operation)
    for entry in entries:
        resolved = []
        for dep in entry["dependencies"]:
            path_target = by_path.get(dep.get("path"))
            id_target = by_id.get(dep.get("id"))
            if path_target is not None and id_target is not None and \
                    path_target is not id_target:
                raise BundleError("lmz provider returned an ambiguous dependency",
                                  code="provider_result", path=entry["path"],
                                  dependency=dep)
            target = path_target or id_target
            if target is None:
                raise BundleError("lmz provider returned a missing dependency",
                                  code="provider_result", path=entry["path"],
                                  dependency=dep)
            dep["id"] = target.get("id")
            dep["path"] = target["path"]
            offset = dep.get("offset", 0)
            length = dep.get("length")
            if offset > target["length"] or (length is not None and
                                              offset + length > target["length"]):
                raise BundleError("lmz provider returned an invalid dependency range",
                                  code="provider_result", path=entry["path"],
                                  dependency=dep)
            resolved.append(dep)
        entry["dependencies"] = resolved
    point = raw.get("entry_point", bundle.get("entry_point"))
    point_details = {}
    if isinstance(point, dict):
        point_details = _json_value(point)
        point = point.get("path")
    if point is not None:
        point = _validate_relative_path(point)
        point_details.setdefault("path", point)
    identity = raw.get("bundle_manifest_sha256") or raw.get("manifest_sha256")
    if not isinstance(identity, str) or not HEX_SHA256.fullmatch(identity):
        raise BundleError("lmz provider result lacks manifest digest",
                          code="provider_result", operation=operation)
    payload = _json_value(bundle)
    payload["entries"] = entries
    return {
        "status": "complete" if operation == "materialize" and
        raw.get("status") == "complete" else "verified",
        "valid": True,
        "strict": bool(strict),
        "route": "lmz",
        "provider": "lmz",
        "archive": raw.get("archive"),
        "manifest_sha256": identity,
        "bundle_sha256": _bundle_content_digest(entries),
        "publisher_authenticated": bool((raw.get("validation") or {}).get(
            "publisher_authenticated", False)),
        "trusted_expected_digest": bool((raw.get("validation") or {}).get(
            "expected_digest_match", False)),
        "entries": entries,
        "entry_point": point,
        "entry_point_details": point_details,
        "consumer": _json_value(raw.get("consumer", bundle.get("consumer", {}))),
        "resources": _json_value(bundle.get("resources", {})),
        "bundle": payload,
        "archive_accounting": {
            **_json_value(raw.get("archive_accounting", {})),
            "archive_bytes": raw.get("archive_bytes"),
        },
        "workspace": _json_value(raw.get("workspace", {})),
        "materialization": _json_value(raw.get("materialization", {})),
    }


def _check_materialization_matches_inventory(inventory: dict, result: dict):
    """Require the provider's materialization result to close the verified set."""
    expected = {entry["path"]: entry for entry in inventory.get("entries", [])}
    actual = {entry["path"]: entry for entry in result.get("entries", [])}
    if set(expected) != set(actual):
        raise BundleError("lmz materialization changed the verified entry set",
                          code="provider_result", expected_paths=sorted(expected),
                          actual_paths=sorted(actual))
    for path, expected_entry in expected.items():
        actual_entry = actual[path]
        for field in ("id", "role", "length", "sha256"):
            if actual_entry.get(field) != expected_entry.get(field):
                raise BundleError(
                    "lmz materialization result disagrees with strict inventory",
                    code="provider_result", path=path, field=field,
                    expected=expected_entry.get(field), actual=actual_entry.get(field))
        if actual_entry.get("dependencies", []) != expected_entry.get("dependencies", []):
            raise BundleError(
                "lmz materialization dependencies disagree with strict inventory",
                code="provider_result", path=path)
    if result.get("manifest_sha256") != inventory.get("manifest_sha256"):
        raise BundleError("lmz materialization manifest identity changed",
                          code="provider_result",
                          expected=inventory.get("manifest_sha256"),
                          actual=result.get("manifest_sha256"))
    if result.get("bundle_sha256") != inventory.get("bundle_sha256"):
        raise BundleError("lmz materialization bundle identity changed",
                          code="provider_result",
                          expected=inventory.get("bundle_sha256"),
                          actual=result.get("bundle_sha256"))


def _check_normalized_limits(result: dict, request: BundleRequest):
    try:
        total = sum(int(entry.get("length", entry.get("decoded_bytes", 0)))
                    for entry in result.get("entries", []))
    except (TypeError, ValueError) as exc:
        raise BundleError("lmz result contains an invalid byte count",
                          code="provider_result") from exc
    for field in ("max_bytes", "max_staging_bytes"):
        limit = getattr(request, field)
        if limit is None:
            continue
        try:
            normalized_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise BundleError("resource ceiling is invalid", code="resource_limit",
                              field=field, limit=limit) from exc
        if normalized_limit < 0 or total > normalized_limit:
            raise BundleError("declared lmz bundle exceeds caller ceiling",
                              code="resource_limit", field=field,
                              limit=normalized_limit,
                              declared=total)


def _lmz_mark_start(observer, source, request: BundleRequest, operation: str):
    _mark(observer, "bundle_requested", route="lmz", operation=operation,
          expected_manifest_sha256=request.expected_manifest_sha256,
          limits=request.limits())
    _mark(observer, "source_open", source=source, source_kind="lmz_archive",
          provider="lmz")
    _set_route(observer, planned="lmz", actual="lmz", source=source,
               codec="lmz", provider="lmz")


def _lmz_observe_result(observer, result: dict, *, phase: str):
    if result.get("valid"):
        identity = result.get("manifest_sha256")
        _mark(observer, "bundle_resolved", manifest_sha256=identity,
              entries=len(result.get("entries", [])), provider="lmz", phase=phase)
        _set_route(observer, planned="lmz", actual="lmz", codec="lmz",
                   provider="lmz", source=result.get("archive"),
                   bundle_identity=identity)
        _mark(observer, "route_planned", route="lmz", provider="lmz",
              bundle_identity=identity)
        _mark(observer, "bundle_verified", manifest_sha256=identity,
              entries=len(result.get("entries", [])), provider="lmz")
        accounting = result.get("archive_accounting") or {}
        decoded = accounting.get("decoded_bytes")
        coded = accounting.get("unique_payload_bytes")
        if phase in ("create", "validate", "inventory"):
            if isinstance(decoded, int) and decoded >= 0:
                _add_observer_bytes(observer, decoded=decoded, logical=decoded)
            if isinstance(coded, int) and coded >= 0:
                _add_observer_bytes(observer, coded=coded)
        _set_execution(observer,
                        resource_bytes={
                            "provider_archive_bytes": accounting.get("archive_bytes"),
                            "provider_unique_payload_bytes": coded,
                            "provider_decoded_bytes": decoded,
                        },
                        measurement_method={
                            "provider_archive_accounting": "lmz_public_inventory",
                            "transferred_bytes": "unavailable_not_fabricated",
                            "cancellation_boundary": (
                                "before_and_after_each_public_lmz_call"),
                        })
    else:
        error = BundleError("lmz provider returned an invalid bundle",
                            code="provider_failure", provider="lmz",
                            phase=phase, status=result.get("status"),
                            failure=result.get("failure"))
        _set_execution(observer, measurement_method={
            "provider_result": "lmz_public_inventory_failure",
        })
        _failure(observer, error, phase=f"lmz_{phase}")


def write_manifest(path: str, manifest: dict) -> str:
    """Write a stable manifest and return its path."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def resolve_bundle(source, *, request=None, observer=None, **kwargs) -> BundleDescriptor:
    """Resolve and structurally validate a plain bundle without copying it."""
    req = BundleRequest.from_value(request, **kwargs)
    return _resolve_descriptor(source, req, observer=observer)


def validate_bundle(source, *, request=None, provider_name="plain", observer=None,
                    **kwargs):
    return provider(provider_name).validate(source, request=request,
                                            observer=observer, **kwargs)


def inventory_bundle(source, *, request=None, provider_name="plain", strict=True,
                     observer=None, **kwargs):
    return provider(provider_name).inventory(source, request=request, strict=strict,
                                            observer=observer, **kwargs)


def materialize_bundle(source, destination, *, request=None, provider_name="plain",
                      observer=None, **kwargs):
    return provider(provider_name).materialize(source, destination, request=request,
                                               observer=observer, **kwargs)


def _resolve_descriptor(source, request: BundleRequest, *, observer=None) -> BundleDescriptor:
    _mark(observer, "bundle_requested", route=request.route,
          expected_manifest_sha256=request.expected_manifest_sha256,
          limits=request.limits())
    manifest, root, manifest_path = _load_manifest_source(source)
    _mark(observer, "source_open", source=manifest_path or root,
          source_kind="plain_bundle", root=root)
    payload = _identity_payload(manifest)
    schema = _parse_schema(payload.get("schema", payload.get("version", "1.0")))
    if schema[0] != BUNDLE_SCHEMA_MAJOR:
        raise BundleError(f"unsupported bundle schema major {schema[0]}",
                          code="unsupported_major", schema=f"{schema[0]}.{schema[1]}")
    if schema[1] > BUNDLE_SCHEMA_MINOR:
        raise BundleError(f"unsupported bundle schema minor {schema[1]}",
                          code="unsupported_minor", schema=f"{schema[0]}.{schema[1]}")
    identity = canonical_manifest_sha256(manifest)
    for expected, code, label in (
            (request.expected_manifest_sha256, "expected_digest", "manifest"),):
        if expected is not None and _normalize_digest(expected) != identity:
            raise BundleError(f"expected {label} digest does not match",
                              code=code, expected=expected, actual=identity)
    declared = payload.get("manifest_sha256", payload.get("identity"))
    if declared is not None and _normalize_digest(declared) != identity:
        raise BundleError("declared manifest identity does not match",
                          code="manifest_identity", expected=declared, actual=identity)
    entries = _parse_entries(payload)
    _check_dependency_graph(entries)
    entry_point_value = payload.get("entry_point") or payload.get("graph")
    entry_point_details = {}
    entry_point = entry_point_value
    if isinstance(entry_point_value, dict):
        entry_point_details = _json_value(dict(entry_point_value))
        entry_point = entry_point_value.get("path")
    if entry_point is not None:
        entry_point = _validate_relative_path(entry_point)
        if entry_point not in {entry.path for entry in entries}:
            raise BundleError("entry point is not an entry", code="missing_entry",
                              path=entry_point)
        entry_point_details.setdefault("path", entry_point)
        entry_point_details.setdefault(
            "kind", next((entry.role for entry in entries if entry.path == entry_point),
                         "opaque"))
    consumer = payload.get("consumer") or payload.get("consumer_constraints") or {}
    resources = payload.get("resources") or payload.get("resource") or {}
    if not isinstance(consumer, dict) or not isinstance(resources, dict):
        raise BundleError("consumer and resources must be objects", code="manifest_shape")
    generation = None
    if root is not None:
        generation = _snapshot_generation(root, entries, manifest_path)
        if request.source_generation is not None and request.source_generation != generation:
            raise BundleSourceChanged("caller source generation is stale",
                                      expected=request.source_generation,
                                      actual=generation)
    descriptor = BundleDescriptor(
        schema=f"{schema[0]}.{schema[1]}",
        manifest=manifest,
        entries=tuple(entries),
        identity=identity,
        source_root=root,
        manifest_path=manifest_path,
        source_generation=generation,
        entry_point=entry_point,
        entry_point_details=entry_point_details,
        consumer=_json_value(consumer),
        resources=_json_value(resources),
    )
    _mark(observer, "bundle_resolved", manifest_sha256=identity,
          source_generation=generation, entries=len(entries))
    _set_route(observer, planned=request.route, actual=request.route,
               source=manifest_path or root, codec="plain", provider="plain",
               bundle_identity=identity, source_generation=generation)
    _mark(observer, "route_planned", route=request.route, provider="plain",
          bundle_identity=identity)
    return descriptor


def _load_manifest_source(source):
    root = None
    manifest_path = None
    if isinstance(source, BundleDescriptor):
        return source.manifest, source.source_root, source.manifest_path
    if isinstance(source, dict):
        root_value = source.get("source_root") or source.get("root")
        manifest = source.get("manifest", source)
        if root_value is not None:
            root = _safe_directory(os.path.abspath(os.fspath(root_value)), "source root")
        if not isinstance(manifest, dict):
            raise BundleError("manifest must be an object", code="manifest_shape")
        return manifest, root, None
    path = os.path.abspath(os.fspath(source))
    if os.path.isdir(path):
        root = _safe_directory(path, "source root")
        for name in ("bundle.json", "manifest.json"):
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate):
                manifest_path = _safe_regular_path(candidate, "manifest")
                break
        if manifest_path is None:
            raise BundleError("bundle.json or manifest.json is missing",
                              code="missing_manifest", source=root)
    elif os.path.isfile(path):
        manifest_path = _safe_regular_path(path, "manifest")
        root = _safe_directory(os.path.dirname(path), "source root")
    else:
        raise BundleError("plain bundle source is not a directory or manifest",
                          code="missing_source", source=path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, UnicodeError, ValueError) as exc:
        raise BundleError("bundle manifest cannot be read", code="manifest_invalid",
                          source=manifest_path, exception=type(exc).__name__) from exc
    if not isinstance(manifest, dict):
        raise BundleError("bundle manifest must be an object", code="manifest_shape")
    return manifest, root, manifest_path


def _identity_payload(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise BundleError("manifest must be an object", code="manifest_shape")
    payload = manifest.get("bundle")
    if isinstance(payload, dict):
        payload = dict(payload)
    else:
        payload = dict(manifest)
    for key in ("manifest_sha256", "identity", "canonical_sha256"):
        payload.pop(key, None)
    return payload


def _parse_schema(value) -> tuple[int, int]:
    if isinstance(value, dict):
        major, minor = value.get("major"), value.get("minor", 0)
    elif isinstance(value, int):
        major, minor = value, 0
    else:
        match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", str(value))
        if not match:
            raise BundleError("bundle schema version is invalid", code="manifest_shape")
        major, minor = match.group(1), match.group(2) or 0
    try:
        major, minor = int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise BundleError("bundle schema version is invalid", code="manifest_shape") from exc
    if major < 0 or minor < 0:
        raise BundleError("bundle schema version is invalid", code="manifest_shape")
    return major, minor


def _parse_entries(payload: dict) -> list[BundleEntry]:
    raw_entries = payload.get("entries", payload.get("artifacts"))
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BundleError("bundle entries must be a non-empty list", code="manifest_shape")
    entries = []
    paths = set()
    ids = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise BundleError("bundle entry must be an object", code="manifest_shape")
        path = _validate_relative_path(raw.get("path", raw.get("name")))
        if path in paths:
            raise BundleError("duplicate bundle entry path", code="duplicate_path", path=path)
        paths.add(path)
        role = str(raw.get("role", "opaque"))
        if role not in SUPPORTED_ROLES:
            raise BundleError("unsupported bundle entry role", code="unsupported_role",
                              path=path, role=role)
        raw_length = raw.get("length")
        if raw_length is None:
            raw_length = raw.get("decoded_bytes", raw.get("size", raw.get("bytes")))
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise BundleError("bundle entry length is invalid", code="manifest_shape",
                              path=path) from exc
        if length < 0:
            raise BundleError("bundle entry length is negative", code="manifest_shape",
                              path=path)
        digest = _normalize_digest(raw.get("sha256", raw.get("digest")))
        dependencies = []
        raw_deps = raw.get("dependencies", raw.get("depends_on", []))
        if raw_deps is None:
            raw_deps = []
        if not isinstance(raw_deps, list):
            raise BundleError("entry dependencies must be a list", code="manifest_shape",
                              path=path)
        for dep in raw_deps:
            dep_id = None
            if isinstance(dep, str):
                dep_path, offset, dep_length = dep, None, None
            elif isinstance(dep, dict):
                dep_path = dep.get("path", dep.get("entry", dep.get("name")))
                dep_id = dep.get("id", dep.get("identity"))
                if dep_path is None and dep_id is None:
                    raise BundleError("dependency lacks path and id",
                                      code="manifest_shape", path=path)
                offset, dep_length = dep.get("offset"), dep.get("length")
            else:
                raise BundleError("dependency must be a path or object",
                                  code="manifest_shape", path=path)
            if dep_path is not None:
                dep_path = _validate_relative_path(dep_path)
            elif dep_id is not None:
                dep_id = str(dep_id)
            if offset is not None or dep_length is not None:
                try:
                    offset = int(0 if offset is None else offset)
                    dep_length = int(0 if dep_length is None else dep_length)
                except (TypeError, ValueError) as exc:
                    raise BundleError("dependency range is invalid",
                                      code="invalid_range", path=path,
                                      dependency=dep_path) from exc
                if offset < 0 or dep_length < 0:
                    raise BundleError("dependency range is negative",
                                      code="invalid_range", path=path,
                                      dependency=dep_path)
            dependencies.append((dep_path, dep_id, offset, dep_length))
        entry_id = raw.get("id", raw.get("identity"))
        if entry_id is None:
            entry_id = _entry_id(path)
        else:
            entry_id = str(entry_id)
        if entry_id in ids:
            raise BundleError("duplicate bundle entry identity",
                              code="duplicate_identity", identity=entry_id)
        ids.add(entry_id)
        representation = raw.get("representation", raw.get("codec", "plain"))
        stored = raw.get("stored_length", raw.get("stored_bytes"))
        metadata = {}
        for key in ("size", "decoded_bytes", "complete_stored_cost",
                    "compression_smaller", "mode"):
            if key in raw:
                metadata[key] = _json_value(raw[key])
        if stored is not None:
            try:
                stored = int(stored)
            except (TypeError, ValueError) as exc:
                raise BundleError("stored entry length is invalid", code="manifest_shape",
                                  path=path) from exc
            if stored < 0:
                raise BundleError("stored entry length is negative", code="manifest_shape",
                                  path=path)
        entries.append(BundleEntry(
            path=path, role=role, length=length, sha256=digest,
            dependencies=tuple(BundleDependency(dep_path or "", offset, dep_length,
                                                entry_id=dep_id)
                                for dep_path, dep_id, offset, dep_length in dependencies
                                ), entry_id=entry_id,
            representation=representation, stored_length=stored,
            consumer=raw.get("consumer", {}) if isinstance(raw.get("consumer", {}), dict)
            else {}, metadata=metadata,
        ))
    sorted_paths = sorted(paths)
    for left in sorted_paths:
        for right in sorted_paths:
            if left != right and right.startswith(left + "/"):
                raise BundleError("bundle path is both file and directory",
                                  code="path_conflict", path=left, conflicting=right)
    by_path = {entry.path: entry for entry in entries}
    by_id = {entry.entry_id: entry for entry in entries}
    for entry in entries:
        resolved_dependencies = []
        for dep in entry.dependencies:
            target = by_path.get(dep.path) or by_id.get(dep.entry_id)
            if target is None:
                raise BundleError("bundle dependency is missing", code="missing_dependency",
                                  path=entry.path,
                                  dependency=dep.path or dep.entry_id)
            if dep.entry_id is not None and dep.entry_id != target.entry_id:
                raise BundleError("bundle dependency path and id disagree",
                                  code="ambiguous_dependency", path=entry.path,
                                  dependency=dep.path)
            if dep.offset is not None or dep.length is not None:
                offset = dep.offset or 0
                length = dep.length or 0
                if offset + length > target.length:
                    raise BundleError("bundle dependency range exceeds entry",
                                      code="invalid_range", path=entry.path,
                                      dependency=dep.path, offset=offset, length=length,
                                      dependency_length=target.length)
            resolved_dependencies.append(BundleDependency(
                target.path, dep.offset, dep.length, entry_id=target.entry_id))
        entry_index = entries.index(entry)
        entries[entry_index] = BundleEntry(
            path=entry.path, role=entry.role, length=entry.length,
            sha256=entry.sha256, dependencies=tuple(resolved_dependencies),
            entry_id=entry.entry_id, representation=entry.representation,
            stored_length=entry.stored_length, consumer=entry.consumer,
            metadata=entry.metadata,
        )
    return entries


def _check_dependency_graph(entries: list[BundleEntry]) -> None:
    graph = {entry.path: [dep.path for dep in entry.dependencies] for entry in entries}
    active, done = set(), set()

    def visit(path):
        if path in active:
            raise BundleError("bundle dependency cycle", code="dependency_cycle", path=path)
        if path in done:
            return
        active.add(path)
        for child in graph[path]:
            visit(child)
        active.remove(path)
        done.add(path)

    for path in graph:
        visit(path)


def _validate_relative_path(value) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BundleError("bundle path is empty or invalid", code="unsafe_path", path=value)
    if "\\" in value or value.startswith("/") or value.startswith("//") or DRIVE_PATH.match(value):
        raise BundleError("bundle path is not relative POSIX syntax",
                          code="unsafe_path", path=value)
    parts = value.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise BundleError("bundle path contains an unsafe component",
                          code="unsafe_path", path=value)
    return value


def _normalize_digest(value) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise BundleError("bundle SHA-256 is invalid", code="manifest_shape", sha256=value)
    return value.lower()


def _safe_directory(path: str, label: str) -> str:
    path = os.path.abspath(path)
    _safe_chain(path, label, final_directory=True)
    return path


def _safe_regular_path(path: str, label: str) -> str:
    path = os.path.abspath(path)
    _safe_chain(os.path.dirname(path), f"{label} parent", final_directory=True)
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise BundleError(f"{label} is unavailable", code="missing_source",
                          path=path) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise BundleError(f"{label} is not a regular file", code="unsafe_source", path=path)
    return path


def _safe_chain(path: str, label: str, *, final_directory=False):
    path = os.path.abspath(path)
    current = os.path.sep
    for component in [part for part in path.split(os.path.sep) if part]:
        current = os.path.join(current, component)
        try:
            st = os.lstat(current)
        except OSError as exc:
            raise BundleError(f"{label} component is unavailable",
                              code="unsafe_source", path=current) from exc
        if stat.S_ISLNK(st.st_mode):
            raise BundleError(f"{label} contains a symlink", code="unsafe_source",
                              path=current)
        if not stat.S_ISDIR(st.st_mode):
            raise BundleError(f"{label} component is not a directory",
                              code="unsafe_source", path=current)
    if final_directory and not os.path.isdir(path):
        raise BundleError(f"{label} is not a directory", code="unsafe_source", path=path)


def _safe_entry_path(root: str | None, relative: str, *, must_exist=True) -> str:
    relative = _validate_relative_path(relative)
    if root is None:
        raise BundleError("plain bundle has no source root", code="missing_source")
    root = _safe_directory(root, "source root")
    current = root
    parts = relative.split("/")
    for index, component in enumerate(parts):
        current = os.path.join(current, component)
        try:
            st = os.lstat(current)
        except OSError as exc:
            if not must_exist and index == len(parts) - 1:
                return current
            raise BundleError("bundle entry is missing", code="missing_entry",
                              path=relative) from exc
        if stat.S_ISLNK(st.st_mode):
            raise BundleError("bundle entry is a symlink", code="unsafe_source",
                              path=relative)
        if index < len(parts) - 1 and not stat.S_ISDIR(st.st_mode):
            raise BundleError("bundle entry parent is not a directory",
                              code="unsafe_source", path=relative)
        if index == len(parts) - 1 and (not stat.S_ISREG(st.st_mode)):
            raise BundleError("bundle entry is not a regular file",
                              code="unsafe_source", path=relative)
    return current


def _stage_entry_path(stage: str, relative: str) -> str:
    current = stage
    for component in _validate_relative_path(relative).split("/"):
        current = os.path.join(current, component)
    return current


def _stat_token(st) -> dict:
    return {
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "ctime_ns": int(getattr(st, "st_ctime_ns", int(st.st_ctime * 1e9))),
        "device": int(getattr(st, "st_dev", 0)),
        "file_id": int(getattr(st, "st_ino", 0)),
    }


def _snapshot_generation(root: str, entries: list[BundleEntry],
                         manifest_path: str | None = None) -> dict:
    root = _safe_directory(root, "source root")
    try:
        root_stat = os.stat(root)
    except OSError as exc:
        raise BundleError("source root cannot be stat'ed", code="unsafe_source",
                          source=root) from exc
    facts = {}
    for entry in entries:
        path = _safe_entry_path(root, entry.path)
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise BundleError("bundle entry disappeared", code="missing_entry",
                              path=entry.path) from exc
        facts[entry.path] = _stat_token(st)
    generation = {"root": os.path.realpath(root),
                  "root_token": _stat_token(root_stat), "entries": facts}
    if manifest_path is not None:
        try:
            manifest_stat = os.lstat(manifest_path)
        except OSError as exc:
            raise BundleSourceChanged("bundle manifest disappeared") from exc
        if stat.S_ISLNK(manifest_stat.st_mode) or not stat.S_ISREG(manifest_stat.st_mode):
            raise BundleError("bundle manifest is not a regular file",
                              code="unsafe_source", path=manifest_path)
        generation["manifest"] = _stat_token(manifest_stat)
    return generation


def _check_generation(path: str, generation: dict | None, relative: str):
    if generation is None:
        return
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise BundleSourceChanged(path=relative, reason="source disappeared") from exc
    got = _stat_token(st)
    expected = generation.get("entries", {}).get(relative)
    if expected != got:
        raise BundleSourceChanged(path=relative, expected=expected, actual=got)


def _check_generation_root(descriptor: BundleDescriptor):
    if descriptor.source_root is None or descriptor.source_generation is None:
        return
    try:
        got = _stat_token(os.stat(descriptor.source_root))
    except OSError as exc:
        raise BundleSourceChanged(reason="source root disappeared") from exc
    # A materialization destination may intentionally be a child of the
    # source root. Creating and publishing its invocation-owned staging
    # directory changes the root directory's timestamps even though no source
    # entry changed. Root identity is therefore the race check here; every
    # declared source file still gets the full size/timestamp/device/inode
    # comparison above.
    expected_root = descriptor.source_generation.get("root_token") or {}
    if (got.get("device"), got.get("file_id")) != \
            (expected_root.get("device"), expected_root.get("file_id")):
        raise BundleSourceChanged(reason="source root changed",
                                  expected=expected_root,
                                  actual=got)
    expected_manifest = descriptor.source_generation.get("manifest")
    if descriptor.manifest_path is not None and expected_manifest is not None:
        try:
            got_manifest = _stat_token(os.lstat(descriptor.manifest_path))
        except OSError as exc:
            raise BundleSourceChanged("bundle manifest disappeared") from exc
        if got_manifest != expected_manifest:
            raise BundleSourceChanged("bundle manifest changed",
                                      expected=expected_manifest,
                                      actual=got_manifest)


def _open_verified_entry(root: str, path: str, entry: BundleEntry,
                         generation: dict | None):
    _check_generation(path, generation, entry.path)
    try:
        fd = _open_relative_nofollow(root, path)
        st = os.fstat(fd)
    except OSError as exc:
        raise BundleError("bundle entry cannot be opened safely", code="unsafe_source",
                          path=entry.path, exception=type(exc).__name__) from exc
    if not stat.S_ISREG(st.st_mode) or int(st.st_size) != entry.length:
        os.close(fd)
        raise BundleSourceChanged("source entry changed before open", path=entry.path)
    expected = (generation or {}).get("entries", {}).get(entry.path)
    if expected is not None and expected != _stat_token(st):
        os.close(fd)
        raise BundleSourceChanged("source entry identity changed before open",
                                  path=entry.path, expected=expected,
                                  actual=_stat_token(st))
    return fd, _stat_token(st)


def _open_relative_nofollow(root: str, path: str) -> int:
    """Open a regular path through no-follow directory descriptors where able."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if (os.name == "posix" and nofollow and
            getattr(os, "open", None) in getattr(os, "supports_dir_fd", set())):
        root = os.path.abspath(root)
        relative = os.path.relpath(path, root)
        components = relative.split(os.sep)
        current_fd = os.open(root, flags | getattr(os, "O_DIRECTORY", 0) | nofollow)
        try:
            for component in components[:-1]:
                next_fd = os.open(component,
                                  flags | getattr(os, "O_DIRECTORY", 0) | nofollow,
                                  dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            # The final lstat/generation comparison catches a replacement
            # after this descriptor is opened; O_NOFOLLOW covers the final
            # component and the directory walk covers every parent component.
            return os.open(components[-1], flags | nofollow, dir_fd=current_fd)
        finally:
            os.close(current_fd)
    return os.open(path, flags | nofollow)


def _hash_regular_file(path: str) -> dict:
    _safe_regular_path(path, "file")
    digest = hashlib.sha256()
    length = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(COPY_CHUNK)
            if not block:
                break
            digest.update(block)
            length += len(block)
    return {"length": length, "sha256": digest.hexdigest()}


def _verify_source_files(descriptor: BundleDescriptor, request: BundleRequest, *, observer=None):
    if descriptor.source_root is None:
        raise BundleError("plain bundle requires a source root", code="missing_source")
    content = {}
    for entry in descriptor.entries:
        _check_cancel(request.cancellation, f"verify:{entry.path}")
        path = _safe_entry_path(descriptor.source_root, entry.path)
        actual = _hash_regular_file(path)
        if actual["length"] != entry.length:
            raise BundleError("source entry length mismatch", code="length_mismatch",
                              path=entry.path, expected=entry.length,
                              actual=actual["length"])
        if actual["sha256"] != entry.sha256:
            raise BundleError("source entry digest mismatch", code="digest_mismatch",
                              path=entry.path, expected=entry.sha256,
                              actual=actual["sha256"])
        _check_generation(path, descriptor.source_generation, entry.path)
        content[entry.path] = {**actual, "verified": True}
    _check_generation_root(descriptor)
    actual_bundle = _bundle_content_digest(descriptor.entries)
    if request.expected_bundle_sha256 is not None and \
            _normalize_digest(request.expected_bundle_sha256) != actual_bundle:
        raise BundleError("expected bundle digest does not match", code="expected_digest",
                          expected=request.expected_bundle_sha256, actual=actual_bundle)
    return content


def _bundle_content_digest(entries: tuple[BundleEntry, ...] | list[BundleEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.path if isinstance(item, BundleEntry)
                        else item.get("path")):
        path = entry.path if isinstance(entry, BundleEntry) else entry.get("path")
        length = entry.length if isinstance(entry, BundleEntry) else entry.get(
            "length", entry.get("decoded_bytes", entry.get("size")))
        sha256 = entry.sha256 if isinstance(entry, BundleEntry) else entry.get(
            "sha256", entry.get("digest"))
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(length).to_bytes(8, "big", signed=False))
        digest.update(bytes.fromhex(sha256))
    return digest.hexdigest()


def _validation_dict(descriptor, content, request):
    return {
        "valid": True,
        "schema": descriptor.schema,
        "manifest_sha256": descriptor.identity,
        "bundle_sha256": _bundle_content_digest(descriptor.entries),
        "publisher_authenticated": False,
        "trusted_expected_digest": bool(request.expected_manifest_sha256 or
                                        request.expected_bundle_sha256),
        "source_generation": descriptor.source_generation,
        "entries": [dict(entry.to_dict(), actual=content[entry.path])
                    for entry in descriptor.entries],
        "entry_point": descriptor.entry_point,
        "entry_point_details": descriptor.entry_point_details,
        "consumer": descriptor.consumer,
        "resources": descriptor.resources,
    }


def _inventory_dict(descriptor, content, request, *, strict):
    return {
        "valid": all(item.get("verified") for item in content.values()) if strict else None,
        "strict": bool(strict),
        "schema": descriptor.schema,
        "manifest_sha256": descriptor.identity,
        "bundle_sha256": _bundle_content_digest(descriptor.entries),
        "publisher_authenticated": False,
        "trusted_expected_digest": bool(request.expected_manifest_sha256 or
                                        request.expected_bundle_sha256),
        "source_generation": descriptor.source_generation,
        "entry_point": descriptor.entry_point,
        "entry_point_details": descriptor.entry_point_details,
        "consumer": descriptor.consumer,
        "resources": descriptor.resources,
        "declared_materialized_bytes": descriptor.total_bytes,
        "entries": [dict(entry.to_dict(), actual=content[entry.path])
                    for entry in descriptor.entries],
    }


def _prepare_destination(destination):
    path = os.path.abspath(os.fspath(destination))
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    if not name or name in (".", ".."):
        raise BundleError("destination name is invalid", code="unsafe_destination",
                          destination=path)
    try:
        _safe_directory(parent, "destination parent")
    except BundleError as exc:
        details = dict(exc.details)
        details.setdefault("destination", parent)
        raise BundleError(str(exc), code="unsafe_destination",
                          **details) from exc
    if os.path.lexists(path):
        raise BundleError("materialization destination already exists",
                          code="destination_exists", destination=path)
    return path, parent, name


def _prepare_destination_anchor(destination):
    """Preflight and hold the intended destination parent by descriptor."""
    path, parent, name = _prepare_destination(destination)
    try:
        expected = _stat_token(os.lstat(parent))
    except OSError as exc:
        raise BundleError("destination parent cannot be inspected safely",
                          code="unsafe_destination", destination=parent) from exc
    parent_fd, _opened_token = _open_stable_directory(
        parent, "destination parent", expected=expected)
    try:
        _check_destination_absent(parent_fd, name, path)
    except BaseException:
        os.close(parent_fd)
        raise
    return path, parent, name, parent_fd, expected


def _open_stable_directory(path: str, label: str, *, expected=None) -> tuple[int, dict]:
    """Open a no-follow directory chain and verify its preflight identity."""
    if (os.name != "posix" or not getattr(os, "O_NOFOLLOW", 0) or
            not getattr(os, "O_DIRECTORY", 0) or
            os.open not in getattr(os, "supports_dir_fd", ()) or
            os.stat not in getattr(os, "supports_dir_fd", ()) or
            os.mkdir not in getattr(os, "supports_dir_fd", ())):
        raise BundleError("stable directory descriptors are unavailable",
                          code="atomic_publish_unavailable", path=path)
    path = os.path.abspath(path)
    try:
        observed = _stat_token(os.lstat(path))
    except OSError as exc:
        raise BundleError(f"{label} cannot be inspected safely",
                          code="unsafe_destination", path=path) from exc
    if expected is not None and observed != expected:
        raise BundleError(f"{label} changed during preflight",
                          code="destination_parent_changed", path=path,
                          expected=expected, actual=observed)
    try:
        _safe_chain(path, label, final_directory=True)
    except BundleError as exc:
        details = dict(exc.details)
        details.setdefault("path", path)
        raise BundleError(str(exc), code="unsafe_destination", path=path,
                          **{key: value for key, value in details.items()
                             if key != "path"}) from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(os.path.sep, flags)
    try:
        components = [part for part in path.split(os.path.sep) if part]
        for component in components:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        actual = _stat_token(os.fstat(current_fd))
        if expected is not None and actual != expected:
            raise BundleError(f"{label} changed while opening",
                              code="destination_parent_changed", path=path,
                              expected=expected, actual=actual)
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise BundleError(f"{label} is not a directory",
                              code="unsafe_destination", path=path)
        return current_fd, actual
    except OSError as exc:
        try:
            os.close(current_fd)
        except OSError:
            pass
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BundleError(f"{label} contains a symlink or non-directory",
                              code="unsafe_destination", path=path) from exc
        raise BundleError(f"{label} cannot be opened safely",
                          code="unsafe_destination", path=path,
                          exception=type(exc).__name__) from exc
    except BaseException:
        try:
            os.close(current_fd)
        except OSError:
            pass
        raise


def _check_parent_anchor(parent: str, parent_fd: int, expected: dict, *, phase: str):
    """Reject a parent path replacement while retaining the original fd."""
    try:
        path_stat = os.lstat(parent)
        fd_stat = os.fstat(parent_fd)
    except OSError as exc:
        raise BundleError("destination parent changed or disappeared",
                          code="destination_parent_changed", path=parent,
                          phase=phase, exception=type(exc).__name__) from exc
    if (not stat.S_ISDIR(path_stat.st_mode) or
            _stat_token(path_stat).get("device") != expected.get("device") or
            _stat_token(path_stat).get("file_id") != expected.get("file_id") or
            _stat_token(fd_stat).get("device") != expected.get("device") or
            _stat_token(fd_stat).get("file_id") != expected.get("file_id")):
        raise BundleError("destination parent changed",
                          code="destination_parent_changed", path=parent,
                          phase=phase, expected=expected,
                          actual=_stat_token(path_stat))


def _check_destination_absent(parent_fd: int, name: str, destination: str):
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise BundleError("destination cannot be inspected safely",
                          code="unsafe_destination", destination=destination) from exc
    raise BundleError("materialization destination already exists",
                      code="destination_exists", destination=destination)


def _create_owned_stage(parent_fd: int) -> tuple[str, int, dict]:
    """Create a private staging directory relative to the held parent fd."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(32):
        stage_name = f".lmsluice-bundle-{os.getpid()}-{os.urandom(12).hex()}"
        try:
            os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            stage_fd = os.open(stage_name, flags, dir_fd=parent_fd)
            token = _stat_token(os.fstat(stage_fd))
            return stage_name, stage_fd, token
        except BaseException:
            try:
                os.rmdir(stage_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    raise BundleError("cannot allocate invocation-owned staging directory",
                      code="staging_unavailable")


def _mkdir_relative_fd(root_fd: int, parts: list[str]) -> int:
    current_fd = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            try:
                os.mkdir(part, 0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_relative_fd(root_fd: int, relative: str, *, write=False) -> int:
    parts = _validate_relative_path(relative).split("/")
    parent_fd = _mkdir_relative_fd(root_fd, parts[:-1])
    try:
        flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY)
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(parts[-1], flags, 0o600, dir_fd=parent_fd) if write else \
            os.open(parts[-1], flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _hash_fd(fd: int) -> dict:
    st_before = os.fstat(fd)
    if not stat.S_ISREG(st_before.st_mode):
        raise BundleError("staging output is not a regular file", code="unsafe_destination")
    read_fd = os.dup(fd)
    digest = hashlib.sha256()
    length = 0
    try:
        os.lseek(read_fd, 0, os.SEEK_SET)
        while True:
            block = os.read(read_fd, COPY_CHUNK)
            if not block:
                break
            digest.update(block)
            length += len(block)
    finally:
        os.close(read_fd)
    st_after = os.fstat(fd)
    if _stat_token(st_before) != _stat_token(st_after):
        raise BundleSourceChanged("staging output changed while hashing")
    return {"length": length, "sha256": digest.hexdigest()}


def _remove_owned_tree(directory_fd: int):
    """Remove only descendants addressed through the owned directory fd."""
    for name in list(os.listdir(directory_fd)):
        try:
            st = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(st.st_mode):
            child_fd = None
            try:
                child_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                   getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                child_token = _stat_token(os.fstat(child_fd))
                if child_token.get("device") != _stat_token(st).get("device") or \
                        child_token.get("file_id") != _stat_token(st).get("file_id"):
                    raise OSError(errno.EAGAIN, "staging child changed while opening")
                _remove_owned_tree(child_fd)
            finally:
                if child_fd is not None:
                    os.close(child_fd)
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                current_token = _stat_token(current)
                if (current_token.get("device") != child_token.get("device") or
                        current_token.get("file_id") != child_token.get("file_id")):
                    raise OSError(errno.EAGAIN, "staging child ownership changed")
                os.rmdir(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        else:
            file_fd = None
            try:
                file_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                                  dir_fd=directory_fd)
                opened = _stat_token(os.fstat(file_fd))
                original = _stat_token(st)
                if (opened.get("device") != original.get("device") or
                        opened.get("file_id") != original.get("file_id")):
                    raise OSError(errno.EAGAIN, "staging file ownership changed")
            finally:
                if file_fd is not None:
                    os.close(file_fd)
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                current_token = _stat_token(current)
                if (current_token.get("device") != original.get("device") or
                        current_token.get("file_id") != original.get("file_id")):
                    raise OSError(errno.EAGAIN, "staging file ownership changed")
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _remove_owned_stage(parent_fd: int, stage_name: str | None, stage_fd: int,
                        stage_token: dict | None) -> str:
    if stage_fd < 0:
        return "none"
    try:
        _remove_owned_tree(stage_fd)
    except OSError:
        return "descendant_cleanup_failed"
    if stage_name is None or stage_token is None:
        return "removed_descendants"
    try:
        st = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "ownership_missing_retained"
    except OSError:
        return "ownership_uncheckable"
    if _stat_token(st).get("device") != stage_token.get("device") or \
            _stat_token(st).get("file_id") != stage_token.get("file_id"):
        return "ownership_mismatch_retained"
    try:
        os.rmdir(stage_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return "removed"
    except OSError:
        return "ownership_check_failed"
    return "removed"


def _atomic_publish_fd_no_replace(parent_fd: int, stage_name: str, name: str,
                                  destination: str):
    """Atomically publish using the already verified parent descriptor."""
    if os.name != "posix":
        raise BundleError("atomic no-replace publication is unavailable",
                          code="atomic_publish_unavailable")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise BundleError("atomic no-replace publication is unavailable",
                              code="atomic_publish_unavailable")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(parent_fd, os.fsencode(stage_name), parent_fd,
                           os.fsencode(name), RENAME_NOREPLACE)
        if result != 0:
            error_no = ctypes.get_errno()
            if error_no == errno.EEXIST:
                raise BundleError("materialization destination appeared",
                                  code="destination_exists", destination=destination)
            raise BundleError("atomic no-replace publication failed",
                              code="atomic_publish_failed", errno=error_no,
                              destination=destination)
        os.fsync(parent_fd)
    except BundleError:
        raise
    except OSError as exc:
        raise BundleError("atomic no-replace publication failed",
                          code="atomic_publish_failed", destination=destination,
                          exception=type(exc).__name__) from exc


def _atomic_publish_no_replace(stage: str, parent: str, name: str):
    """Publish a same-filesystem directory atomically without replacement.

    Linux/WSL exposes ``renameat2(RENAME_NOREPLACE)`` through libc.  A plain
    ``os.rename`` is deliberately not used because it can replace a directory
    that appeared after the preflight check.  Platforms without this primitive
    fail closed instead of weakening the destination ownership guarantee.
    """
    expected = _stat_token(os.lstat(parent))
    parent_fd, _ = _open_stable_directory(parent, "destination parent", expected=expected)
    try:
        _atomic_publish_fd_no_replace(parent_fd, os.path.basename(stage), name,
                                      os.path.join(parent, name))
    finally:
        os.close(parent_fd)


def _write_at(fd: int, payload: bytes, offset: int):
    view = memoryview(payload)
    written = 0
    while written < len(view):
        if hasattr(os, "pwrite"):
            count = os.pwrite(fd, view[written:], offset + written)
        else:  # pragma: no cover - native Windows fallback
            os.lseek(fd, offset + written, os.SEEK_SET)
            count = os.write(fd, view[written:])
        if not count:
            raise OSError("short write while materializing bundle")
        written += count


def _add_report(total: Report, report: Report):
    for name in ("jobs", "fetched_bytes", "placed_bytes", "fetch_seconds",
                 "place_seconds", "stalls_full", "stalls_empty"):
        setattr(total, name, getattr(total, name) + getattr(report, name))
    total.seconds += report.seconds
    total.failed = total.failed or report.failed
    if report.error_type:
        total.error_type = report.error_type


def _limit(value):
    return value if value is not None else (1 << 62)


def _check_limits(descriptor: BundleDescriptor, request: BundleRequest):
    total = descriptor.total_bytes
    for name, limit in (("max_bytes", request.max_bytes),
                        ("max_staging_bytes", request.max_staging_bytes),
                        ("max_transferred_bytes", request.max_transferred_bytes)):
        if limit is None:
            continue
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise BundleError("resource ceiling is invalid", code="resource_limit",
                              limit=limit, field=name) from exc
        if limit < 0 or total > limit:
            raise BundleError("declared bundle exceeds caller ceiling",
                              code="resource_limit", field=name, limit=limit,
                              declared=total)


def _check_cancel(cancellation, boundary: str):
    if cancellation is None:
        return
    try:
        if callable(cancellation):
            cancelled = bool(cancellation())
        elif hasattr(cancellation, "is_set"):
            cancelled = bool(cancellation.is_set())
        elif hasattr(cancellation, "cancelled"):
            cancelled = bool(cancellation.cancelled)
        else:
            cancelled = bool(cancellation)
    except Exception as exc:
        raise BundleCancelled("cancellation probe failed", boundary=boundary,
                              exception=type(exc).__name__) from exc
    if cancelled:
        raise BundleCancelled(boundary=boundary)


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "bytes": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _mark(observer, event, **details):
    if observer is None:
        return
    try:
        marker = getattr(observer, "mark", None)
        if marker is not None:
            marker(event, **details)
    except Exception:
        return


def _failure(observer, exc, *, phase):
    if observer is None:
        return
    try:
        callback = getattr(observer, "failure_event", None)
        if callback is not None:
            callback(exc, phase=phase)
    except Exception:
        return


def _attach_report(observer, report):
    if observer is None:
        return
    try:
        callback = getattr(observer, "attach_report", None)
        if callback is not None:
            callback(report)
    except Exception:
        return


def _add_observer_bytes(observer, **values):
    if observer is None:
        return
    try:
        callback = getattr(observer, "add_bytes", None)
        if callback is not None:
            callback(**values)
    except Exception:
        return


def _set_execution(observer, **values):
    if observer is None:
        return
    try:
        callback = getattr(observer, "set_execution", None)
        if callback is not None:
            callback(**values)
    except Exception:
        return


def _set_route(observer, **values):
    if observer is None:
        return
    try:
        callback = getattr(observer, "set_route", None)
        if callback is not None:
            callback(**values)
    except Exception:
        return


def _lmz_digest_kwargs(request: BundleRequest) -> dict:
    out = {}
    if request.expected_manifest_sha256 is not None:
        out["expected_manifest_sha256"] = request.expected_manifest_sha256
    if request.expected_bundle_sha256 is not None:
        out["expected_bundle_sha256"] = request.expected_bundle_sha256
    return out


__all__ = [
    "BUNDLE_SCHEMA_MAJOR", "BUNDLE_SCHEMA_MINOR", "SUPPORTED_ROLES",
    "BundleError", "BundleSourceChanged", "BundleCancelled",
    "BundleDependency", "BundleEntry", "BundleDescriptor", "BundleRequest",
    "BundleResult", "BundleProvider", "PlainBundleProvider", "LmzBundleProvider",
    "canonical_manifest_bytes", "canonical_manifest_sha256", "make_manifest",
    "write_manifest", "provider", "resolve_bundle", "validate_bundle",
    "inventory_bundle", "materialize_bundle",
]
