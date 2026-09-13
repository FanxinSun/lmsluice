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

    def to_dict(self) -> dict:
        out = {"path": self.path}
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
    representation: str = "plain"
    stored_length: int | None = None
    consumer: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "path": self.path,
            "role": self.role,
            "length": self.length,
            "sha256": self.sha256,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "representation": self.representation,
        }
        if self.entry_id is not None:
            out["id"] = self.entry_id
        if self.stored_length is not None:
            out["stored_length"] = self.stored_length
        if self.consumer:
            out["consumer"] = _json_value(self.consumer)
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
    transferred_bytes: int
    route: str = "plain"
    authenticated: bool = False
    entry_point: str | None = None
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
        stage = None
        published = False
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
            destination_abs, parent, name = _prepare_destination(destination)
            stage = tempfile.mkdtemp(prefix=f".lmsluice-bundle-{os.getpid()}-",
                                     dir=parent)
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
                out_path = _stage_entry_path(stage, entry.path)
                out_fd = None
                src_fd = None
                try:
                    src_fd, before = _open_verified_entry(
                        descriptor.source_root, src_path, entry,
                        descriptor.source_generation)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    out_fd = os.open(out_path, flags, 0o600)
                    source = _FdSource(src_fd, src_path, entry.length)
                    src_fd = None
                    jobs = [(offset, min(max(1, int(chunk_bytes)),
                                         entry.length - offset))
                            for offset in range(0, entry.length,
                                                max(1, int(chunk_bytes)))]

                    def fetch(job):
                        _check_cancel(req.cancellation, f"among_fetches:{entry.path}")
                        return source.pread(job[0], job[1])

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
                    os.close(out_fd)
                    out_fd = None
                    source.close()
                    source = None
                    actual = _hash_regular_file(out_path)
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
                    if 'source' in locals() and source is not None:
                        source.close()
                        source = None
            _check_generation_root(descriptor)
            _check_cancel(req.cancellation, "before_publish")
            _mark(observer, "transfer_complete", bytes=transferred, route="plain")
            _mark(observer, "reconstruction_complete", entries=len(materialized),
                  bytes=descriptor.total_bytes)
            _atomic_publish_no_replace(stage, parent, name)
            published = True
            stage = None
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
            if stage is not None and not published:
                shutil.rmtree(stage, ignore_errors=True)
            _set_execution(observer, cleanup={
                "owned_staging": "published" if published else "removed",
                "staging_exists_after": bool(stage and os.path.lexists(stage)),
            })
            _set_execution(observer, wall_seconds=time.perf_counter() - started,
                           limits=req.limits(), destination_ownership=(
                               "published" if published else "staging_removed"))


class LmzBundleProvider(BundleProvider):
    """Lazy adapter for the accepted public lmz bundle API.

    It forwards validation, inventory, creation and materialization calls; it
    deliberately contains no archive parser or codec fallback.  The import is
    attempted only when this provider is selected or its capability is queried.
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
        fn = self._function("create_bundle")
        args = {"manifest": manifest, **kwargs} if manifest is not None else kwargs
        return fn(source_root, archive, **args)

    def validate(self, source, *, request=None, observer=None, **kwargs):
        del observer
        req = BundleRequest.from_value(request, **kwargs)
        fn = self._function("validate_bundle")
        return fn(source, **_lmz_digest_kwargs(req))

    def inventory(self, source, *, request=None, strict=True, observer=None, **kwargs):
        del observer
        req = BundleRequest.from_value(request, **kwargs)
        fn = self._function("inventory_bundle")
        return fn(source, strict=strict, **_lmz_digest_kwargs(req))

    def materialize(self, source, destination, *, request=None, observer=None, **kwargs):
        del observer
        req = BundleRequest.from_value(request, **kwargs)
        fn = self._function("materialize_bundle")
        return fn(source, destination, **_lmz_digest_kwargs(req))


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
    """Create a complete deterministic plain manifest from source files."""
    root = os.path.abspath(os.fspath(source_root))
    built = []
    for item in entries:
        path = _validate_relative_path(item.get("path"))
        file_path = _safe_entry_path(root, path)
        facts = _hash_regular_file(file_path)
        entry = dict(item)
        entry.update({"path": path, "length": facts["length"],
                      "sha256": facts["sha256"]})
        built.append(entry)
    payload = {
        "schema": str(schema),
        "entries": built,
        "entry_point": entry_point,
        "consumer": consumer or {},
        "resources": resources or {},
        **metadata,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return {"bundle": payload}


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
    entry_point = payload.get("entry_point") or payload.get("graph")
    if entry_point is not None:
        entry_point = _validate_relative_path(entry_point)
        if entry_point not in {entry.path for entry in entries}:
            raise BundleError("entry point is not an entry", code="missing_entry",
                              path=entry_point)
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
        try:
            length = int(raw.get("length", raw.get("bytes")))
        except (TypeError, ValueError) as exc:
            raise BundleError("bundle entry length is invalid", code="manifest_shape",
                              path=path) from exc
        if length < 0:
            raise BundleError("bundle entry length is negative", code="manifest_shape",
                              path=path)
        digest = _normalize_digest(raw.get("sha256", raw.get("digest")))
        entry_id = raw.get("id", raw.get("identity"))
        if entry_id is not None:
            entry_id = str(entry_id)
            if entry_id in ids:
                raise BundleError("duplicate bundle entry identity",
                                  code="duplicate_identity", identity=entry_id)
            ids.add(entry_id)
        dependencies = []
        raw_deps = raw.get("dependencies", raw.get("depends_on", []))
        if raw_deps is None:
            raw_deps = []
        if not isinstance(raw_deps, list):
            raise BundleError("entry dependencies must be a list", code="manifest_shape",
                              path=path)
        for dep in raw_deps:
            if isinstance(dep, str):
                dep_path, offset, dep_length = dep, None, None
            elif isinstance(dep, dict):
                dep_path = dep.get("path", dep.get("entry", dep.get("name")))
                offset, dep_length = dep.get("offset"), dep.get("length")
            else:
                raise BundleError("dependency must be a path or object",
                                  code="manifest_shape", path=path)
            dep_path = _validate_relative_path(dep_path)
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
            dependencies.append(BundleDependency(dep_path, offset, dep_length))
        representation = str(raw.get("representation", raw.get("codec", "plain")))
        stored = raw.get("stored_length", raw.get("stored_bytes"))
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
            dependencies=tuple(dependencies), entry_id=entry_id,
            representation=representation, stored_length=stored,
            consumer=raw.get("consumer", {}) if isinstance(raw.get("consumer", {}), dict)
            else {},
        ))
    sorted_paths = sorted(paths)
    for left in sorted_paths:
        for right in sorted_paths:
            if left != right and right.startswith(left + "/"):
                raise BundleError("bundle path is both file and directory",
                                  code="path_conflict", path=left, conflicting=right)
    by_path = {entry.path: entry for entry in entries}
    for entry in entries:
        for dep in entry.dependencies:
            target = by_path.get(dep.path)
            if target is None:
                raise BundleError("bundle dependency is missing", code="missing_dependency",
                                  path=entry.path, dependency=dep.path)
            if dep.offset is not None or dep.length is not None:
                offset = dep.offset or 0
                length = dep.length or 0
                if offset + length > target.length:
                    raise BundleError("bundle dependency range exceeds entry",
                                      code="invalid_range", path=entry.path,
                                      dependency=dep.path, offset=offset, length=length,
                                      dependency_length=target.length)
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
    for entry in sorted(entries, key=lambda item: item.path):
        digest.update(entry.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(entry.length).to_bytes(8, "big", signed=False))
        digest.update(bytes.fromhex(entry.sha256))
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
    _safe_directory(parent, "destination parent")
    if os.path.lexists(path):
        raise BundleError("materialization destination already exists",
                          code="destination_exists", destination=path)
    return path, parent, name


def _atomic_publish_no_replace(stage: str, parent: str, name: str):
    """Publish a same-filesystem directory atomically without replacement.

    Linux/WSL exposes ``renameat2(RENAME_NOREPLACE)`` through libc.  A plain
    ``os.rename`` is deliberately not used because it can replace a directory
    that appeared after the preflight check.  Platforms without this primitive
    fail closed instead of weakening the destination ownership guarantee.
    """
    if os.name != "posix":
        raise BundleError("atomic no-replace publication is unavailable",
                          code="atomic_publish_unavailable")
    stage_name = os.path.basename(stage)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        raise BundleError("destination parent cannot be opened safely",
                          code="unsafe_destination", destination=parent) from exc
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
                                  code="destination_exists", destination=os.path.join(parent, name))
            raise BundleError("atomic no-replace publication failed",
                              code="atomic_publish_failed", errno=error_no,
                              destination=os.path.join(parent, name))
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
