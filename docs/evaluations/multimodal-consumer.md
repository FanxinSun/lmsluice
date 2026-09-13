# MM-SLUICE-01 bundle delivery and consumer boundary

This document records the lmsluice side of the multimodal artifact-to-consumer
contract. It covers verified complete-bundle delivery, owned materialization,
additive lifecycle evidence and the optional ONNX Runtime CPU boundary. It is
an engineering mechanics unit. The generated fixture does not provide ASR,
TTS, vision, language or specialist task quality evidence.

## Contract and ownership

`lmsluice.bundle` accepts a plain bundle source directory containing
`bundle.json` (or `manifest.json`) and every file listed by its `entries` list.
A manifest may use the accepted producer shape with a top-level `bundle` object;
that object is the canonical identity payload. The versioned payload contains:

* `schema`, `entry_point`, consumer constraints and declared resources;
* complete entries with relative POSIX `path`, `role`, byte `length` and
  SHA-256; and
* dependency paths with optional `offset` and `length` ranges.

The supported roles are `graph`, `weights`, `config`, `preprocess`,
`vocabulary`, `calibration` and `opaque`. The canonical manifest digest is
deterministic UTF-8 JSON with sorted keys and compact separators. Every entry's
path, length, digest, role and dependency metadata therefore contributes to the
bundle identity. `bundle_sha256` in validation and materialization results is a
stable digest of the sorted entry content identities (path, length and SHA-256).

The digest says that bytes equal an expected digest. It does not authenticate a
publisher. Results expose `publisher_authenticated: false` and only identify a
caller supplied digest as a trusted equality check. A filesystem stat or cache
generation token is never treated as content authentication.

The provider owns artifact format handling. `PlainBundleProvider` uses the
existing lmsluice `Source` and bounded two-stage `transport` primitives to
fetch source ranges into a private staging tree. `LmzBundleProvider` lazily
delegates `create_bundle`, `validate_bundle`, `inventory_bundle` and
`materialize_bundle` to the accepted public lmz API. It contains no lmz archive
parser, copied codec or materializer. No lmz import is needed for the plain
route or core import, and an absent or incomplete lmz API is reported as
`NOT_RUN` by the probe.

The public core surface is:

```python
from lmsluice.bundle import (
    BundleEntry, BundleDependency, BundleDescriptor, BundleRequest,
    canonical_manifest_sha256, inventory_bundle, materialize_bundle,
    resolve_bundle, validate_bundle,
)

descriptor = resolve_bundle("bundle-directory")
validated = validate_bundle("bundle-directory",
                            expected_manifest_sha256=descriptor.identity)
result = materialize_bundle("bundle-directory", "owned-destination",
                            max_bytes=64 << 20)
```

`BundleRequest` carries expected identity, source generation, consumer/entry
point selection, cancellation and transport/materialization ceilings. The
plain materializer verifies all source entries before publishing and verifies
the actual staged length and digest again after transport. Invalid ranges,
missing dependencies, cycles, duplicate paths or identities, unsupported
versions and file/directory conflicts fail before readiness.

## Safe materialization

Only canonical relative POSIX paths are accepted. Absolute, drive-letter, UNC,
backslash, empty-component, dot and traversal forms are rejected. Source root,
manifest parents and entry parents are checked with `lstat`; symlinks and
special files are rejected. The final source open uses `O_NOFOLLOW` where the
platform provides it and compares the descriptor identity with the resolved
generation. A source digest mismatch or size/device/inode/timestamp change
raises a structured failure and does not yield a complete result.

The destination must have an existing safe parent and must not already exist.
All files are created below a fresh invocation-owned staging directory on that
same filesystem. On POSIX, publication uses Linux/WSL
`renameat2(RENAME_NOREPLACE)`. A destination that appears after preflight is
retained and the operation fails; a plain replacing rename is not used. On a
platform without a safe atomic no-replace primitive the operation fails closed.
Failure cleanup removes only the invocation's staging tree. It never removes a
caller destination or unrelated path.

The caller ceilings are checked against declared complete entry bytes before
work begins. `max_bytes` and `max_staging_bytes` cover the declared materialized
payload; `max_transferred_bytes` covers source bytes fetched by this operation.
The queue and chunk size bound transport-owned in-flight payloads. Session,
engine, activation, whole-process RSS and device memory are separate
measurements and are never hidden in the transport ceiling.

Cancellation is cooperative at resolve, before and among entry fetches and
places, after fetch, and before publication. It cleans descriptors, workers and
owned staging. It does not claim to preempt an already-running backend call or
provide durable resume.

## Lifecycle evidence

The existing schema-1 `ReadinessRecord` remains additive. Old tensor events keep
their meaning. Complete-bundle records may add:

```text
bundle_requested → bundle_resolved → route_planned → fetch_started
→ first_payload → reconstruction_complete → materialization_complete
→ consumer_initialized → use_started → consumer_first_valid_output
→ consumer_ready → release
```

`source_open`, `allocation`, `staging`, `transfer_complete` and
`terminal_failure` remain available in the same record. Event time uses one
`time.monotonic_ns` origin; absent stages remain JSON `null`. The record also
contains provider/bundle/source-generation route facts, fetched/decoded/
materialized/repeated bytes, caller limits, destination ownership, cleanup,
cancellation and backend-memory measurement status. RSS/PSS/HWM sampling is
current-process best effort. Child pages, allocator-specific, device and
energy measurements remain explicitly `UNMEASURED` when no facility is
available.

Transport completion, a decoded payload, a materialized graph or a created
session does not create a consumer result. `use_started` marks the beginning of
the caller's inference invocation; `consumer_first_valid_output` and
`consumer_ready` are caller-owned. The optional adapter emits the latter events
only after a caller validity hook accepts the actual output. Rejected output
records a failure and release with both consumer events missing. A supplied
cancellation token is checked before and after session initialization and
before inference; cancellation releases the owned session, while an
already-running backend call is not claimed to be preemptible.

## Optional ONNX Runtime CPU

`lmsluice.onnxruntime_adapter` imports ONNX Runtime only when selected. The
adapter requires `CPUExecutionProvider`, checks inspectable engine/backend/
version/extension declarations, creates one owned session, runs caller inputs,
passes outputs to a caller validity hook and exposes idempotent `release`/
`close`. It does not register custom operator libraries or load model-supplied
code. Session allocator memory is reported `UNMEASURED` unless an independent
backend facility is supplied. ONNX I/O binding concerns graph inputs/outputs;
it does not establish zero-copy loading of compressed initializers.

When ONNX Runtime is available, the probe uses the same generated graph,
external sidecar, input and CPU provider for direct loading and lmsluice plain
materialization and requires the same validity-hook output. When it is absent,
the comparison is `NOT_RUN`, with the capability reason retained. The required
core and plain path do not change in either case.

The tracked fixture contains a valid small ONNX `Add` graph with an external
`weights.bin` initializer and an explicit non-zero range, plus config,
preprocessing, vocabulary, calibration and opaque files. It also declares
acoustic/visual/state fields as schema examples. The graph's synthetic result
`2 + 1 = 3` is an integration smoke check only.

## Evidence and claim boundary

The all-in-one probe is
[`scripts/probe-multimodal-bundle-readiness.sh`](../../scripts/probe-multimodal-bundle-readiness.sh).
It creates a fresh run directory, generates the fixture, validates the complete
identity, exercises plain delivery and all bounded negative cases, runs the
simulated consumer lifecycle, detects optional capabilities, runs the full
regression suite, retains structured records and builds/verifies an evidence
archive. Expected induced failures remain `FAIL` evidence rows with separate
`PASS` detection/cleanup rows. Optional unavailable arms remain `NOT_RUN`.

The probe reports separate `INCONCLUSIVE` rows for real multimodal quality,
SLUICE-A1 affordability and SLUICE-B1 specialized deployment. No licensed real
model, audio/image/prompt set, host/device, customer, BOM, power, thermal,
cloud, WAN or target-device evidence is supplied. No device or niche is
selected. Generated bytes prove structural transport and integrity mechanics
only; the later real-model gate requires the same graph/sidecars/input/backend
comparison plus task-level quality evidence.
