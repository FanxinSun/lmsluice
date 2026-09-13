# Portable-device readiness plan and evidence gates

This work tests whether lmsluice remains useful if a small portable AI device
combines voice and vision workloads with tight memory, power and connectivity
limits. The strategic trigger is a reported possibility of an OpenAI portable
voice/vision device. No device specification, benchmark, price, customer or
competitor result is assumed here. Any such input remains a hypothesis until a
real target supplies an allowed workload and measurement environment.

The engineering question is bounded: can the existing standard-library core
move byte-identical language, speech and vision artifacts under constrained
links and memory, while reporting when bytes arrive, when a tensor is first
available, when a consumer first does useful work, and when it is ready? The
market questions are evaluated separately. SLUICE-A1 asks whether the measured
workflow changes an affordability or resource constraint. SLUICE-B1 asks which
specialized deployment workflow, if any, has a credible technical wedge. A
positive result in one does not select the other.

## Frozen contract

[`docs/evaluations/workload.json`](../evaluations/workload.json) is the source
of truth. The generated artifact has stable bytes, named modality roles and a
memory-edge tensor. Its full-file SHA-256 is checked before every campaign.
Consumers check the complete file or exact selected tensor payloads. Hash
agreement proves preservation of this fixture; it is not a model-quality or
voice-quality score.

For MM-SLUICE-01 the canonical bundle payload uses schema `{"major": 1,
"minor": 0}`, bundle `id`/`version`, `source` and `license`, an entry-point
object, `consumer.required_operators` and `extensions`, `preprocess`,
`temporal_state`, `resources` and `evaluation`. Entries carry stable ids,
decoded and stored measurements, SHA-256 digests and dependency ids, paths and
ranges. The plain directory format retains its top-level `bundle` envelope for
compatibility; the optional lmz adapter removes that envelope and translates
legacy aliases before calling lmz's public bundle API. It snapshots only the
declared files, so a source `bundle.json` is never forwarded as an archive
artifact.

The lmz route returns the same normalized lmsluice result shape as the plain
route and retains the supplied observer. Archive decoded, stored and
materialized accounting is reported only when lmz provides it. lmz does not
provide a cooperative in-call cancellation hook, so cancellation is checked
before and after each public inventory, validation, creation or materialization
call; no backend preemption is claimed and transferred bytes remain unavailable
rather than being inferred from archive size. Materialization must match the
strict pre-call inventory entry set, identities and digests before lmsluice
stages or publishes it. The default probe does not import the sibling lmz
checkout. A separate run may pass `--lmz-root` for read-only interoperability
evidence against the accepted clean `main` commit.

Plain and lmz publication stage through a directory descriptor opened with
no-follow path components. The descriptor remains held through no-replace
publication and ownership-based cleanup; a destination-parent symlink or
replacement causes a structured failure without following or deleting the
replacement. This safety evidence is bounded to POSIX systems with the
required descriptor and `renameat2` primitives; unsupported platforms fail
closed.

The coded archive is generated from the same plaintext. Its measured codec
metadata makes the archive container hash run dependent, so the campaign
records its bytes and hash as evidence without freezing that incidental value.
Missing optional codec, CUDA, torch or crypto support is recorded as an
availability result with the reason. It is never silently replaced by an
unrelated implementation.

## P0: contract and capability inventory

The campaign generates the fixture in a fresh output directory, checks the
stable hash, byte count and all modality roles, and writes a capability
manifest. The manifest records Python/platform, CPU and disk headroom,
optional modules, codec backend, CUDA availability, cache assumptions and the
resource plan. The default fixture is far below the planned 128 MiB artifact,
2 GiB run-storage and 1 GiB process-working-set limits.

P0 passes when the same contract is regenerated and the campaign can state
which optional measurements are unavailable. It does not turn an unavailable
portable target into a simulated target claim.

## P1: additive readiness and resource reporting

`lmsluice.observability.ReadinessRecord` is opt in. Existing report fields and
loader behavior remain available without an observer. The record uses one
`time.monotonic_ns` origin and preserves missing events as `null`. It carries
route intent and actual route, fallback, codec/cipher and credential mode;
logical, coded, transferred, fetched, placed and repeated bytes; requested and
covered spans; tensor-window and retained-output bytes; worker, inflight,
retry, wall and CPU fields; and failure type/message with credential-bearing
query values removed.

The bounded sampler measures the current process's RSS/PSS/HWM where the host
exposes them. It does not sum child pages, guess allocator or device memory,
or invent energy data. Those fields stay explicitly `UNMEASURED`. The caller
marks `consumer_first_useful` and `consumer_ready`, so transport completion is
never presented as application readiness.

P1 passes only when successful and failing observer cases retain records,
missing consumer events are visible, transport counters do not double count,
and fetch/place workers have left after both normal completion and failure.

## P2: bounded engineering experiments

The harness covers four groups:

* A shared 96 KiB/s limiter is applied to plain and coded local sources. One
  and two fetch workers are compared under the same aggregate cap. The result
  reports observed rate, latency, stage stalls and worker cleanup; it does not
  treat thread count as link capacity.
* A 32 KiB stream target exercises normal windows, the 96 KiB memory-edge
  tensor, noncontiguous selected spans, retained consumer output and consumer
  departure. Host resource peaks and coverage are retained. CUDA alignment is
  `NOT_RUN` unless an actual target and approved adapter are available.
* Reset, truncation, corrupt archive, too-small destination and unwritable
  destination cases are induced locally. Wrong-key and authenticated
  ciphertext tamper cases run only with the existing crypto backend. The
  induced row remains a `FAIL` evidence row; a separate detection row carries
  the engineering verdict. An ordinary retry from zero records repeated work
  and does not claim durable resume.
* The existing regression suite remains the compatibility gate. No inference
  framework is added to the core.

P2 passes when expected failures propagate, failure records classify them,
output hashes remain byte exact on successful paths, constrained memory cases
complete, and no transport workers leak. A reproduced in-scope defect is
fixed before the final campaign; otherwise the report says that no existing
core defect was reproduced rather than inventing one.

## P3: SLUICE-A1 affordability and resource evaluation

A1 compares a normal mmap loader, lmsluice plain transport and, when its
backend exists, the stdlib-coded route. It runs first-fetch and warm-reuse
arms with interleaved order, five repetitions per arm, a 30-repetition local
warm tail for empirical variability, synthetic scheduler interference and a
resident control. The same frozen output hash is required for every arm.

The report separates observed startup/transport/resource behavior from any
bill-of-materials conclusion. It includes wins and losses, medians, ranges,
an empirical p95 only when the sample count supports it, cache-control method,
peak process memory, optional backend availability and the reason a route may
not have run. Without a named device, power budget, memory ceiling, storage
cost, production volume and customer/host evidence, a lower BOM or device
affordability claim remains `INCONCLUSIVE`.

## P4: SLUICE-B1 specialized deployment evaluation

B1 compares three independent candidate workflows:

1. restricted/private artifact distribution through the existing local sealed
   mechanism when crypto and coded fixtures are available;
2. a bandwidth-constrained population using a local range server and a local
   no-range fallback, with output hashes checked;
3. switching language and specialist subsets while a synthetic periodic task
   runs, with interference recorded but no voice deadline inferred.

For each candidate the report records the consumer hypothesis, buyer
hypothesis, trigger, source/destination, alternative, observed technical
value, integration and support effort, evidence class, recommendation and
customer-evidence status. Local loopback and synthetic tasks are mechanism
evidence only. No outreach, customer claim, real authenticated cloud store or
forced niche selection is performed.

## P5: reproducible delivery and review

[`scripts/probe-device-readiness.sh`](../../scripts/probe-device-readiness.sh)
is the one user-facing launch. It resolves the repository relative to itself,
creates a fresh run directory, checks Python and tracked inputs, runs the full
campaign and regression, retains logs and result files after failures, prints
the exact result location and returns the campaign status. The script is
portable Bash and has no dependency installation or manual fixture copy.

The accepted branch contains the script, harness, deterministic fixture,
observability code, tests and evidence schema/sample. Final delivery verifies
shell syntax, a clean tracked-file export from another working directory, the
exact remote branch/ref and the remote tree. The default run reports actual
target/backend limits and keeps unavailable or inconclusive evidence visible.

Advanced optimization candidates, durable resume, a real WAN/device run,
device-memory alignment and customer or production-cost validation remain
gates for a later settled manager brief. They are not implemented merely
because an older roadmap lists them.
