# lmsluice canonical master plan

## Current authority and intake — 2026-10-03

This section is the **one authoritative project execution order**, shared by
all agents. The September 23 two-step order preserved below is **retired as a
project-wide order**, not deleted. Its PaP ownership matrix, design constraints
and future-fixture specification remain incorporated requirements for
SLUICE-PAP-0/1. References to strategy Section 0 as the governing plan are
retired; that document is a retained requirements/rationale annex.

The user's instruction, “intake all handed over update docs”, authorizes this
documentation reconciliation. It does not start any held implementation,
experiment, download, target/cloud work, history integration or publication.
The [intake record](../docs/handover-intake.md) identifies superseding handovers
and evidence boundaries. Agent review bookkeeping is separate from this plan.

## Current authoritative execution order

Rows are ordered; the two independent assessment tracks in row 4 may proceed
separately when individually authorized. A blocked optional track does not
block the other. Rows define leaf-level bounded scopes, not automatic dispatch.
Correctness and ownership constraints apply throughout, not just at row 2.
SLUICE-USE-0 is the shared technical qualification scope when needed, not a
prerequisite to defining or assessing A1/B1 independently. An adequate existing
route result may satisfy it without a new campaign. The order does not defer
necessary correctness repairs pending product-strategy success.

| Order | Stable step | Current status and completion gate |
|---|---|---|
| 1 | **SLUICE-PAP-0** | PaP mapping accepted September 25; local cross-document correction completed by this intake. Overall **IN_PROGRESS**: publication/history integration and any renewed upstream assessment remain pending. Preserve the original completion conditions below. |
| 2 | **SLUICE-CORRECTNESS-0** | **ONGOING constraint; no new run assigned.** Preserve accepted P0–P5/cache and MM-SLUICE-01 behavior, plain/stdlib compatibility, existing tensor integrations, safe ownership, five invariants and live boundary loans. A future change must validate its affected contracts and disclose skips/failures. |
| 3 | **SLUICE-USE-0** | **HELD, scope/input approval required.** Qualify one useful real delivery-to-consumer route using adequate existing evidence first. Freeze normal/plain/optional-coded comparisons, artifact closure, inputs, output acceptance, engine/backend, target, cache state, copies/synchronization, limits and stop conditions before execution. Actual-engine opening/output remains unproved by MM-SLUICE-01. |
| 4 | **SLUICE-A1** and **SLUICE-B1** | **Independent, INCONCLUSIVE product assessments.** Synthetic engineering collection is accepted, not a device/customer verdict. Each requires its own named inputs and bounded authorization; neither requires the other's success. Requirements are retained below and in strategy Section 0. |
| 5 | **SLUICE-PAP-1** | **NOT_STARTED / HELD.** Preserve the complete original generated-fixture definition below. Requires fixed C06 descriptor and PAP-1/PAP-2 consumer contracts plus a new bounded approval. Broad PaP implementation additionally waits for a concrete device/workflow; no device is selected here. |
| 6 | **SLUICE-OPT-0** | **HELD candidate pool, not an implementation queue.** Select only a measured limiting gap after the applicable route/value assessment. Order-aware delivery, write overlap/delta, multi-model reuse, cache policy, durable resume and device work are separate possible scopes, not commitments to implement all of them. |

### Incorporated live requirements and historical phase disposition

All original explanations and decisions in [strategy](../docs/strategy.md),
[boundary](../docs/boundary.md), the [engineering gap record](../docs/evaluations/engineering-gaps.md)
and the original PaP text below are retained as specification/rationale inputs.
Their numbered lists describe requirements or historical sequences; they do
not schedule work independently of the order above.

| Retained content | Canonical disposition |
|---|---|
| Strategy Section 0, MM-SLUICE-01; readiness P0–P5 | **ACCEPTED mechanics**, not future work to rebuild. Maintenance gate: SLUICE-CORRECTNESS-0; actual-consumer gap: SLUICE-USE-0. |
| Strategy Section 0, SLUICE-A1 | Preserve normal/plain/coded same-workload comparison; first useful output versus complete readiness, bytes/storage, peak memory, CPU contention, latency tails, energy/thermal and low-resource/unified-memory accounting. Separate BOM reduction from headroom and operating cost; include host/service cost and broad capability/quality. Target, costs and real quality remain unavailable. |
| Strategy Section 0, SLUICE-B1 | Preserve independent candidate comparison (private/restricted delivery, constrained distribution, model switching); identify buyer, trigger, alternatives, integration/support burden, security/recovery and willingness-to-adopt evidence. Compare at least two credible workflows before selecting a niche. A mechanism PASS or encryption feature is not customer validation. |
| Strategy next-cycle validation and correctness; boundary I1–I5 | SLUICE-CORRECTNESS-0 throughout. Keep full failures/detection, optional NOT_RUN, unavailable metrics, no-false-ready and clean ownership evidence. No separate old next-cycle order remains active. |
| Historical Phase A0 and Phase A | Standard-codec adapter and drop-in tensor/integration surfaces already implemented; maintain compatibility, do not restart. Historical consumer results are conditional on recorded versions/workloads. |
| Historical Phase B | Opt-in cache implemented; accepted fast-path correction reads zero source-content bytes on a valid hit. Stat identity is not publisher authentication. Sampled cache codec/ratio versus actual writer choice remains a candidate gap, not declared repaired by this intake. |
| Historical Phase 0 | Scaling/slow-link measurements retained as dated evidence. SLUICE-USE-0 reuses adequate comparisons; no generic rerun or expansion to an unsupported device table. |
| Historical Phase 0.5 | Object reads/uploads implemented, anonymous real reads recorded, signing/local failure checks accepted. Signed hosted-store/WAN evidence remains open; only an approved endpoint/credential/resource scope may close it. |
| Historical Phases 1–3 | Consumer-order streaming, overlap/delta write paths and model-family/switching reuse remain SLUICE-OPT-0 candidates. File-order streaming is not consumer-order delivery; restart-from-zero is not durable resume. |
| Historical Phase 4; probe GPU proposals | Device decode/placement optimization conditional on demonstrated benefit and a validated backend. GDS stays shelved; GPU calibration and live loans are not silently resolved. |
| Original intake residual register | Missing historical fixtures/logs and broken cache symlinks remain recorded gaps, not downloads or repairs assigned here. Warning cleanup belongs to bounded correctness work; parent gitlink integration remains separate, unapproved repository coordination. Old large-model campaigns stay closed. |
| September 20 PaP follow-ons | Preserve conditional engine comparison, range/resume/prefetch/cache-source and identity-scoped admission/GC ideas under SLUICE-USE-0 / SLUICE-OPT-0. G5/use-case need and a new owner-approved brief are required; no global aggressive dedup service or control transport is assigned. |

### Accepted baseline and remaining evidence gates

P0–P5 and cache correction were accepted at `809a7172f03288e4fe9496545d3f35a66cb5ed65`:
195 total tests = 157 passes + 38 skips. MM-SLUICE-01 was accepted at
`c5927d9f22474bfeccfe0bb27e4ad430cedbefd5`: clean regression 207 total = 169
passes + 38 skips. Both commits are ancestors of the inspected local HEAD.
These are attributed historical results, not tests rerun during this intake.

MM-SLUICE-01's real lmz API interop on generated artifacts passed separately;
the default lmz and actual ONNX Runtime arms were NOT_RUN. Consumer-valid-output
was simulated lifecycle evidence. Trained speech/vision/language quality,
target power/thermal/memory, affordability and specialist value remain open.
No universal route threshold or OpenAI-device comparison is established.

CPU/CUDA tensor placement exists; the inspected torch adapter rejects MPS.
Host loading followed by application-owned MPS transfer is a distinct route,
not direct Metal placement, zero-copy interoperability or CUDA translation.
Bundle publication is accepted only for tested POSIX/WSL safe primitives and
fails closed elsewhere. Historical Mac host measurements do not certify a
Mac bundle or MPS route. LMZ's existing real-model/corpus evidence is credited
without being repurposed as lmsluice application benefit.

Runtime/OS retains inference residency, temporal state, deadlines, safety and
permissions; vram retains training residency; lmz retains formats/decoders.
C06 semantic trust, grants, licensing, compatibility and release acceptance
stay external. No new dependency on siblings or custom silicon is imposed.
Portfolio time/stream limits were proposals, not policy; resource/effort caps
remain unset until a concrete approved unit. The observer track is not canceled.

### Documentation completion and publication boundary

This intake resolves the September 25 authority conflict locally and records
coverage separately. It does not claim Supreme reacceptance or publication.
Inspected local `master` remains one commit ahead and one behind the existing
`origin/master` tracking ref. No fresh fetch, merge, rebase, commit, push, tag or
gitlink update is part of this documentation task. Existing report-return and
publication history is retained, not reset by introducing these status notes.

## Retired September 23 order and preserved PaP specification

The original text below is preserved verbatim. Its two-step-only authority,
IN_PROGRESS snapshot and old checklist wording are **retired as current
project-wide authority**; use the current order above. Its detailed contract,
exclusions, fixture definition and pending publication condition remain live
requirements of the named current steps, not a second executable sequence.

<details>
<summary>Original September 23 plan — retained history and incorporated requirements</summary>

Status: planning baseline for PaP intake, consolidated 2026-09-23.

This is lmsluice's single canonical master plan. Its authoritative execution
order is the two local steps below. The PaP-AI umbrella plan remains at
`/home/rog/business/PaP-AI/plan/PLAN.md`; its labels are dependencies and
ownership inputs, not a second lmsluice execution order. No lmsluice subplan is
needed for these leaf-level steps.

The existing [boundary record](../docs/boundary.md) and
[strategy record](../docs/strategy.md) remain historical and rationale inputs;
they are not execution plans and are preserved byte-for-byte by this packet.

## Authoritative execution order

1. **SLUICE-PAP-0 — PaP package-delivery mapping and ownership alignment**
   (upstream label: `PAP-SLUICE-01`).

   **Status:** IN_PROGRESS as this plan-only packet.

   This step completes when this canonical plan has been validated, the one
   scoped documentation commit has been created and published when the remote
   fast-forward condition permits it, and the complete final evidence has been
   returned to the originating lmsluice Router. It maps PaP package delivery
   onto the existing lmsluice boundary without assigning PaP control or engine
   ownership to this library.

   **Lifecycle and ownership mapping**

   | Phase | lmsluice responsibility | External responsibility / claim boundary |
   |---|---|---|
   | source | Select and open the authorized artifact source and transport route; retain a first-class plain route. | Source authorization, release policy and semantic trust remain external. |
   | inventory | Orchestrate complete entry/dependency inventory through the selected provider. | lmz is optional and owns only its format/structural contract. |
   | validate | Enforce declared byte identity, closure, safe paths, limits and provider-result normalization that lmsluice exposes. | C06 semantic authenticity, signature/trust, grants, licence approval, base/engine compatibility and evaluation acceptance stay with PaP skills/runtime policy. |
   | materialize | Own bounded staging, safe destination placement, integrity checks, cleanup and truthful transfer/materialization accounting. | OS storage policy and runtime residency are not inferred from placement. |
   | initialize | Expose the materialized result and observe a supported adapter boundary only. | Engine/runtime owns loading, session/state allocation, compatibility and initialization success. |
   | first-valid-output | Record a supported lifecycle observation without false readiness. | Engine/evaluator owns output validity, task quality and semantic acceptance. |
   | release | Release lmsluice-owned transport, staging and placement resources and retain truthful records. | Consumer/runtime releases engine sessions/state; PaP control, safety and permissions remain outside lmsluice. |

   The artifact contract is mapped to `PAP-C06`. C07 lifecycle timestamps and
   resource fields are explicit observations or unknowns; they do not transfer
   inference-engine ownership to lmsluice. The generated package descriptor is
   dependent on umbrella `PAP-1`, and the generated consumer lifecycle contract
   is dependent on umbrella `PAP-2`.

   lmsluice does not own PaP control transport, sensor/control messages, leases,
   permissions, safety fencing, engine execution, evaluation policy, training
   residency or release authority. It owns delivery and the supported adapter
   boundary only.

   **Retained design decisions**

   - Plain, codec-independent, stdlib delivery is mandatory.
   - Optional providers, including lmz, must not become a core dependency.
   - Logical execution blocks, codec chunks, bundle entries, runtime
     allocations and OS pages are distinct and must not be conflated.
   - Transfer, cache and placement facts remain separate from initialization,
     engine state, first-valid-output and task-quality facts.
   - Cancellation is claimed only at supported provider or adapter boundaries;
     a host timeout does not prove interruption of a blocking external call.
   - Byte, timing and resource counters that are not exposed remain unavailable
     or null with a reason; they are never estimated into a PASS.
   - Owned staging, safe publication and cleanup, caller ceilings and
     no-false-ready behavior remain mandatory.

   **Smallest later generated PaP fixture definition**

   This step defines the later fixture without implementing it. It must be
   gated on fixed generated PaP C06 descriptor semantics and the generated
   consumer contract for umbrella `PAP-1` and `PAP-2`.

   - Inputs and dependencies: a fixed generated C06-compatible package
     descriptor; complete generated artifact closure; a deterministic generated
     consumer contract; a first-class plain route; and an optional lmz route
     only when that provider is available.
   - Positive path: source → inventory → validate → materialize → initialize →
     first-valid-output → release, with separate phase timestamps and owned
     resource lifetimes.
   - Required negative cases: missing or extra dependency; digest or base
     mismatch; unsafe path, symlink or destination-parent race; source mutation
     or interruption; allocation ceiling; cancellation at each actually
     supported boundary; provider failure; engine initialization failure; and
     rejected output, each with no false readiness.
   - Resource controls fixed before execution: entry-count, per-entry and total
     byte limits; in-flight, staging and destination allocation ceilings;
     deadline and cancellation boundaries; and separate engine counters when
     exposed. Unavailable counters remain unavailable with reasons.
   - Evidence outputs: a complete result ledger with positive, induced-failure
     and separate detection rows, skipped or NOT_RUN rows and inconclusive rows;
     exact input, source, manifest, output and environment identities; lifecycle
     ordering; separate transfer, materialization and engine accounting;
     cleanup and ownership evidence; regression output; and a readable hashed
     evidence archive and index.
   - Claim limits: the fixture cannot establish real ASR, TTS, vision or
     language quality; WAN or cloud behavior; target power, thermal or memory
     behavior; affordability; specialist demand; release readiness; or safety.

2. **SLUICE-PAP-1 — smallest generated PaP delivery/lifecycle fixture**

   **Status:** NOT_STARTED.

   This future step requires a new bounded user-approved packet after the C06
   descriptor semantics and the `PAP-1`/`PAP-2` generated-consumer contract are
   fixed. It must not be implemented by this plan. Its definition is the
   fixture specification under the preceding step; the existence of this plan
   does not imply that `PAP-1`, `PAP-2`, `SKILL-0`, MM2 or any runtime work is
   complete.

## Accepted evidence and boundaries

MM-SLUICE-01 is accepted generated-bundle and optional-lmz interoperability
evidence at commit `c5927d9f22474bfeccfe0bb27e4ad430cedbefd5`. The acceptance
record is at `/home/rog/.codex/handover/2026-09-14-lmsluice-mm-sluice-01.ACCEPTANCE.md`
and the complete correction report is at
`/home/rog/.codex/handover/2026-09-14-lmsluice-mm-sluice-01-acceptance-corrections.REPORT.md`.
Those records establish complete plain delivery, safe owned placement, optional
actual-lmz interoperability and bounded consumer lifecycle mechanics only. They
do not establish real ASR/TTS/vision/language quality, WAN or cloud behavior,
target-device power/thermal/memory behavior, affordability, specialist demand
or PaP readiness. Historical evidence is referenced, not reconstructed or
rerun by this planning packet.

The PaP-AI planning inputs are read-only references for this plan:

- `/home/rog/business/PaP-AI/plan/PLAN.md`, published at
  `05d3a44fd023e00156908be7e2d57d999f32c843`;
- `/home/rog/business/PaP-AI/docs/EXISTING-PROJECTS.md` for ownership and
  project boundaries; and
- `/home/rog/business/PaP-AI/docs/ARCHITECTURE.md` for C06/C07 definitions.

No source, API, test, fixture, probe, dependency, environment, model, engine,
cloud, target, sensor, control, permission, release, MM2, merge, tag, release
or gitlink work is authorized by this plan. The dirty boundary and strategy
records, the clean README, all accepted MM-SLUICE evidence, sibling projects,
environments and credentials remain outside this packet.

## Plan acceptance checklist

The canonical plan is accepted only when its one execution order remains stable,
its two local IDs are unique order entries, the lifecycle mapping and ownership
exclusions above are present, the future fixture remains gated and unimplemented,
all repository-local Markdown links resolve, and the scoped validation,
commit/publication and final evidence return are recorded truthfully. A remote
advance, authentication failure or any need for merge or rebase blocks
publication rather than authorizing history integration.

</details>
