# lmsluice canonical master plan

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
