# Engineering fixes and remaining gaps

The correction campaign is identified by the exact source commit in its
`manifest.json`; the tracked [`final-evidence.json`](final-evidence.json)
resolves that commit in a Git archive. It completes with engineering status
`PASS` when required failures are empty. Its exact row counts, every positive,
negative, induced-failure, `NOT_RUN`, unavailable and inconclusive result, and
archive hashes are in the required correction report and retained run output.
Induced failures are expected inputs to the fault campaign, not hidden
regressions. Each has a separate detection row, and the complete traceback
remains under the run's `errors/` directory.

During implementation and focused tests, these in-scope issues were found and
fixed before the final run:

* `Model.load()` now emits `first_tensor` after the requested bytes have been
  gathered, so a full-load record can distinguish transfer completion from
  consumer use.
* Streaming transport no longer labels the first window as a complete request.
  It suppresses per-window completion and emits `transfer_complete` only after
  the consumer exhausts the stream; early consumer departure leaves the event
  missing while worker cleanup is still checked.
* Nested readiness event details now survive JSON serialization to the bounded
  detail depth. Finalizing a record after transport observation uses absolute
  counters, preventing duplicate fetched/transferred/placed bytes.
* The campaign's truncation injector now raises a terminal short-range error,
  the local HTTP handler resolves paths consistently, and fault classification
  accepts the stdlib decompressor's actual error type. These fixes keep failure
  evidence meaningful and do not change the default loader path.
* The portable sampler treats Python's `resource` module as optional, allowing
  native Windows to retain `UNMEASURED` resource fields instead of failing at
  import. Optional coded, crypto, torch and CUDA cases retain explicit
  availability results.

The final run reproduced byte-identical output for plain, coded, private sealed,
HTTP range, HTTP no-range and selected specialist paths. Reset, truncation,
corruption, destination bounds, unwritable destination, wrong key and tampered
ciphertext all propagated as expected. A retry after an injected reset starts
from zero and records 145,012 repeated bytes; durable resume is not claimed.

The correction additionally measures event-only and sampled observer overhead
against the same uninstrumented path with 30 balanced pairs each; reports
five-sample A1 arm medians/ranges/paired ratios without p95; retains p95 only
for the 30-sample tail; separates application process startup from loader
readiness across five child launches; surfaces the local signature/body-checking
multipart failure and successful abort; and exercises changed-artifact cache
refusal, refresh, range/no-range and restart-from-zero behavior. These are
bounded local engineering facts and do not change the strategic evidence
boundary.

Remaining gaps have clear evidence boundaries:

| gap | current evidence | next bounded work |
|---|---|---|
| Real voice/vision usefulness and readiness | Synthetic tensor hashes and caller event marks only | Name a runtime, input/output contract and first-useful acceptance signal |
| Target-device memory alignment and device counters | CUDA driver detectable, torch absent; alignment `NOT_RUN` | Run an approved target campaign with its adapter and memory budget |
| True cold storage on this WSL2 host | Cache drop attempted; lower-cache state is not fully observable | Use a controllable storage target and record its cache state |
| WAN/intermittent link and authenticated hosted store | Local bounded limiter plus loopback range/no-range; real credentials unavailable | Supply an approved target endpoint and failure budget |
| Affordability | Local startup/resource timings only | Supply power, thermal, memory, storage, volume and unit-cost inputs |
| Niche/customer fit | Three technical workflow hypotheses; no outreach authorized | Name a buyer/consumer and obtain approved acceptance evidence |
| Durable resume and changed-artifact policy | Ordinary retry from zero only | Settle a separate resume and cache-invalidation brief |

The advanced optimization candidates listed in older roadmap material remain
held. This branch does not add an inference-framework dependency, claim a
portable-device benchmark, select a niche, or alter sibling projects.
