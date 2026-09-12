# SLUICE-A1 affordability and resource evaluation

This is an engineering affordability evaluation for the frozen synthetic
workload. It does not price a product. The run compares a normal mmap loader,
lmsluice plain transport and the stdlib-coded route on the executor's WSL2
host. The current correction campaign is identified by the exact
`source_commit` in `manifest.json` and the export-substituted marker in
[`final-evidence.json`](final-evidence.json). Its complete raw evidence is
retained in the run directory and deterministic archive reported by the probe;
the compact campaign row is `A1-summary` in `results.jsonl`. The earlier
correction return is historical. The final cache fast-path review supersedes
it; the current complete return is
`/home/rog/.codex/handover/2026-09-13-lmsluice-cache-fastpath-final-review.REPORT.md`.

The measurements below are the dated initial campaign sample retained for
context. They are superseded by the correction campaign's five-sample arm
summaries, 30-sample tail and raw rows. They do not identify a product price.

The reference clean tracked-export campaign
(`/tmp/lmsluice-device-readiness.WKXRxT`) used 30 interleaved first-fetch and
warm-reuse
samples: five repetitions for each of three arms in each scenario. It also
ran a 30-repetition warm local tail for the normal loader and lmsluice plain
arms, plus synthetic scheduler interference and a resident control. Every
successful arm reproduced the frozen full-file SHA-256
`067ffed1b1199b4b01fb53c78ec66e10a2d70ed9471a280e84984d26e8aaa61c`.

Observed medians from that reference WSL2 run were:

| scenario | normal loader | lmsluice plain | stdlib-coded |
|---|---:|---:|---:|
| first fetch | 0.001676 s | 0.001951 s | 0.002038 s |
| warm reuse | 0.001197 s | 0.001650 s | 0.001841 s |

The five-sample arms do not support an empirical p95. In the 30-sample warm
tail, the normal loader median/p95 were 0.001196/0.001265 s and lmsluice plain
median/p95 were 0.001697/0.001917 s. Plain transport beat the normal loader in
two paired samples and coded transport in one of the ten comparisons. These are small local synthetic timings,
not a claim that lmsluice is slower or faster on a target device.

That clean tracked-export run had 16 logical CPUs, about 16.8 GiB available memory
at setup and about 12.2 GB free disk. The fixture was 145,012 bytes; the process plan stayed far
below the 1 GiB working-set and 2 GiB run-storage limits. Host memory samples
were collected per process. Allocator, device and energy costs were not
measured. CUDA was detectable through lmsluice, but torch was absent, so the
CUDA alignment case was `NOT_RUN`.

The correction campaign attempts cache control on the actual source path for
each arm and records `cold_attempt` or `warm_or_layered` per sample. WSL2
lower-cache state is not fully observable, so no row is treated as a verified
cold-storage measurement. It also retains an optional `lmz-coded` arm when the
configured interpreter can import lmz, or an explicit per-arm `NOT_RUN` row
otherwise. No portable device, power envelope, memory price, storage price,
production volume, host fleet, customer, or BOM data was supplied. Consequently A1 is
`INCONCLUSIVE` for affordability and resource feasibility. It supports a
bounded technical statement: the current core can preserve bytes and expose
startup/resource evidence under the campaign limits. It cannot support a
lower-BOM or cheaper-device decision.

The next A1 decision requires a named target and its allowed workload,
available memory, power or thermal budget, storage/link conditions, unit cost
inputs and an approved host/customer evidence source. Advanced optimization
work remains held until that brief is settled.
