# SLUICE-A1 affordability and resource evaluation

This is an engineering affordability evaluation for the frozen synthetic
workload. It does not price a product. The run compares a normal mmap loader,
lmsluice plain transport and the stdlib-coded route on the executor's WSL2
host. The complete raw evidence is retained in the run directory reported by
the probe; the compact campaign row is `A1-summary` in `results.jsonl`.

The final local campaign used 30 interleaved first-fetch and warm-reuse
samples: five repetitions for each of three arms in each scenario. It also
ran a 30-repetition warm local tail for the normal loader and lmsluice plain
arms, plus synthetic scheduler interference and a resident control. Every
successful arm reproduced the frozen full-file SHA-256
`067ffed1b1199b4b01fb53c78ec66e10a2d70ed9471a280e84984d26e8aaa61c`.

Observed medians from the final WSL2 run were:

| scenario | normal loader | lmsluice plain | stdlib-coded |
|---|---:|---:|---:|
| first fetch | 0.001227 s | 0.001929 s | 0.001998 s |
| warm reuse | 0.001526 s | 0.002481 s | 0.005787 s |

The five-sample arms do not support an empirical p95. In the 30-sample warm
tail, the normal loader median/p95 were 0.001263/0.001657 s and lmsluice plain
median/p95 were 0.001832/0.006325 s. Neither lmsluice arm beat the normal loader
in the ten paired first-fetch/warm-reuse comparisons. These are small local synthetic timings,
not a claim that lmsluice is slower or faster on a target device.

The run had 16 logical CPUs, about 20.0 GiB available memory at setup and about
45.4 GB free disk. The fixture was 145,012 bytes; the process plan stayed far
below the 1 GiB working-set and 2 GiB run-storage limits. Host memory samples
were collected per process. Allocator, device and energy costs were not
measured. CUDA was detectable through lmsluice, but torch was absent, so the
CUDA alignment case was `NOT_RUN`.

The first-fetch cache operation was attempted locally, but WSL2 lower-cache
state is not fully observable. The campaign records that limitation in each
row instead of calling the result a verified cold-storage measurement. No
portable device, power envelope, memory price, storage price, production
volume, host fleet, customer, or BOM data was supplied. Consequently A1 is
`INCONCLUSIVE` for affordability and resource feasibility. It supports a
bounded technical statement: the current core can preserve bytes and expose
startup/resource evidence under the campaign limits. It cannot support a
lower-BOM or cheaper-device decision.

The next A1 decision requires a named target and its allowed workload,
available memory, power or thermal budget, storage/link conditions, unit cost
inputs and an approved host/customer evidence source. Advanced optimization
work remains held until that brief is settled.
