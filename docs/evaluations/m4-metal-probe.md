# Evaluation Report: Apple M4 Metal GPU Probe (`probe/metal-bench.js`)

**Date:** 2026-11-09  
**Target:** Apple M4 Mac mini (10 GPU cores, 24.0 GB Unified Memory, LPDDR5X-7500, 120.0 GB/s spec)  
**Harness:** `probe/metal-bench.js` (JavaScript for Automation via `osascript -l JavaScript`)  
**Data Artifacts:**  
- Canonical full suite: `results/probe_m4/metal-bench-20261109-125645.json` (device facts, DRAM bandwidth, FP32 FMA, dependent chain, static staircase shapes & S-sweep pool bounds)  
- Dedicated threads & contention suite: `results/probe_m4/metal-bench-20261109-152727-threads-contention.json` (latency-bound T-sweep & self-measured real CPU contention)  

---

## 1. Executive Summary & Calibration Cross-Check

`gate.py` models GPU decode throughput via two terms:
- **Compute term:** $\text{resident\_lanes} \times \text{clock} / k$, where $k$ is lane-cycles per decoded byte.
- **Bandwidth term:** $\text{peak\_dram} \times \text{achieved\_fraction} / (1 + 1/\text{expansion})$.

For Apple Silicon GPUs, `gate.py`'s row previously stated: *"lanes and clock not taken — occupancy is the question and a guess would be the whole answer"*. The RTX 5080 calibration requested in `probe/README.md` cannot be executed on an Apple Silicon host. Instead, this probe performs an **independent two-method calibration cross-check**:
1. A parallel codex session performed an empirical grid sweep measuring lmz's Metal decoder knee.
2. `metal-bench.js` independently measures the hardware occupancy staircase across identical compiled kernel footprints.

**Primary Calibration Finding:** For the compiled direct kernel shape ($T = 128, S = 19,456$ bytes compiled static footprint), the occupancy staircase displays a wave 1 capacity cap at **$G = 40$ threadgroups** jumping at $G = 42$ across 10 GPU cores, establishing **exactly 4.0 TGs per core**. This directly confirms and matches the knee identified by codex's decoder grid sweep.

**Per-Core Threadgroup Pool Bounds:** The clean staircases bound the per-core threadgroup memory pool:
- 4 TGs/core at 19,456 B ($4 \times 19,456 = 77,824$ B);
- 3 TGs/core at 20,992 B ($T=32$, $4 \times 20,992 = 83,968$ B does not fit), 23,552 B, and 24,576 B;
- 2 TGs/core at 28,672 B, 31,744 B, and 32,768 B.
- Boundary shape $S = 20,480$ B sat right on the boundary: it stepped at 4.0 TGs/core in one run ($4 \times 20,480 = 81,920$ B) and at 3.2 TGs/core in another.
- Taking only the clean staircases holding across runs bounds the hardware pool per core at:
  $$\mathbf{77,824 \le \text{pool} < 83,968\text{ B}\ (\approx 80\text{ KiB})}$$
  *(Note: Dynamic Caching on M3/M4 may make this allocation dynamically register-dependent).*

**Mechanism Note (Observation):** Codex's measured decoder $k$ is $\approx 269$ lane-cycles/byte at $\le 1$ TG/core (`lmz/docs/evaluations/m4-metal-decoder-remeasure.md`). Our dependent chain measures $\approx 68.8$ cycles per dependent threadgroup memory load at 1575 MHz. Therefore, one decode byte step costs approximately **3.9 such dependent hops** ($269 / 68.8 \approx 3.9$: LUT load + device refill load + ALU/ballot overhead).

---

## 2. Measurement Hygiene & Environment Conditions

All benchmark kernels were executed under the shared GPU lock (`~/.claude/agy-bridge/locks/m4-gpu`). Every kernel output was deterministically validated (`500/500 verified: true`).

```yaml
Hardware Model: Mac16,10 (Mac mini, late 2024)
SoC: Apple M4 (10 GPU cores, AGXAcceleratorG16G)
Physical Memory: 24.0 GB LPDDR5X-7500 (120.0 GB/s theoretical peak)
macOS Version: macOS 15.4.1 (Build 24E263)
Git Commit: 5ee399a (guarded shim on stock macOS)
Thermal Status: Normal (No thermal throttling recorded)
Parallels VM: prl_vm_app active (Windows 11 VM untouched)
GPU Idle Check: 0% median utilization over 5 seconds (samples: 0%, 0%, 0%, 0%, 2%)
Lock Protocol: atomic mkdir ~/.claude/agy-bridge/locks/m4-gpu
```

---

## 3. Device Facts Enumeration

Queried via `MTLCopyAllDevices()`, `MTLComputePipelineState`, and `ioreg -r -c IOAccelerator`:

| Parameter | Probed Value | Architectural Meaning |
| :--- | :--- | :--- |
| **Device Name** | `Apple M4` | Primary integrated GPU |
| **Registry ID** | `4294968453` | IOKit Registry entry |
| **Unified Memory** | `true` | Coherent CPU/GPU DRAM pool |
| **GPU Cores (`ioreg`)** | `10 cores` | `AGXAcceleratorG16G` |
| **Max TG Memory Length** | `32768 bytes` (32 KiB) | Hard architectural limit per threadgroup block |
| **Max Threads Per TG** | `1024` | Maximum dispatchable threads per block |
| **Thread Execution Width** | `32` | SIMD execution lane width |
| **Recommended Working Set** | `16384 MB` (16.0 GB) | Safe OS working set ceiling |
| **GPU Families Supported** | `Apple7=true, Apple8=true, Apple9=true, Mac2=true, Metal3=true` | Apple9 / M4 GPU generation |

---

## 4. DRAM Streaming Bandwidth

> **Conditions:** Mac16,10 | M4 10 GPU cores | macOS 15.4.1 | Git 5ee399a | VM Active | Thermal Normal | Lock Held  
> **Harness:** Coalesced `uint4` streaming reads with per-TG reduction, and streaming copy (`dst = src`). 7 runs per size (median, min, max reported). Verified against closed-form analytical sum.

| Buffer Size | Read Bandwidth (Median) | Read Min / Max | Read Verified | Copy Traffic Moved (2×) | Copy Bytes Copied (1×) | Copy Verified | Top Foreign Process Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8 MB** | 101.7 GB/s | 18.9 / 106.4 GB/s | `true` | 91.9 GB/s | 46.0 GB/s | `true` | none |
| **16 MB** | 93.1 GB/s | 25.7 / 97.0 GB/s | `true` | 50.8 GB/s | 25.4 GB/s | `true` | Terminal.app (1.6 ms) |
| **32 MB** | 43.4 GB/s | 36.0 / 101.9 GB/s | `true` | 62.3 GB/s | 31.2 GB/s | `true` | WindowServer (0.5 ms) |
| **64 MB** | 97.1 GB/s | 90.2 / 99.5 GB/s | `true` | 90.9 GB/s | 45.5 GB/s | `true` | Terminal.app (0.2 ms) |
| **128 MB** | 89.9 GB/s | 85.5 / 100.6 GB/s | `true` | 93.3 GB/s | 46.7 GB/s | `true` | Terminal.app (0.5 ms) |
| **256 MB** | 103.0 GB/s | 95.0 / 104.4 GB/s | `true` | 90.9 GB/s | 45.4 GB/s | `true` | WindowServer (2.3 ms) |
| **512 MB** | 103.2 GB/s | 98.0 / 103.7 GB/s | `true` | 92.5 GB/s | 46.3 GB/s | `true` | Terminal.app (6.8 ms) |
| **1024 MB** | 101.1 GB/s | 95.4 / 102.7 GB/s | `true` | 93.2 GB/s | 46.6 GB/s | `true` | WindowServer (8.9 ms) |
| **2048 MB** | 101.7 GB/s | 97.9 / 102.5 GB/s | `true` | — | — | — | WindowServer (11.5 ms) |

### Bandwidth Findings:
1. **Steady-State Rate:** Median streaming read bandwidth across sizes $\ge 256$ MB is **102.3 GB/s**, achieving **85.3%** of the 120.0 GB/s LPDDR5X-7500 spec.
2. **Buffer Scaling Note:** Small buffers exhibit lower minimum rates due to per-dispatch overhead and command buffer synchronization. An on-chip cache boundary cannot be inferred from this data.
3. **Copy Bandwidth:** Stream copy achieves **93.2 GB/s** of total memory bus traffic moved ($2 \times \text{bytes} = \text{read} + \text{write}$), representing **46.6 GB/s** of net copied payload.

---

## 5. Sustained FP32 FMA, Int32 Throughput, and DVFS Clock Measurement

> **Conditions:** Mac16,10 | M4 10 GPU cores | macOS 15.4.1 | Git 5ee399a | Sustained Duration: 2.004 s | IOReport C-Binding  
> **Methodology:** 4 independent vector FMA chains in registers, sustained $\ge 2.0$ s for DVFS ramp. Clock measured directly during compute dispatch via `/usr/lib/libIOReport.dylib` (`GPUPH` channel).

| Metric | Measured Value | Unit / Formula | Status |
| :--- | :--- | :--- | :---: |
| **Sustained Execution Time** | `2.004` | seconds | DVFS fully ramped |
| **IOReport P-State Distribution** | `P15: 47,845,358 ticks (99.53%), P8: 106,202 ticks, P3: 76,906 ticks` | 24 MHz ticks | Measured |
| **Residency-Weighted Clock ($f_{\text{meas}}$)** | **`1575.37`** | MHz (Top state: P15 = 1578 MHz) | Measured via C FFI |
| **FP32 FMA Throughput** | **`3662.5`** | GFLOP/s (10 cores × 16 TGs × 256 T × 2.8M iters × 64 FLOPs) | `verified=true` |
| **Theoretical Peak @ 1575 MHz (128 ALUs)** | `4032.9` | GFLOP/s ($10 \times 128 \times 2 \times 1.57537$) | Reference peak |
| **Issue Fraction @ 1575 MHz** | **`90.8%`** | $3662.5 / 4032.9$ | Derived issue efficiency |
| **Effective FP32 Lanes Per Core** | **`116.2`** | lanes / core ($\text{GFLOP/s} / [2 \times \text{cores} \times f_{\text{meas}}]$) | Empirically calibrated |
| **Derived Clock @ 128 ALUs (100% issue)** | **`1.431`** | GHz ($\text{GFLOP/s} / [10 \times 128 \times 2]$) | Derived nominal |
| **Int32 Throughput (rANS Op Mix)** | **`1234.9`** | Gops/s (mul, shift, mask, add, select) | `verified=true` |

---

## 6. Dependent Threadgroup-Memory Chain ($k$ Mechanism)

> **Conditions:** Mac16,10 | M4 10 GPU cores | macOS 15.4.1 | Git 5ee399a | Clock = 1575 MHz  
> **Setup:** 16 KiB pseudo-random single-cycle permutation table (4096 × uint32) in threadgroup memory; each lane executes pointer-chasing dependent loads (`idx = lut[idx]`).

### Case (a): Single Thread Latency
- **Hops:** 1,000,000 hops
- **Median Time:** 0.04369 s
- **Latency Per Hop:** **`43.692 ns / hop`** (`verified=true`)
- **Cycles Per Hop (@ 1575 MHz):** **`68.83 cycles / hop`** (Derived from measured P15 clock)

### Case (b): Single Threadgroup Throughput (128 threads)
- **Hops:** 500,000 hops per thread
- **Median Time:** 0.02362 s
- **Aggregate Hop Rate:** **`2.710 Ghops/s`** (`verified=true`)

### Case (c): Latency-Hiding Curve Across All 10 Cores
Dispatching $G = 1 \dots 16$ TGs per core ($10 \dots 160$ TGs total, 100,000 hops per thread):

| TGs Per Core | Total TGs | Aggregate Hop Rate | Execution Time | Verified |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 10 | 27.03 Ghops/s | 4.74 ms | `true` |
| **2** | 20 | 52.16 Ghops/s | 4.91 ms | `true` |
| **3** | 30 | 76.97 Ghops/s | 4.99 ms | `true` |
| **4** | 40 | **98.24 Ghops/s** | 5.21 ms | `true` |
| **6** | 60 | 88.37 Ghops/s | 8.69 ms | `true` |
| **8** | 80 | 96.11 Ghops/s | 10.65 ms | `true` |
| **10** | 100 | 112.76 Ghops/s | 11.35 ms | `true` |
| **12** | 120 | 90.37 Ghops/s | 17.00 ms | `true` |
| **14** | 140 | 105.40 Ghops/s | 17.00 ms | `true` |
| **16** | 160 | 93.22 Ghops/s | 21.97 ms | `true` |

---

## 7. Occupancy Staircase (Cross-Check of Grid Sweep Knee)

> **Conditions:** Mac16,10 | M4 10 GPU cores | macOS 15.4.1 | Git 5ee399a | GPU Lock Held  
> **Methodology:** Dispatched $G = 1 \dots 240$ threadgroups (step = 2) with fixed latency-bound dependent memory operations.

### Primary Kernel Shapes Summary:

| Kernel Shape | Threads ($T$) | TG Memory ($S$) | Step Transition ($G$) | Wave Capacity ($G_{\text{cap}}$) | Derived TGs / Core |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. lmz Direct (Compiled)** | **128** | **19,456 B** | **$G = 42$** | **40** | **4.0 TGs / core** |
| **2. lmz Direct (Source)** | 128 | 23,552 B | $G = 32$ | 30 | 3.0 TGs / core |
| **3. lmz Per-Stream** | 32 | 20,992 B | $G = 32$ | 30 | 3.0 TGs / core |
| **4. CUDA `k_shared` Dynamic** | 192 | 31,744 B | $G = 22$ | 20 | 2.0 TGs / core |
| **5. CUDA `k_shared` Full** | 192 | 32,768 B | $G = 22$ | 20 | 2.0 TGs / core |

### Raw Multi-Wave Curves (Selected $G$ points across 1 to 6 waves):

| $G$ Dispatched | Shape 1: lmz direct compiled ($T=128, S=19.5\text{K}$) | Shape 2: lmz direct source ($T=128, S=23.6\text{K}$) | Shape 3: lmz per-stream ($T=32, S=21.0\text{K}$) | Shape 4: CUDA dynamic ($T=192, S=31.7\text{K}$) | Shape 5: CUDA full ($T=192, S=32.8\text{K}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.674 ms | 1.679 ms | 1.644 ms | 1.698 ms | 1.699 ms |
| **10** | 1.682 ms | 1.684 ms | 1.644 ms | 1.700 ms | 2.101 ms |
| **20** | 1.737 ms | 1.744 ms | 1.659 ms | 1.841 ms *(Wave 1 cap)* | 2.694 ms |
| **30** | 1.829 ms | 1.837 ms *(Wave 1 cap)* | 1.693 ms *(Wave 1 cap)* | 3.457 ms | 4.750 ms |
| **40** | **2.093 ms** *(Wave 1 cap)* | 3.411 ms | 3.294 ms | 3.662 ms *(Wave 2 cap)* | 5.607 ms |
| **42** | **3.591 ms** *(Jump to Wave 2)* | 3.513 ms | 3.333 ms | 5.211 ms | 8.464 ms |
| **50** | 3.449 ms | 3.513 ms | 3.335 ms | 5.218 ms | 5.219 ms |
| **60** | 3.572 ms | 3.655 ms *(Wave 2 cap)* | 3.355 ms *(Wave 2 cap)* | 6.096 ms *(Wave 3 cap)* | 5.487 ms |
| **80** | 4.014 ms *(Wave 2 cap)* | 5.288 ms | 4.990 ms | 7.301 ms *(Wave 4 cap)* | 7.305 ms *(Wave 4 cap)* |
| **100** | 5.409 ms | 6.885 ms | 6.625 ms | 9.129 ms *(Wave 5 cap)* | 9.124 ms *(Wave 5 cap)* |
| **120** | 5.973 ms *(Wave 3 cap)* | 7.292 ms *(Wave 4 cap)* | 6.706 ms *(Wave 4 cap)* | 10.942 ms *(Wave 6 cap)* | 10.939 ms *(Wave 6 cap)* |
| **160** | 7.918 ms *(Wave 4 cap)* | 10.371 ms | 9.945 ms | 14.560 ms *(Wave 8 cap)* | 14.568 ms *(Wave 8 cap)* |
| **200** | 9.849 ms *(Wave 5 cap)* | 12.376 ms | 11.677 ms | 18.179 ms *(Wave 10 cap)* | 18.184 ms *(Wave 10 cap)* |
| **240** | 11.768 ms *(Wave 6 cap)* | 14.512 ms *(Wave 8 cap)* | 13.404 ms *(Wave 8 cap)* | 21.788 ms *(Wave 12 cap)* | 21.684 ms *(Wave 12 cap)* |

---

## 8. S Sweep & T Sweep Evaluation

> **Conditions:** Mac16,10 | M4 10 GPU cores | macOS 15.4.1 | Git 5ee399a | GPU Lock Held

### S Sweep ($T = 128$, $S = 0 \dots 32$ KiB):
- **$S = 0, 4\text{K}, 8\text{K}, 12\text{K}, 16\text{K}$:** **NOT CLEAN.** With low or zero threadgroup memory, these kernels are issue-bound rather than memory-capacity-bound, exhibiting soft transitions and anomalous $G=1$ points ($S=4\text{K}$ baseline $1.97$ ms vs $1.67$ ms). These curves are excluded from capacity bounds.
- **$S = 20,480$ B:** Step transition at $G = 34$ (Wave capacity 32 TGs $\implies 3.2$ TGs/core; in an earlier run it stepped at $G = 42$ $\implies 4.0$ TGs/core, $4 \times 20,480 = 81,920$ B).
- **$S = 24,576$ B:** Step transition at $G = 32$ (Wave capacity 30 TGs $\implies 3.0$ TGs/core; $3 \times 24,576 = 73,728$ B).
- **$S = 28,672$ B:** Step transition at $G = 22$ (Wave capacity 20 TGs $\implies 2.0$ TGs/core; $2 \times 28,672 = 57,344$ B).
- **$S = 32,768$ B:** Step transition at $G = 22$ (Wave capacity 20 TGs $\implies 2.0$ TGs/core; $2 \times 32,768 = 65,536$ B).

**Per-Core Pool Conclusion:** Taking only the clean staircases holding across runs bounds the hardware pool at:
$$\mathbf{77,824 \le \text{pool} < 83,968\text{ B}\ (\approx 80\text{ KiB})}$$

### T sweep: thread residency per core ($S = 1{,}024$ B, latency-bound chain)

> **Conditions:** Mac16,10 · Apple M4, 10 GPU cores · macOS 15.4.1 · GPU lock held · five runs on 2026-11-09, all retained in `results/probe_m4/`
> (`…151844…`, `…152429…`, `…152727…`, `…153740…`, `…154236…-threads-contention.json`).
> **Canonical run: `154236`**, taken with the final code. Runs 1–4 used earlier analysis code; their `tSweep` and contention data are genuine, but their emitted rows filled skipped sections from hard-coded defaults (see the post-run note) and must not be quoted.

**Method.** Each thread runs a dependent threadgroup-memory chain (`idx = lut[idx]`, latency-bound) in a fixed 1 KiB allocation, which cannot bind: ≥ 76 such threadgroups fit the pool. Sweep T ∈ {32 … 1024} and G, and find the first wave cap. A shape counts only if its staircase is clean (a jump, a second jump near 2× the cap, and a cap that is a multiple of the 10 cores). A single per-core thread cap C would predict floor(C / T) threadgroups per core at every T. The table gives the best such C for each run:

| Run | Clean shapes | Best single cap C | Fits | Threadgroups per core by T (clean shapes) |
|---|---:|---:|---:|---|
| `151844` | 3/10 | 1,024 | 2/3 | 192:3 · 512:2 · 1024:1 |
| `152429` | 7/10 | 992 | 5/7 | 32:31 · 96:**32** · 192:5 · 256:3 · 384:2 · 768:1 · 1024:1 |
| `152727` | 10/10 | 1,024 | 6/10 | 32:32 · 64:15 · 96:11 · 128:8 · 192:5 · 256:3 · 384:3 · 512:2 · 768:1 · 1024:1 |
| `153740` | 10/10 | 1,024 | 8/10 | 32:32 · 64:11 · 96:12 · 128:8 · 192:5 · 256:4 · 384:2 · 512:2 · 768:1 · 1024:1 |
| **`154236`** | 10/10 | 1,280 | 7/10 | 32:34 · 64:11 · 96:13 · 128:10 · 192:4 · 256:5 · 384:3 · 512:2 · 768:1 · 1024:1 |

**What this shows on this M4:**
1. **Per-core thread residency is not reproducible with this method.** The same T gives different counts between runs: T = 128 gives 8 and 10; T = 96 gives 11, 12, 13 and once 32. In the canonical run, resident threads range from 704 to 1,280 per core.
2. **No single cap is stable.** A single cap fits most clean shapes in every run, but not the same cap: 1,024 in three runs, 992 in one and 1,280 in one.
3. **What is solid:** a 1,024-thread threadgroup is resident one per core in all five runs, and Metal guarantees it is legal. About 32 threadgroups of 32 threads co-reside (31–34 per core).
4. **Emitted value.** The probe emits `max_threads_per_unit` only when one cap fits ≥ ¾ of ≥ 4 clean shapes in that run. Run `154236` did not, so it emits the **API lower bound, 1,024**, and labels it as such. For `gate.py` this binds nothing derived: shared memory binds first for every lmz kernel shape measured (4 × 128 = 512 and 2 × 192 = 384 threads per core).
5. A high-register variant (64 live uint values per thread, T = 1024) stayed at one threadgroup per core in run `152727`. That is one observation.
6. **Correction.** An earlier draft of this section reported "1,024–1,152 threads per core, bounded by clean staircases". That range came from analysis code that **kept only staircases whose threads per core fell inside [1,024, 1,152]** and then set the value to 1,024 as a literal. The manager removed that code on 2026-11-09; the data never supported the range.

---

## 9. Unified-memory contention (CPU perl streaming, self-measured rates)

> **Conditions:** as in §8. **Canonical run `154236`** (final code); the other four runs are retained and summarised below.

**Load.** Each of N perl workers copies a 64 MiB string with a forced write (`substr`), which defeats copy-on-write. Each measures its own rate in read + write bytes (Time::HiRes) and writes it to a file the probe reads back. Unlike the earlier copy-on-write version, these workers generate real traffic. **GPU:** 256 MB streaming read, with per-threadgroup sums verified.

| N workers | GPU streaming read | Measured CPU read+write | Total |
|---:|---:|---:|---:|
| 0 | 102.0 GB/s | 0.0 | 102.0 |
| 1 | 74.8 | 43.4 | 118.1 |
| 2 | 66.2 | 57.0 | 123.3 |
| 4 | 56.3 | 66.2 | 122.6 |
| 6 | 54.6 | 65.9 | 120.5 |
| 8 | 53.6 | 66.4 | 120.0 |
| 10 | 54.3 | 70.8 | 125.1 |
| 0 (repeat) | 100.1 | 0.0 | 100.1 |

**Across all five runs:**
- GPU streaming read falls **41–47%**, to 47.7–54.3 GB/s, once CPU traffic reaches 59–72 GB/s.
- The no-load baseline varies between runs (82–102 GB/s), with desktop GPU clients present.
- The closing no-load control recovers to within 2% in the canonical run. Run `152429` is the least stable, drifting 12%.

**Limitation.** The totals sit at 117–125 GB/s, at or up to 4% above the M4's 120 GB/s LPDDR5X-7500 spec. On one shared bus a total above the spec is impossible, so the CPU share is overstated: each worker times its whole lifetime, including gaps between GPU dispatches when it has the bus to itself. The probe therefore records the totals and their ratio to the GPU-only rate, and no longer passes or fails them. Earlier drafts did, against 125 and 130 GB/s constants in the code — machine constants, and too lax. Codex's decoder campaign in lmz counted only copies wholly inside each row's window and stayed ≤ 106.8 GB/s; it is the more precise measurement.

**Conclusion (this M4).** ~60–70 GB/s of CPU memory traffic roughly halves the GPU's streaming-read bandwidth, and (per lmz's decoder campaign) costs the compute-bound decoder 17–29%. The bus is shared without a hard partition. No claim about arbitration policy is made.

---

## 10. The `gate.py` device row

Since the 2026-11-09 fixes, the probe's suggested row takes every field from its own run or leaves it `null`. `gb_s` is always `null`, because a DRAM spec cannot be queried. For this M4, `gate.py` carries:

| Field | Value | Source |
|---|---|---|
| `units` | 10 | `ioreg` `gpu-core-count` |
| `shmem_cap_per_block` | 32,768 | `maxThreadgroupMemoryLength` |
| `shmem_pool_per_unit` | 77,824 | full run `125645`, clean staircases (pool < 83,968) |
| `max_threads_per_unit` | 1,024 | API lower bound (§8: residency not reproducible) |
| `clock_ghz` | 1.578 | IOReport state P15, sustained under load |
| `gb_s` | 120 | LPDDR5X-7500 spec, entered from the spec sheet (measured read ~102) |
| `backend` | metal | predicted with the Metal kernel's own constants |

---

## 11. Zero-Install Constraint Proof & Subprocess Inventory

The harness was executed in a pristine environment stripped of developer tools:
```bash
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=$HOME DEVELOPER_DIR=/nonexistent osascript -l JavaScript probe/metal-bench.js --quick
```
**Verification:** Succeeded with return code 0 and **500/500 `verified` flags `true`**, requiring zero Python, Xcode, clang, or swift tools.

### Inventory of All Shell Subprocesses:
1. `/usr/sbin/ioreg -r -c AGXDeviceUserClient -l -w0`: Samples per-process AGX GPU time.
2. `/usr/sbin/ioreg -lw0 -n pmgr`: Reads `voltage-states9` frequency table.
3. `/usr/sbin/ioreg -r -c IOAccelerator -l -w0`: Queries GPU core count, IOClass, and device utilization.
4. `/usr/sbin/sysctl -n hw.model`, `/usr/sbin/sysctl -n hw.memsize`: Queries hardware model and DRAM.
5. `/usr/bin/sw_vers -productVersion`, `/usr/bin/sw_vers -buildVersion`: Queries macOS versions.
6. `/usr/bin/pmset -g therm`: Checks thermal throttling state.
7. `/usr/bin/pgrep -fli 'prl_vm_app'`: Verifies Parallels VM status.
8. `/usr/bin/perl`: Background workload for optional contention test.
9. `/bin/kill -9`: Kills spawned Perl test PIDs only.
10. `/bin/pwd`: Resolves relative JSON output paths.
11. `git rev-parse --short HEAD`: Command Line Tools shim on stock Mac (guarded; safely falls back to `"unknown"` when toolkits are absent).

## Post-run code change (manager, 2026-11-09)

The JSON artifact `results/probe_m4/metal-bench-20261109-125645.json` was produced before one final code change.
The lmsluice manager removed a `staircase.crossCheck` block from the probe's JSON output. It held literals
(`codexKneeImpliedTgsPerCore: 4`, `match: true`, and a `|| 4` fallback), so every Mac would have reported this M4's result
and a "match". It was replaced by `staircase.decoderShapeTgsPerCore`: the decoder shape's TGs/core from *that run's*
clean staircase, or `null`. No measurement code changed.

The artifact's `crossCheck` entry is still true for this machine. The decoder shape measured 4 TGs/core here
(`poolEvaluation`: cap 40, clean), and the codex grid sweep independently implies 4. The cross-check is a fact about
this M4 and is stated in this report, not emitted by the probe.

The edited script was re-validated under `env -i … DEVELOPER_DIR=/nonexistent osascript -l JavaScript probe/metal-bench.js --quick`
on this M4: exit 0, 500/500 `verified` flags true. In `--quick` mode the coarser G grid can leave the decoder shape not clean,
so that run reported `decoderShapeTgsPerCore: null` and a wider pool bound, 70,656 ≤ pool < 83,968 B. That's the intended
honest behaviour, not a regression. The full run above is the canonical result: 77,824 ≤ pool < 83,968 B.

### Round-3 code changes (manager, 2026-11-09)

After the threads/contention round, the manager audited `probe/metal-bench.js` and found this-machine results in the code again. All of them were removed:
- **Thread filter.** The threads analysis kept only staircases with 1,024–1,152 threads per core, then set 1,024 as a literal with a fixed note. It now uses every clean staircase and emits a cap only if one fits ≥ ¾ of them; otherwise it emits the queried API lower bound, labelled.
- **Section defaults.** Skipped sections defaulted to this M4's results: bandwidth 102.3 GB/s, clock 1575.4 MHz, issue 90.8%, 3662.5 GFLOP/s, pool 77,824–83,968 B, and a 120 GB/s spec. All now default to `null`. Even full runs had kept the pool literal whenever the staircases disagreed, and always printed "Hardware Spec (LPDDR5X-7500): 120.0 GB/s".
- **Row and platform.** The row's name and the platform string said "Apple M4" literally; both now come from the Metal device and `hw.model`.
- **Contention bound.** The contention check passed totals against 125 and 130 GB/s literals; it now records totals and their ratio to the GPU-only rate.
- **Bridged types.** JXA returns NSUInteger properties as strings, so the cap and thread values came out as `"32768"` and `"1024"`; both are now coerced to numbers.

**Validation:** on this M4, under `env -i … DEVELOPER_DIR=/nonexistent`, a sections-only run emitted `null` for every field it didn't measure (6/6 checks verified), a full `--quick` run computed every field (606/606 verified), and the canonical threads/contention run `154236` exited 0.
