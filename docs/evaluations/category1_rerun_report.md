# Category 1 Re-run Report: Inference Delivery (SLUICE-USE-0 Evidence)

> **Status (2026-11-09): manager-audited evidence; the basis of SLUICE-USE-0's bounded qualification.** The USB SSD route is qualified
> (see [PLAN.md](../../plan/PLAN.md) row 3); the internal-NVMe route is not. Executed by the agy executor in three audited rounds
> (raw rows `results/cat1_rerun/runs.jsonl`, 54 rows, append-only).
> Not measured here: the coded arm's read stage separately from MPS ingest (so the gate's decode term is unchecked),
> a true warm cache, energy/thermal, and any machine other than this one. Supersedes
> [category1_inference_report.md](category1_inference_report.md).

> **Platform:** Apple M4 Mac mini (10 CPU cores, 24 GB Unified Memory, macOS 15.4.1)  
> **Environment:** Python 3.11.17, torch 2.12.0 (MPS), transformers 5.8.1, lmz 1.3.0, lmsluice 0.2.0 (git `8647de6b`)  
> **Model:** `Qwen3-4B-Instruct-2507` (8.05 GB uncompressed, 3 shards, 398 BF16 tensors)  
> **Storage Links:** Internal NVMe (`/Users/masshj/models`, APFS, 3.12 GB/s cold) | USB SSD (`/Volumes/Elements 1`, `/dev/disk4s1`, exFAT, 0.373 GB/s cold)  
> **Cache Invariant Protocol:** USB verified via `diskutil unmount`/`mount`; Internal verified via `fcntl(F_NOCACHE)` re-staging; `mincore()` residency checked before every cold run.

---

## 1. Executive Summary & Honest Verdicts

This evaluation measures the end-to-end model delivery time from storage to a ready inference model on Apple Silicon MPS ($t_0 \to t_{\text{end}}$) across 10 arms, comparing incumbent `safetensors`, `lmsluice` plain mmap, `lmsluice` coded `lmz`, and parallel eager readers (Arms D and D2).

### Headline Verdicts by Location:

#### 1. External USB SSD Link (0.373 GB/s cold storage link):
- **Coded vs Incumbent Plain (C-usb vs A-usb):** Coded transport is **2.00× faster** cold (15.12 s vs 30.29 s).
- **Coded vs Best Plain Measured (C-usb vs D2-usb parallel reader, buffer freed per shard):** Coded transport is **1.52× faster** cold (15.12 s vs 22.93 s). *(Note: Round-2 Arm D-usb at 22.28 s is superseded by D2-usb).* 
- **Coded vs Ideal Plain Reader (Projection at 0.373 GB/s storage ceiling):** Coded transport is projected to be **1.48× faster** (15.12 s vs 22.37 s projection).
- **Break-even:** Pays off on cold load **0.34** (encode only) or load **1.26** (encode + USB transfer).

#### 2. Internal NVMe Link (3.12 GB/s cold storage link):
- **Coded vs Incumbent Plain (C-int vs A-int):** Coded transport is **1.43× faster** on headline cold (5.15 s vs 7.35 s).
- **Coded vs Best Plain Measured (C-int vs D2-int parallel reader, buffer freed per shard):** Coded and parallel plain show **no measurable difference** cold (5.15 s [range 4.65–5.62 s] for C-int vs 5.26 s [range 5.16–5.71 s] for D2-int, a **1.02×** difference). With overlapping min/max ranges, neither route is an outright winner. *(Note: Round-2 Arm D-int at 5.66 s is superseded by D2-int due to harness host buffer overlap).* 
- **Memory Footprint Trade-off:** Coded transport incurs **4772 MB peak RSS vs 521 MB for incumbent (9.1× higher)** due to holding the host buffer while copying to MPS. Arm D2 also incurs 4468 MB peak RSS for the same reason.
- **Ideal Plain Projection:** At the unthrottled 3.12 GB/s NVMe storage ceiling, an ideal plain reader would reach ~3.38 s (1.53× faster than coded). Neither Python framework loader reached that headline rate due to cold demand-paging and allocation bounds.

---

## 2. Predicted vs. Measured Performance

> **Platform:** Apple M4 Mac mini (10 CPU cores, 24 GB Unified Memory, macOS 15.4.1)  
> **Environment:** Python 3.11.17, torch 2.12.0 (MPS), transformers 5.8.1, lmz 1.3.0, lmsluice 0.2.0 (git `8647de6b`)  
> **Model:** `Qwen3-4B-Instruct-2507` (8.05 GB uncompressed, 3 shards, 398 BF16 tensors)  
> **Storage Links:** Internal NVMe (`/Users/masshj/models`, APFS, 3.12 GB/s cold) | USB SSD (`/Volumes/Elements 1`, `/dev/disk4s1`, exFAT, 0.373 GB/s cold)  
> **Cache Invariant Protocol:** USB verified via `diskutil unmount`/`mount`; Internal verified via `fcntl(F_NOCACHE)` re-staging; `mincore()` residency checked before every cold run.

| Storage Location | Transport Arm | Implementation Route | Predicted Bound | Predicted Rate | Measured Cold $t_0 \to t_{\text{end}}$ | Effective Cold Rate | Repeat $t_0 \to t_{\text{end}}$ (Partially Resident) | Speedup vs Incumbent (Cold) |
|---|---|---|---|---:|---:|---:|---:|---:|
| **USB SSD** | **A-usb** (Incumbent) | `safetensors.torch.load_file` (mmap) | Storage (0.373 GB/s) | 0.373 GB/s | 30.29 s | 0.27 GB/s | 26.39 s | 1.00× (baseline) |
| **USB SSD** | **B-usb** | `lmsluice.torchadapter` (plain mmap) | Storage (0.373 GB/s) | 0.373 GB/s | 30.08 s | 0.27 GB/s | 25.33 s | 1.01× |
| **USB SSD** | **D-usb** *(superseded)* | Parallel eager reader (`m.load`, held buffer) | Storage (0.373 GB/s) | 0.373 GB/s | 22.28 s | 0.36 GB/s | — | 1.36× |
| **USB SSD** | **D2-usb** | Parallel eager reader (`m.load`, freed buffer) | Storage (0.373 GB/s) | 0.373 GB/s | 22.93 s | 0.35 GB/s | — | 1.32× |
| **USB SSD** | **C-usb** | `lmsluice` Coded LMZ (archive) | Storage / ratio | 0.566 GB/s | **15.12 s** | **0.53 GB/s** | 15.15 s | **2.00×** |
| *USB SSD* | *Ideal Plain (Proj)* | *Saturating 0.373 GB/s link* | *Storage* | *0.373 GB/s* | *22.37 s* | *0.36 GB/s* | *—* | *1.35×* |
| **Internal NVMe** | **A-int** (Incumbent) | `safetensors.torch.load_file` (mmap) | Storage (3.12 GB/s) | 3.12 GB/s | 7.35 s | 1.09 GB/s | 6.97 s | 1.00× (baseline) |
| **Internal NVMe** | **B-int** | `lmsluice.torchadapter` (plain mmap) | Storage (3.12 GB/s) | 3.12 GB/s | 7.56 s | 1.06 GB/s | 6.92 s | 0.97× |
| **Internal NVMe** | **D-int** *(superseded)* | Parallel eager reader (`m.load`, held buffer) | Storage / host copy | 3.12 GB/s | 5.66 s | 1.42 GB/s | — | 1.30× |
| **Internal NVMe** | **D2-int** | Parallel eager reader (`m.load`, freed buffer) | Storage / host copy | 3.12 GB/s | 5.26 s | 1.53 GB/s | — | 1.40× |
| **Internal NVMe** | **C-int** | `lmsluice` Coded LMZ (archive) | Decode bound | 2.09 GB/s | **5.15 s** | **1.56 GB/s** | 5.06 s | **1.43×** |
| *Internal NVMe* | *Ideal Plain (Proj)* | *Saturating 3.12 GB/s link* | *Storage* | *3.12 GB/s* | *3.38 s* | *2.38 GB/s* | *—* | *2.18×* |

---

## 3. Comprehensive Results Table (Medians, Ranges & Splits)

> **Platform:** Apple M4 Mac mini (10 CPU cores, 24 GB Unified Memory, macOS 15.4.1)  
> **Environment:** Python 3.11.17, torch 2.12.0 (MPS), transformers 5.8.1, lmz 1.3.0, lmsluice 0.2.0 (git `8647de6b`)  
> **Model:** `Qwen3-4B-Instruct-2507` (8.05 GB uncompressed, 3 shards, 398 BF16 tensors)  
> **Storage Links:** Internal NVMe (`/Users/masshj/models`, APFS, 3.12 GB/s cold) | USB SSD (`/Volumes/Elements 1`, `/dev/disk4s1`, exFAT, 0.373 GB/s cold)  
> **Cache Invariant Protocol:** USB verified via `diskutil unmount`/`mount`; Internal verified via `fcntl(F_NOCACHE)` re-staging; `mincore()` residency checked before every cold run.

Repeats: **3 verified cold runs per arm**, plus 2 repeat runs for Arms A, B, C.

| Arm ID | Source Device | Transport Method | Cold $t_0 \to t_{\text{end}}$ (min / med / max) | Cold Splits (Load / SD / Fwd) | Cold Peak RSS | Disk Read Cold | Repeat $t_0 \to t_{\text{end}}$ (med) | Repeat Disk Read |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **A-usb** | USB SSD | safetensors mmap | 29.48 / **30.29** / 31.36 s | 29.44s / 0.005s / 0.845s | 528.4 MB | 8.05 GB | 26.39 s | 8.05 GB |
| **B-usb** | USB SSD | lmsluice plain mmap | 29.24 / **30.08** / 30.37 s | 29.25s / 0.005s / 0.830s | 525.1 MB | 8.05 GB | 25.33 s | 8.05 GB |
| **D-usb** *(superseded)* | USB SSD | parallel eager (`m.load`, held buffer) | 22.26 / **22.28** / 23.43 s | 22.06s / 0.004s / 0.214s | 8195.1 MB | 8.05 GB | — | — |
| **D2-usb** | USB SSD | parallel eager (`m.load`, freed buffer) | 22.67 / **22.93** / 24.92 s | 22.69s / 0.004s / 0.238s | 4494.0 MB | 8.05 GB | — | — |
| **C-usb** | USB SSD | lmsluice coded archive | 15.06 / **15.12** / 16.90 s | 14.88s / 0.004s / 0.232s | 4558.5 MB | 5.34 GB | 15.15 s | 5.33 GB |
| **A-int** | Internal | safetensors mmap | 6.83 / **7.35** / 8.28 s | 6.60s / 0.005s / 0.757s | 521.2 MB | 8.04 GB | 6.97 s | 8.05 GB |
| **B-int** | Internal | lmsluice plain mmap | 7.42 / **7.56** / 7.70 s | 6.78s / 0.006s / 0.854s | 522.9 MB | 8.04 GB | 6.92 s | 8.05 GB |
| **D-int** *(superseded)* | Internal | parallel eager (`m.load`, held buffer) | 5.34 / **5.66** / 5.80 s | 5.18s / 0.005s / 0.477s | 5957.8 MB | 8.04 GB | — | — |
| **D2-int** | Internal | parallel eager (`m.load`, freed buffer) | 5.16 / **5.26** / 5.71 s | 4.74s / 0.004s / 0.513s | 4468.0 MB | 8.04 GB | — | — |
| **C-int** | Internal | lmsluice coded archive | 4.65 / **5.15** / 5.62 s | 4.64s / 0.005s / 0.489s | 4772.2 MB | 5.34 GB | 5.06 s | 3.90 GB |

---

## 4. D2 Per-Shard Splits & Gate Check Analysis

### Arm D2 Per-Shard Timing Splits (Cold Internal NVMe)

| Shard | Size | Plain Read Time ($t_{\text{read}}$) | Read Throughput | MPS Ingest ($t_{\to\text{mps}}$) | Combined Shard Time |
|---|---|---:|---:|---:|---:|
| `model-00001-of-00003.safetensors` | 3.96 GB | 1.27 s | 3.11 GB/s | 0.34 s | 1.61 s |
| `model-00002-of-00003.safetensors` | 3.99 GB | 1.29 s | 3.10 GB/s | 1.44 s | 2.73 s |
| `model-00003-of-00003.safetensors` | 0.10 GB | 0.03 s | 3.25 GB/s | 0.01 s | 0.04 s |
| **Total / Effective** | **8.05 GB** | **2.59 s** | **3.10 GB/s** | **1.79 s** | **4.38 s** |


### Gate Check Analysis: Two-Stage Verification

1. **(a) Read Stage Verification:**
   - **Gate Assumed Storage Rate:** 3.12 GB/s.
   - **Measured Plain Read Rate (Arm D2-int):** **3.10 GB/s** (8.045 GB read across shards in 2.59 s cold).
   - **Verdict:** **YES** — With host buffers freed between shards, plain read stage reaches saturation of the physical internal NVMe storage link (3.10 GB/s vs 3.12 GB/s gate assumption), completely confirming the gate's storage bandwidth assumption.

2. **(b) Endpoint Ratio Analysis (The Shared MPS Ingest Cost):**
   - **Shared MPS Ingest Duration:** **1.79 s** spent on `.to("mps")` host-to-device memory copy and MPS synchronization.
   - **Impact on Endpoint Ratio:** Both plain and coded transport must materialize device tensors on MPS, paying this invariant ~1.8 s ingest penalty (along with ~0.005 s `load_state_dict` and ~0.76 s first forward pass). Because this shared ~2.3 s post-read cost is added to both routes, the end-to-end delivery ratio ($t_0 \to t_{\text{end}}$) is compressed toward ~1.0× (5.15 s vs 5.26 s), even when storage/decompression rates differ during the read stage.


---

## 5. Key Engineering Findings (Mechanisms & Explanations)

### F1: Torchadapter `_mps` Unconditionally Maps Plain Files
In `lmsluice.torchadapter`:
`_mps` calls `_host(m, torch)`, where `prefer_map=True` defaults to True. Even on a cold local file where `_host` docstrings state that transport pread should beat cold mapping, `open_model.map()` is chosen. Consequently, Arm B performs identically to Arm A (plain mmap) rather than executing the parallel transport. Arm D2 confirms that bypassing mmap via `open_model.load()` accelerates plain loads from 7.35 s to 5.26 s.

### F2: USB Profile Keying Across Remounts
`lmsluice plan` reported `source unknown` on USB because the machine profile in `profile.json` is keyed by `st_dev` (`dev:16777241`), but macOS dynamically allocated `dev:16777237` upon volume remount. External filesystem keys are not persistent across mount lifecycles.

### F3: Memory Footprint of Coded and Eager Adapters
Arms C and D/D2 exhibit ~4.5–4.8 GB peak RSS on unified memory, whereas Arms A and B exhibit ~525 MB peak RSS. This difference arises because Arms C and D/D2 hold an intermediate decoded/read host buffer while transferring tensors via `.to("mps")`. Incumbent `safetensors` streams individual tensors directly to MPS without materializing the full checkpoint in a single host buffer.

### F4: Cold Mmap Demand-Paging Cap on Internal NVMe
The limitation of plain incumbent `safetensors` on internal NVMe (1.09 GB/s headline, 1.20 GB/s load split against a 3.12 GB/s link) is **demand-paging a cold mmap**, not MPS dispatch latency.
- *Manager diagnostic (Apple M4, internal NVMe, shard 1 3.96 GB, indicative):*
  - Resident file $\to$ `safetensors.load_file(device="mps")`: 0.64–0.99 s = **4.0–6.2 GB/s** (confirming MPS is not the bottleneck).
  - Verified cold (0.00% resident) $\to$ `safetensors.load_file(device="mps")`: **1.32–1.39 GB/s**.
  - Cold serial mmap page touch only (no torch): **0.92 GB/s**.
  - Cold 16-thread $\times$ 64 MiB `preadv` into one buffer: **1.66–2.89 GB/s**.
Cold single-threaded page faulting across 8 GB of mmapped address space throttles throughput on both NVMe and USB SSD.

### F5: Harness Host Buffer Overlap in Arm D vs Arm D2 Immediate Release
In the initial Round 2 implementation of Arm D, `base = m.load()` retained the previous shard's 4 GB host buffer while reading the next shard. On unified memory, holding shard 1's MPS tensors (4 GB) + shard 1's host buffer (4 GB) + shard 2's host buffer (4 GB) created ~12 GB memory pressure, which throttled shard 2 reads from ~1.3 s to 2.04–2.54 s.
In Arm D2, immediately releasing each shard's host buffer (`m.close(); del base; del mv; gc.collect()`) completely relieves this memory pressure, enabling `lmsluice`'s parallel transport to sustain ~3.1 GB/s cold read throughput across all shards.
**Architectural Implication:** Any `lmsluice` pipeline loading models shard-by-shard on unified memory architectures must explicitly release host buffers before reading the subsequent shard.

---

## 6. Cold Cache Integrity & Audit Disclosures

1. **Repeat Runs Disclosure:** Repeat runs immediately following cold runs were **partially resident, not warm**. Shard 1 exhibited ~0.0% residency, Shard 2 exhibited ~21–75% residency, and each repeat run read 3.8 to 8.05 GB from physical storage. They are correctly reported as repeat runs under partial cache residency; no fully warm claim is made.
2. **Cold Residency Audit & Deviation Disclosure:**
   - Shards 1 and 2 (3.96 GB and 3.99 GB) exhibited **0.00%** residency across all cold runs.
   - Shard 3 (99.6 MB) exhibited **2.1%–3.2%** residency (~2–3 MB resident) after unmount/remount on USB and fresh inode staging. This minor deviation from the <1.0% threshold is immaterial because the cold read was 97% cold on shard 3 and 100% cold on shards 1 and 2 (>99.96% cold across the 8.05 GB model), and every cold run read the full byte count from storage.
   - All cold runs are retained without data pruning.

---

## 7. Numerical Verification

All arms evaluated against **Arm A-int** as ground truth reference:

| Arm | Logit Equality (`torch.equal`) | Max Abs Logit Difference | 32-Token Greedy Match | Verdict |
|---|---|---:|---|---|
| **A-usb** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **B-usb** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **D-usb** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **D2-usb** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **C-usb** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **A-int** (Ref) | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **B-int** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **D-int** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **D2-int** | True | 0.00000000 | 100% Bit-Exact | **PASS** |
| **C-int** | True | 0.00000000 | 100% Bit-Exact | **PASS** |

Greedy Generation Output (identical across all evaluated arms):
> *"Hello! Tell me what makes Apple Silicon unique in one short sentence. Apple Silicon is unique because it is a custom-designed, integrated system-on-a-chip (SoC) that combines the CPU, GPU, Neural Engine, and"*
