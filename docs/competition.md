# The competition — what else moves model weights, and how lmsluice compares

*Written 2026-08-30. Every competitor number below is **published by its
author** and is quoted with the hardware it was taken on; every lmsluice number
is from `MEASURED.md` and names this box (9800X3D / RTX 5080 / 24 GB WSL2,
Gen4 x16, Gen4 NVMe). **No cross-machine comparison here is like-for-like**, and
the places where that matters most are marked. Sources are listed in §12.*

---

## 0. Why the four-row table was the wrong shape

`docs/strategy.md` §4b compares lmsluice against four products. That table is
right about each of the four and wrong about the shape of the field, because
**lmsluice is not one product**. It is a router that happens to ship a
transport, and the things it is measured against sit in six different lanes
that barely talk to each other:

| lane | the job | who owns it | is lmsluice in it? |
|---|---|---|---|
| **A** | open a local file fast | safetensors, GGUF/llama.cpp | yes, and it says it loses on fast NVMe |
| **B** | object storage → GPU, cold start | tensorizer, Run:ai, SageMaker FML, fastsafetensors | **not close** — no S3, no auth, no engine integration |
| **C** | get the bytes onto the machine at all | hf_xet, OCI/ModelPack, Alluxio/JuiceFS | **partly, and this is the best arithmetic in the repo** |
| **D** | make the bytes fewer, losslessly | ZipNN, ZipLLM, DFloat11, DietGPU, nvCOMP | **lmz's lane, not lmsluice's** |
| **E** | don't load at all — snapshot and restore | Modal, Cerebrium, Baseten, vLLM sleep mode | no, and this lane is eating lane B |
| **F** | decide *whether* to compress | ZFS/LZ4 early abort, and nobody else | **this is the whole product** |

The one-line reading: **lanes B and D are crowded and well funded, lane E is
quietly winning the argument lane B was having, and lane F is empty.**

---

## 1. Lane A — local loaders (the incumbent)

| | what it is | impl / deps | sources | compresses | published numbers (their hardware) |
|---|---|---|---|---|---|
| **safetensors** | header + raw tensor bytes, mmap'd, zero-copy | Rust core, Python/JS bindings; Apache-2.0 | local file | **no** | 76.6× vs `torch.load` on CPU (gpt2, Xeon @2.0 GHz); 2.1× on GPU (Tesla T4). Both warm — mmap makes the CPU figure a *deferred* read, not a fast one |
| **torch.load / .bin** | pickle | PyTorch | local | no | the baseline the above beats |
| **GGUF + llama.cpp** | quantised binary designed to be mmap'd | pure C/C++, zero deps | local | quantisation (lossy), not entropy coding | model load 241 ms vs Ollama's 553 ms on the same file |
| **fastsafetensors** | safetensors loader over **GPUDirect Storage** | IBM Research; C++/CUDA + `cufile`; in vLLM as `--load-format fastsafetensors` | local, GDS-capable FS | no | **4.8–7.5×** vs the default safetensors deserializer; **26.4 GB/s** NVMe read, Llama-70B on 4 GPUs; vLLM startup 12.39 s → 4.74 s (Llama-2-13B, 4×L40S) |
| **ServerlessLLM `sllm-store`** | loading-optimised checkpoint format + multi-tier (GPU/DRAM/SSD) pipeline | C++/CUDA, OSDI'24 | local tiers | no | **3.6–8.2×** vs existing loaders; **6–10×** vs the safetensors loader |
| **DeepNVMe / DCP** | async and GDS-backed checkpoint I/O | DeepSpeed / PyTorch | local | no | PyTorch reports **6× faster async checkpointing** with cached plans |

**What this lane tells lmsluice.** It is honest about losing here and it should
stay honest, but the reason has moved. The incumbent is no longer "safetensors
at disk speed" — it is **safetensors at 26.4 GB/s with the CPU removed from the
path entirely**. lmsluice's gate on a machine with GDS is not `1.05 vs 6.34`,
it is `1.05 vs 26.4`, a 25× tax. Nothing in lane A is reachable, and nothing in
lane A reaches the machines lmsluice is for: GDS needs an NVIDIA GPU, a
supported filesystem and `cufile`; there is no such thing on a phone, a SATA
laptop or a 9p mount.

---

## 2. Lane B — object storage → GPU (the crowded one)

| | what it is | impl / deps | sources | compresses | published numbers (their hardware) |
|---|---|---|---|---|---|
| **CoreWeave tensorizer** | custom `.tensors` serialisation, deserialised straight to GPU | Python + PyTorch, boto3, libsodium; in vLLM as `--load-format tensorizer` | **local, HTTP(S), S3**, Redis (preliminary) | **no** | GPT-J 20 GB at **~5 GB/s wire speed on 40 GbE**; local ~2.25 GiB/s; HTTP 0.875–1.125 GiB/s; `plaid_mode` ~4.625 GiB/s. **Also: per-tensor encryption** (libsodium), bf16/fp8 support |
| **Run:ai Model Streamer** (NVIDIA) | concurrent streaming of *unmodified safetensors* into GPU memory | C++ core + Python; in vLLM as `--load-format runai_streamer` / `runai_streamer_sharded` | local, **S3**, Azure Blob, GCS | **no** | Llama-3-8B (15 GB) on g5.12xlarge / A10G: **S3 4.88 s @ concurrency 32** (≈3.1 GB/s) vs tensorizer 37.36 s; IO2 SSD 7.53 s vs safetensors 47 s; GP3 14.34 s vs 47.99 s. vLLM total readiness S3 23.18 s vs tensorizer 65.18 s |
| **SageMaker Fast Model Loader** | streams weights from S3 to the accelerator in chunks | AWS proprietary, LMI container v13+ | S3 only | no | "up to **15×** faster"; Llama-3.1-70B in **~1 minute** on ml.p4d.24xlarge (≈2.3 GB/s); ~19% latency reduction when scaling a new copy |
| **fastsafetensors + FSx for Lustre** | GDS from a parallel FS | AWS reference architecture | Lustre | no | see lane A |

**Three things worth taking from this lane.**

1. **The format is not the moat; the concurrency is.** Run:ai beats tensorizer
   by 7.7× from S3 while reading *plain safetensors* — because it opens 32
   streams and tensorizer opened 16. That is the same finding as
   `MEASURED.md`'s "fetching wants queue depth", arrived at independently and
   shipped in C++.
2. **Nobody in this lane compresses.** Not one of them sends fewer bytes. On a
   40 GbE or S3-in-region link they are right not to: the link is 3–5 GB/s and
   a CPU decoder would be the bottleneck. That is lmsluice's own gate agreeing
   with them, which is worth saying out loud — **the gate predicts the market's
   behaviour**, and the market is sitting on the tax side of it.
3. **The prize is smaller than it looks.** The 2026 cold-start study of vLLM
   (Kabakibo, Trivedi & Wang, arXiv 2606.07362) finds, all in **§3.3 "Impact of
   SSDs"**, that **model loading is 7–10% of total startup**, that flushing the
   buffer cache so reads come from SSD rather than warm DRAM slows the loading
   step by **0.5×** while total startup changes by only **1.04×** (their
   Figure 13, averaged across models and normalised to the DRAM baseline).
   Separately,
   **§3.5 with Figure 15** gives `torch.compile` graph storage at **11–21 s**
   uncached against **3–6 s** cached (the cached figure is Figure 6). A perfect
   loader in this lane wins a few percent of a number dominated by compilation.

   *Testbed, verified:* the SSD experiment runs on the paper's node **n3 — Intel
   Xeon Platinum 8568Y+ with an H100** — over an **LVM volume mirroring four
   PCIe 5.0 SSDs, 25 GB/s read and 15 GB/s write by `fio`** for large transfers,
   verbatim *"the first ten models listed in Table 1, each repeated five times"*.
   An earlier version of this paragraph attributed it to node n1's EPYC 9354,
   which is the paper's *main* node and not the one the storage experiment used.

   *One figure deliberately not cited:* the paper's **Figure 14** compares three
   loading backends across Llama2-13B, Yi-6B, Llama2-7B and Falcon-7B, and its
   textual claim is that Tensorizer is consistently lowest. Its twelve bar values
   are **not recoverable from the PDF** — glyph extraction returns a different
   subset of them depending on the grouping tolerance used (eight values at one
   setting, seven overlapping-but-different at another), so any (model, backend)
   attribution would be an artefact of the extraction rather than a reading of
   the paper. Nothing in this document rests on it.

`strategy.md` says of this lane: *"this is our headline case and they own
it."* The evidence supports the second half and undercuts the first — it should
not be the headline case, because the headline is somebody else's number.

---

## 3. Lane C — getting the bytes onto the machine

This is the lane §2b of `strategy.md` calls the largest case, and it is more
contested than that section allows.

| | what it is | impl | dedup / compression | published numbers |
|---|---|---|---|---|
| **`hf_xet`** (HF's default) | content-defined chunking at ~64 KB, chunk-level dedup, adaptive 1→64 parallel streams | Rust | **dedup, no entropy coding** | 2.5 GB/s sustained for a 594 GB datacenter download; ~94 GB saved of ~191 GB on one GGUF repo. Measured **14.8%** total reduction across a 43 TB hub sample by a third party |
| **`hf_transfer`** | parallel range GETs | Rust | none | **deprecated** — `HF_HUB_ENABLE_HF_TRANSFER` is now a no-op; `HF_XET_HIGH_PERFORMANCE=1` replaces it |
| **OCI model artifacts** (ModelPack, KServe modelcar, KAITO) | ship weights as container layers; image volumes, lazy pull | containerd, CNCF ModelPack | **zstd layers** (KAITO chose zstd over gzip explicitly for decompression speed on weight files) | KServe 0.14 promoted OCI storage to stable and added a model cache |
| **Alluxio / Fluid / JuiceFS** | a caching filesystem between object storage and the node | Java/Go, k8s CSI | none | 70B-class load 42 min (cross-region) → 14 min (Alluxio) → **3 min** (Fluid prefetch); peer-node NVMe fetch ~2 GB/s over LAN; Inferless reports 12× |
| **Play for On-device AI** | Android model delivery as AI packs | Google Play | packs sized by **compressed** download, ≤1.5 GB each | the phone case, already productised by the platform vendor |

**The uncomfortable part.** `hf_transfer` — one of the four rows in the current
strategy table — is gone, and what replaced it is *not* "parallel ranges, no
compression". It is chunk dedup with adaptive concurrency, which attacks the
same bytes lmz's directory-level dedup attacks. Two consequences:

- **Xet already took the easy half of the download win** on the public hub, and
  took it at the storage layer where no user has to install anything.
- **Compression and CDC dedup fight each other far less than this section
  originally claimed.** It said entropy-coded bytes do not chunk-dedup, so a
  published archive would lose Xet's cross-version dedup to gain lmz's ratio.
  That has now been measured and it is false for the case that matters. lmz
  chunks fixed spans of *plaintext*, so a shape-preserving edit — the normal
  shape of a fine-tune — leaves every untouched chunk byte-identical after
  coding: replacing the largest tensor changed 52.7% of plain bytes and exactly
  **52.7%** of coded chunks.

  Scattered edits are amplified instead, by roughly the ratio of coded chunk
  size to CDC chunk size — **and that ratio is a dial the encoder holds**. On the
  same 0.30% scattered edit, 1 MiB chunks amplify 107× while 64 KiB chunks
  amplify 10.8×, for 2.1 points of ratio.

  Coded wins while `R < 1 + (1-f)/(c·(f·A - 1))` for `R` revisions sharing a
  base and plain change fraction `c`: about **5 revisions** at 1 MiB chunks and
  about **48** at 64 KiB. **Both figures come from one measured `c` of 0.30%** —
  they are one assumption reported twice, not two results, and `c` for real
  consecutive fine-tune releases is unmeasured and is the input the crossover is
  most sensitive to. A cold fetch of any single revision is a clean win for
  coding either way, since dedup buys nothing when you hold nothing.
  `MEASURED.md` has the method, including a chunker bug that would have
  confirmed the original hypothesis for a reason unrelated to the data.

**What the 64.6% is worth depends on where it is published.** Against plain
storage it stands. On a **Xet-backed hub specifically it overstates the
advantage**, because content-defined chunking already dedups the duplicate
representation for nothing — measured at 100% of chunks shared between a
safetensors directory and the uncompressed-zip `.pth` holding the same weights.
Against Xet, lmz's marginal contribution is the 34.7% shard figure and not the
64.6% one.

Also note **KAITO already switched its OCI layers to zstd** for exactly the
reason `zstdcodec.py` exists. The "nothing installed" argument is not unique;
what is unique is deciding per machine whether to use it.

---

## 4. Lane D — lossless model compression (lmz's lane, listed for completeness)

| | ratio on BF16 LLM weights | decode throughput (their hardware) | GPU decode | notes |
|---|---|---|---|---|
| **lmz** | **34.7% saved** (Llama-3.1-8B shards); **64.6%** whole directory with dedup — but see the note below on what that is worth on a Xet-backed hub | **5.95 GB/s** through lmsluice, 8 threads, `CODEC_BF16` at 1 MiB chunks (9800X3D, WSL2, CRC on); 0.97 GB/s for `CODEC_BF16C` at 8 MiB, which is the same library on the same data; 418 GB/s CUDA shared-table (RTX 5080) | **yes** | joint-entropy bound is 35.0% — 0.3 points away |
| **ZipNN** | 33.6% saved (published, same model) | **1.66 GB/s single-thread**; **up to 80 GB/s @ 16 workers** (Xeon Platinum 8480+, 224 cores, dual NUMA) | "on the way" | integrations shipped: HF `from_pretrained` plugin, safetensors monkey-patch, vLLM, sglang |
| **ZipLLM** (research) | **54.1%** across 3,048 models / 43.19 TB — tensor-level dedup + **BitX delta** against the base model + zstd | 7.9 GB/s decompress, 5.9 GB/s ingest | no | finds 99.64% of hub models are fine-tunes; argues tensor-boundary dedup needs 923K hashes where CDC needs 520M |
| **DFloat11** | ~30% saved (BF16 → ~11 bits) | GPU kernel, decompresses per transformer block **during inference** | **yes, resident** | 1.9–38.8× throughput vs CPU offload; 5.3–13.17× longer context at fixed VRAM |
| **DietGPU** (Meta) | generic ANS + float-exponent mode | **250–600 GB/s** float mode (A100) | **yes** | the GPU rANS prior art |
| **nvCOMP** (NVIDIA) | LZ4 / GDeflate / ANS / Bitcomp | ANS ~480 GB/s end-to-end; LZ4 118–320 GB/s | **yes** | ships with CUDA |

**What this lane says about lmsluice.**

- **The ratio race is over and lmz won it by 1.1 points.** 34.7% against ZipNN's
  33.6%, with a 35.0% ceiling. Nobody is going to beat that lane by enough to
  matter. `strategy.md`'s "lmz already beats it on ratio" is correct **and is
  not a moat**.
- **The dedup race is not over, and ZipLLM is ahead of where this project has
  looked.** lmz's 64.6% is dedup *within one directory*; ZipLLM's 54.1% is
  dedup *and delta across a hub*. Those measure different things, but ZipLLM
  names the structure that matters — almost every model is a fine-tune — and
  that is `strategy.md`'s Phase 3, unbuilt.
- **The CPU decode gap is much smaller than this section first claimed.** It
  gave lmz at 0.39–0.48 GB/s single-threaded and 1.05 at 8–16 threads against
  ZipNN's published 1.66 single-thread, and called the gap 3.5×. Those were
  `CODEC_BF16C` figures. Corrected, on the same box with CRC on: **0.95 GB/s
  single-threaded** (1.09 with CRC off) and **5.95 GB/s through the transport**,
  or 8.29 decode-only at 16 threads. So the single-thread gap is about
  **1.5–1.75× on a different CPU**, not 3.5×, and ZipNN's 1.66 was taken on a
  224-core dual-NUMA Xeon 8480+ where its own published multi-thread figure
  needs that machine to reach 80 GB/s.

  Still not a like-for-like comparison and still not to be quoted as one — but
  the direction of the worry has changed, and it is why fetching torch to
  measure ZipNN was deferred rather than done. Running ZipNN on this box against the same archive, in a
  fresh environment, is the single highest-value competitive measurement
  available and it has not been done.
- **GPU decode is a commodity.** DietGPU at 250–600 GB/s and nvCOMP ANS at
  ~480 GB/s bracket lmz's 418 GB/s. `strategy.md` §2 already de-prioritised the
  device route on cost-and-reach grounds; the landscape adds a second reason —
  it is the part of the stack with the most competent incumbents.

---

## 5. Lane E — don't load at all

The lane that is quietly winning the argument lane B is having.

| | what it does | published numbers |
|---|---|---|
| **Modal memory snapshots** | checkpoint/restore of a whole container, now including GPU memory | restore ~2.5× faster than cold start; Stable Diffusion 13 s → 3.5 s; `import torch` 5 s → 1.05 s p50, 0.69 s p0; a full vLLM config down to ~70 s (6.5×) |
| **Cerebrium snapshots** | same idea | claims 85% faster than Baseten's cached cold starts, up to 94% on vLLM |
| **vLLM sleep mode / Tangram** | keep or reuse GPU memory across model swaps | research and shipped variants |

If the weights are already in a snapshot of device memory, **the loader is not
in the path at all** — not safetensors, not Run:ai, not lmsluice. This is the
strongest argument in this document for `strategy.md`'s decision not to chase
the vLLM cold start: that market's endgame is not a faster loader, it is no
loader. It says nothing at all about a laptop on a SATA SSD, a phone, or a
first download, because none of those have a warm snapshot to restore.

---

## 6. Lane F — deciding whether to compress at all

The lane the product actually lives in, and it is close to empty.

| prior art | what it decides | on what evidence |
|---|---|---|
| **ZFS / LZ4 early abort** | whether *this block* compressed well enough to keep | **ratio** — LZ4 bails within microseconds on incompressible data; ZFS discards the compressed block unless it is under ⅞ of the original |
| **Adaptive-compression patents** (dedup filesystems, sampling heuristics) | which algorithm, or whether to compress | ratio, sampled |
| **"energy / performance / archive" modes** (ZFS research) | a policy chosen by the operator | a declared preference |
| **lmsluice** | whether the **coded route beats the plain route on this machine** | **rate** — measured `decode` against measured `link`, per storage device, against the archive that will actually be read |

**The distinction is real and it is narrow.** "Decide adaptively whether to
compress" is a forty-year-old idea in filesystems. What none of that prior art
does is compare a **decode rate** against a **link rate** and report the verdict
to the user with the numbers behind it — they all ask "did this compress?", not
"is my decoder faster than my disk?". Those are different questions with
different answers, and the second one is the only one that can conclude *"on
this machine the archive loads at 0.74× the speed of the plain file"* — which is
this box's measured verdict, and a recommendation against the thing being sold.
(The 6× this sentence used to quote was the retired figure that paired the
fastest plain read with the slowest coded one; §2 above and `MEASURED.md` carry
the correction.)

So `strategy.md`'s claim — *"they all assume streaming or compression is
better; we measure the machine and sometimes say no"* — **survives contact with
the field**, with one amendment: the filesystems got there first for the ratio
question, and lmsluice should cite them rather than claim the whole idea.

---

## 7. Head-to-head: features

Rows are capabilities; a blank means the tool does not have it.

| | lmsluice | safetensors | tensorizer | Run:ai | fastsafetensors | ZipNN | hf_xet | sllm-store |
|---|---|---|---|---|---|---|---|---|
| local file | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HTTP range GET | ✅ | | ✅ | | | | ✅ | |
| S3 / GCS / Azure | | | ✅ | ✅ | | | HF only | |
| authentication | | | ✅ | ✅ | | | ✅ | |
| **lossless compression** | ✅ | | | | | ✅ | dedup only | |
| **no install required** | ✅ (stdlib zstd) | | | | | | | |
| decode on GPU | ✅ (lmz CUDA) | | | | n/a | | | |
| GPUDirect Storage | | | | | ✅ | | | |
| **returns torch tensors** | ✅ (optional extra) | ✅ | ✅ | ✅ | ✅ | ✅ | n/a | ✅ |
| vLLM integration | ✅ (`--load-format lmsluice`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| pip-installable | ✅ (built, unpublished) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **measures the machine** | ✅ | | | | | | adaptive concurrency only | tiering only |
| **can advise against itself** | ✅ | | | | | | | |
| **reports predicted vs measured** | ✅ | | | | | | | |
| streaming under a memory ceiling | ✅ | | | | | | | ✅ |
| mmap / partial access | ✅ | ✅ | | | | | | |
| encryption | | | ✅ | | | | | |
| runs with no GPU / no CUDA | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| runs on a phone-class device | ✅ (stdlib) | ✅ | | | | | | |

The three rows in bold that only lmsluice has are the product. The two ❌ rows
in the middle — no tensors, not installable — are why nobody can use it, and
they are `strategy.md`'s Phase A, which remains correctly placed.

---

## 8. Head-to-head: numbers

**Read this table as six separate experiments, not one.** Each row is that
project's own published figure on its own hardware.

| | model / size | machine | route | result |
|---|---|---|---|---|
| Run:ai Model Streamer | Llama-3-8B, 15 GB | g5.12xlarge, A10G | S3 → GPU, c=32 | 4.88 s ≈ **3.07 GB/s** |
| tensorizer | GPT-J, 20 GB | 40 GbE | HTTP → GPU | **~5 GB/s** (wire speed) |
| SageMaker FML | Llama-3.1-70B | ml.p4d.24xlarge | S3 → GPU | ~60 s ≈ **2.3 GB/s** |
| fastsafetensors | Llama-70B | 4 GPUs + GDS | NVMe → GPU | **26.4 GB/s** |
| ZipNN | BF16 weights | Xeon 8480+, 224 c | decompress | **1.66 GB/s** 1T, **80 GB/s** 16 workers |
| DietGPU | float data | A100 | decompress | **250–600 GB/s** |
| **lmsluice** | Llama BF16, 1.87 GB | 9800X3D, RTX 5080, WSL2 | 9p mount, coded | **0.646 GB/s** (1.45× the plain route's 0.447) |
| **lmsluice** | same | same | NVMe → VRAM, device decode | 1.36 GB/s vs 3.10 plain — **a 2.3× tax** |
| **lmsluice** | whisper fp32, 151 MB | same | NVMe → VRAM, device decode | 1.39 vs 0.56 GB/s — **2.48×** |

The absolute numbers are not the comparison. **The comparison is that every
competitor row is a fast machine reading fast storage, and lmsluice's winning
row is a slow link** — which is precisely the segmentation `strategy.md` §1
argues for. On any row above the gate, lmsluice would print a verdict telling
the user to use the plain file.

---

## 9. What lmsluice has that nobody in this survey has

1. **A verdict.** Every other tool assumes its own path is best. `lmsluice plan`
   is the only thing here that can tell a user their compressed model is
   costing them 6×.
2. **Predicted beside measured.** `lmsluice bench` prints both. No competitor
   publishes a falsifiable prediction of its own speedup, let alone checks it
   at run time.
3. **A codec that needs nothing installed.** stdlib zstd — 21.1% saved,
   **0.77 GB/s single-threaded** on this box and 1.69 at 8–16 threads, where it
   stops scaling (`MEASURED.md`); an earlier 0.59–0.71 here predates that
   measurement. ZipNN needs numpy, zstandard and
   torch; tensorizer needs torch and boto3; fastsafetensors needs CUDA and
   `cufile`. On the machines this project is for, "pip install" is often the
   whole obstacle.
4. **The machines nobody else targets.** Every competitor above was benchmarked
   on a datacenter node. Not one published number in lanes A, B or D comes from
   a machine without a discrete NVIDIA GPU.
5. **A gate stated as a curve with its provenance.** `gate.py` carries `k` as an
   interval and refuses to pick a reading it cannot justify. That is unusual
   enough to be worth keeping as a public artefact.

## 10. What everybody else has that lmsluice does not

1. **Tensors.** Seven of eight columns in §7 hand you `torch.Tensor`. lmsluice
   hands you a `memoryview`. Nothing else on this list matters until that does.
2. **Object storage with auth.** tensorizer, Run:ai and SageMaker all speak S3.
   lmsluice speaks plain HTTP. §2b calls the download the best case in the
   repository, and it cannot reach the storage the download comes from.
3. **An engine integration.** `--load-format` is the distribution channel for
   this entire field, and six of the tools above are in it.
4. **Native code.** Everything competitive is Rust, C++ or CUDA. The Python
   decode boundary is not a detail — it is the difference between 0.4 and
   1.66 GB/s single-threaded, and the gate is a rate comparison.
5. **Evidence from more than one machine.** Every competitor publishes at least
   a small matrix of hardware. This repository publishes one box, and says so.

---

## 11. What the landscape changes about the plan

Five conclusions, in the order they should affect `strategy.md`:

1. **Keep the decision to skip lane B, and strengthen the reason.** It is not
   just that Run:ai and tensorizer are ahead. It is that model loading is
   7–10% of vLLM startup on fast storage (2606.07362 §3.3), and that the lane's
   endgame is
   memory snapshots, which remove the loader entirely.
2. **`hf_transfer` is the wrong opponent; `hf_xet` is the right one.** The
   comparison row should be rewritten: HF now ships CDC dedup by default, it
   already captures part of the download win, and it **conflicts** with
   shipping entropy-coded bytes. Pricing that conflict is new work and belongs
   in Phase 0.5.
3. **Ratio is not the differentiator; the verdict is.** lmz leads ZipNN by 1.1
   points against a 35.0% ceiling. Lead with §4's wedge — the free diagnostic —
   and treat the ratio as a footnote, because a rival can close 1.1 points and
   cannot close "we measure your machine".
4. **Measure ZipNN here — attempted, blocked, deferred.** `zipnn` 0.5.4 cannot
   be *imported* without `torch>=2.0.0`: `zipnn.py` imports it directly and
   `util_torch.py` decorates with `@torch.jit.script` at module load, so a stub
   cannot carry it. Its C core is importable alone and confirms the architecture
   is byte-grouping plus zstd, the same family as lmz, but takes ten
   undocumented arguments — reconstructing the pipeline around them would
   measure the reconstruction rather than ZipNN.

   Deferred rather than pursued: ~200 MB of torch on a possibly metered link,
   and the corrected figures above shrank the gap that motivated it from 3.5× to
   about 1.5–1.75×. **The import dependency is itself the competitively
   meaningful fact** — for a layer aimed at phones, iGPU laptops and boxes with
   no CUDA, a codec that cannot be loaded without a tensor framework differs
   materially from lmz (no runtime dependencies) and `zstdcodec` (nothing at
   all).
5. **Phase 3 has a named competitor now.** ZipLLM's finding — 99.64% of hub
   models are fine-tunes of something, and cross-model delta gets 54.1% — is
   the argument for `strategy.md`'s "many models, not one archive", made by
   somebody else first, with a hub-scale measurement behind it.

---

## 12. Sources

Loaders and streamers:
[safetensors](https://huggingface.co/docs/safetensors/index) ·
[safetensors speed](https://huggingface.co/docs/safetensors/speed) ·
[tensorizer](https://github.com/coreweave/tensorizer) ·
[Run:ai Model Streamer benchmarks](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/benchmarks.md) ·
[vLLM Run:ai docs](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/) ·
[vLLM tensorizer docs](https://docs.vllm.ai/en/v0.10.1/models/extensions/tensorizer.html) ·
[fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) ·
[vLLM fastsafetensors docs](https://docs.vllm.ai/en/latest/models/extensions/fastsafetensor/) ·
[SageMaker Fast Model Loader](https://aws.amazon.com/blogs/machine-learning/introducing-fast-model-loader-in-sagemaker-inference-accelerate-autoscaling-for-your-large-language-models-llms-part-1) ·
[ServerlessLLM (OSDI'24)](https://www.usenix.org/system/files/osdi24-fu.pdf)

Transfer and distribution:
[Xet storage](https://huggingface.co/docs/hub/xet/index) ·
[From files to chunks](https://huggingface.co/blog/from-files-to-chunks) ·
[Downloading models](https://huggingface.co/docs/hub/models-downloading) ·
[KServe OCI storage](https://kserve.github.io/website/docs/model-serving/storage/providers/oci) ·
[KAITO model-as-OCI](https://kaito-project.github.io/kaito/docs/next/model-as-oci-artifacts/) ·
[OCI cold-start study](https://arxiv.org/abs/2607.16596) ·
[Alluxio / NetEase 30-second cold starts](https://www.cncf.io/blog/2026/05/21/how-netease-games-achieved-30-second-llm-cold-starts-on-kubernetes/) ·
[JuiceFS / BentoML](https://juicefs.com/en/blog/user-stories/accelerate-large-language-model-loading) ·
[Play for On-device AI](https://developer.android.com/google/play/on-device-ai)

Compression:
[ZipNN](https://github.com/zipnn/zipnn) ·
[ZipNN paper](https://arxiv.org/pdf/2411.05239) ·
[ZipLLM](https://arxiv.org/html/2505.06252v3) ·
[DFloat11](https://arxiv.org/pdf/2504.11651) ·
[DietGPU](https://github.com/facebookresearch/dietgpu) ·
[nvCOMP benchmarks](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md)

Cold start and snapshots:
[Breaking the Ice: cold start latency in vLLM](https://arxiv.org/pdf/2606.07362) ·
[Modal memory snapshots](https://modal.com/blog/mem-snapshots) ·
[Modal GPU snapshots](https://modal.com/blog/gpu-mem-snapshots) ·
[Cerebrium snapshots](https://cerebrium.ai/blog/reducing-gpu-cold-starts-with-memory-snapshots-restoring-cuda-workloads-in-second)

Prior art for the decision:
[Adaptive compression for ZFS (thesis)](https://wr.informatik.uni-hamburg.de/_media/research:theses:florian_ehmke_adaptive_compression_for_the_zettabyte_file_system.pdf) ·
[ZFS compression in practice](https://www.servethehome.com/the-case-for-using-zfs-compression/)

Local inference:
[llama.cpp vs vLLM](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine) ·
[Local inference internals](https://www.internalsdecoded.com/articles/local-inference)
