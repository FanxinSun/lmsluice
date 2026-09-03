# Strategy — what lmsluice is for, and what to build next

*Written 2026-08-30, after a session that produced a lot of correctness work and
no direction. Everything below is argued from measurements in `MEASURED.md`
rather than from intuition, and the uncomfortable conclusions are kept.*

## 1. The one fact that should decide the roadmap

The gate is `decode > link`. Both terms vary, but they vary by very different
amounts across the machines this will run on:

- the **decoder** varies by more than this section used to claim. "Maybe 3×
  across CPUs" was wrong: **codec choice alone is 6.1× on one CPU** (`CODEC_BF16`
  against `CODEC_BF16C` on the same library, same data), and thread count adds
  another 8× on top. The honest statement is that the decode term is
  codec-and-thread dependent over more than an order of magnitude, and the
  figure to gate with is the one measured through the transport that will
  actually do the work: **5.95 GB/s** here (`CODEC_BF16`, 1 MiB chunks, 8
  threads, 9800X3D, WSL2, CRC on). The 8.29 GB/s at 16 threads is decode-only
  from RAM and is a bound, not a crossover.
- the **link** varies by a **factor of 100** — 12 GB/s on a Gen5 NVMe, 6 on this
  box's Gen4, 0.55 on a SATA SSD, 0.3 on eMMC, 0.26 on a 9p mount, 0.125 on a
  gigabit network, less on anything metered.

So the product's value is very nearly a function of one variable: **how slow the
link is.** That conclusion survives the correction; what changes is where the
line falls, and it moves in our favour.

Gate is `decode / link` at 5.95 GB/s; the speedup is `min(1/f, gate)` and cannot
exceed **1.49×**, which is all f = 0.673 can buy.

| link | GB/s | source | gate | speedup | verdict |
|---|---|---|---|---|---|
| ext4 on NVMe, host-cached | 6.2 | measured — but this is the *host* cache, not storage | 0.96× | 0.74× **measured** | **tax** |
| ext4 on NVMe, first touch | 2.4 | measured, and an upper bound on cold | 2.48× | 1.49× *projected* | **pays, capped** |
| UFS 4.0 (phone) | 4.0 | spec | 1.49× | 1.49× *projected* | **pays, at the cap** |
| **9p mount to the Windows host** | **0.26** | measured | 22.9× | **1.45× measured** | **pays** |
| **SATA SSD** | 0.55 | spec | 10.8× | 1.49× *projected* | **pays, capped** |
| **eMMC / SD / USB** | 0.30 | spec | 19.8× | 1.49× *projected* | **pays, capped** |
| **1 Gb/s network** | 0.125 | spec | 47.6× | 1.49× *projected* | **pays, capped** |

**Two rows flipped when the decoder figure was corrected**: first-touch NVMe and
phone-class UFS were both marked "tax" against the retired 1.05 GB/s and both
pay against 5.95. Only the host-cache-warm row still loses, and that row is not
a storage measurement at all — it is RAM, which no CPU decoder competes with.

**The market is every machine that is not reading out of a warm cache.** That is
most laptops, every phone, every network mount, every container pulling from
object storage, and *every download that has ever happened*.

**And that market now has a second reason that needs no archive at all.** §4
records it: on a high-latency mount the plain-file route through this transport
is worth **2.62×** over `safetensors`' mmap on the *same bytes*, before any
compression enters. The table above prices the codec; the transport is a
separate and larger win on the same machines, and it asks nothing of the supply
side.

## 2. The uncomfortable consequence

**The GPU decode work buys the least valuable quadrant.** State this carefully,
because the strong version is wrong and someone checking the arithmetic will find
the gap: a device decoder does **not** merely matter less on a fast link. It
converts a tax into a win there — about 1.49× against a plain file on this box,
which is what `MEASURED.md` documents. The win is real.

The weaker, truer claim is about what that win costs and who gets it. It is
bought with a CUDA context, a device allocation, a plane-merge kernel that is on
the codec's side of the boundary, an occupancy model whose inputs no API fully
reports, and a `nvcc` dependency — all to serve the machine population that has a
discrete GPU *and* a link fast enough to outrun a CPU decoder. Where compression
actually pays, the CPU decoder alone already wins by the full ratio, because the
win saturates at 1/f and the CPU clears the gate several times over, with none of
that machinery.

The objection is price and reach, not absence of value — which is why
`devdecode.py`, `merge.cu`, the occupancy model and the Vulkan question are
**not the core of the product**. They matter in exactly two places:

- **unified memory**, where the upload stage vanishes entirely and the device
  decoder is free — Apple silicon, Strix Halo, and YFCE's own silicon;
- **keeping the model coded in VRAM**, which is `vram/`'s charter, not this one's.

That also means the recent bad news — that the 2-CU iGPU's plausible range now
spans the CPU decoder rather than sitting above it — **does not threaten the
project.** It threatens one argument for one device class. The link argument is
untouched by it and is the stronger of the two.

## 2b. The download is probably the largest case, and it is barely named above

Every model starts as a download, and the numbers there are better than any row
in §1:

| | |
|---|---|
| hub or object-store link | 10–100 MB/s |
| gate against a **5.95 GB/s** decoder, measured through the transport | **60–595×** |
| ceiling, at lmz's shard ratio (34.7%) | 1.53× |
| ceiling, at its **directory** ratio (64.6%) | **2.82×** |

Two things make this different in kind rather than in degree.

**The win is not contingent on the gate.** For a local file the router has to
decide whether decoding beats reading. For a download you always transfer fewer
bytes — that is arithmetic, not a measurement — and the only question is whether
the decode adds time on top. At 10–100 MB/s it cannot: the decoder is two orders
of magnitude faster and disappears entirely under the transfer. **This is the one
case where the answer is yes without probing anything.**

**And it pays twice**, in bytes not transferred and in load time not spent, at the
directory ratio rather than the shard one — because a Hugging Face directory ships
the same tensors more than once and lmz dedups across files, which is where 64.6%
comes from against 34.7%.

The machinery is largely built: `source.py` does HTTP range requests with
per-thread keep-alive, and `lmsluice get` exists. What is missing is not code.

**The catch, and it decides the sequencing.** This case needs a *supply side* —
someone has to publish the archive. That splits it in two:

- **Public hubs** — the biggest prize and an ecosystem problem, not a technical
  one. Nobody publishes lmz archives yet, and asking them to is a different kind
  of work with a different timescale.
- **Your own infrastructure** — object storage to compute nodes, node-to-node
  within a cluster, CI pulling models, an appliance fetching updates. **Both ends
  belong to the same operator**, so there is no ecosystem to persuade, and the
  numbers are the best in this document.

The second is immediately actionable and is the right Phase 0.5. The first is
worth wanting and should not block anything.

**Priced against Xet, 2026-08-30, and it survives.** hf_xet chunks at ~64 KB and
dedups across revisions, which looks like it should beat one-shot coding for
anyone tracking a model. Measured: CDC does catch tensor duplication completely
(100% of chunks shared across two containers holding the same weights), but the
premise that coded bytes cannot dedup is false — lmz chunks fixed spans of
*plaintext*, so a shape-preserving edit leaves untouched chunks byte-identical
after coding. Coding wins outright on any cold fetch, and up to **~5 revisions**
at 1 MiB chunks or **~48** at 64 KiB, which costs 2.1 points of ratio. Crossover
formula and conditions in `MEASURED.md`.

## 3. The metric is wrong, and fixing it is the differentiator

Every number in `MEASURED.md` is GB/s. Nobody starting a model cares about GB/s;
they care about **when the model can do work**. Those are different, and the
difference is where an actual product lives:

- a loader delivering 3 GB/s that must finish before inference starts is worse
  than one delivering 1.5 GB/s that lets layer 0 compute at t = 0.2 s;
- `stream()` already yields tensors as they land, but in **file order**, which is
  not execution order and is nobody's schedule;
- the plan prices routes in total seconds, so it cannot express "first useful
  work" at all.

**Time-to-first-useful-work is the metric this package should own.** No loader
reports it. Adding it is cheap, and it reframes the routing decision: on a slow
link the coded route wins on total time *and* wins by more on first-token time,
because fewer bytes have to arrive before the first layer is complete.

## 4. The wedge

`lmsluice plan ./model` costs nothing, changes no pipeline, and answers a
question nobody else can answer: **"on this machine, is your compressed model
helping or hurting, and by how much?"** The answer is frequently surprising and
occasionally embarrassing — on this box, warm, it is *"your archive loads at
0.74× the speed of the plain file"*. (It used to say 6× here; that figure paired
the fastest plain read with the slowest coded one and is retired.)

A free diagnostic that tells someone something true and surprising about their
own machine is how this gets adopted. The transport is what they use once they
believe the number. **Lead with the router, not the pipeline.**

### The transport is the first win, and compression is the second

**On a slow link, the plain-file route is already a multiple over
`safetensors`' mmap, with no archive involved at all.** Measured inside vLLM
0.28.0 on a 9p mount, cold by distinct never-before-read copies, n=5, using
vLLM's own reported weight-loading time: reading the *same* plain safetensors
file through this package's transport instead of vLLM's default loader is worth
**2.62×** [2.45–2.90]. Parallel `pread`s against mmap page-faults, same bytes,
no codec, no lmz, no supply side.

Compression then adds **1.38×** [1.34–1.55] on top — against a ceiling of 1.48×,
which is all that 0.674-of-the-bytes can arithmetically buy, so the decoder is
keeping up and nearly the whole ratio arrives as speed. The two compose to the
3.60× measured end to end.

**This reorders the pitch.** The compressed archive needs somebody to build one:
a supply side, a format, a dependency, a reason to trust it. The transport needs
none of that — it is a faster way to read the file the user already has, and it
is the larger of the two effects on exactly the machines §1 says the value lives
on. That makes the plain-file path the thing to lead with and the archive the
upgrade, not the other way round.

It is also the second time in a week that the biggest available win turned out
to be the transport rather than the codec; the destination-buffer allocation
finding (`MEASURED.md`, "the destination buffer costs more than the transport")
was the first, and it was worth 2.4× for a buffer.

## 4b. Where this stands against what people already use

**[`docs/competition.md`](competition.md) is the full survey** — six lanes,
twenty-odd products, every number labelled with the machine its author took it
on. The short form:

| they use | for | how we compare |
|---|---|---|
| `safetensors` mmap, GGUF | local loading, the incumbent | it wins on a fast NVMe and we say so. Not a fight worth having. |
| `fastsafetensors` + GPUDirect Storage | local loading with the CPU removed | 26.4 GB/s published. The gate against that is a 25× tax — and it needs an NVIDIA GPU, so it does not reach our machines |
| CoreWeave `tensorizer`, Run:ai Model Streamer, SageMaker Fast Model Loader | object storage → GPU cold start | native code, S3-native, wired into vLLM. They own it — **and it is worth less than it looks**: model loading is 7–10% of vLLM startup on fast storage |
| memory snapshots (Modal, Cerebrium) | the same cold start, without loading | the lane's endgame is no loader at all. A second reason not to fight there |
| `hf_xet` (`hf_transfer` is deprecated) | download acceleration | **chunk-level dedup**, not just parallel ranges — it already takes part of the download win, and CDC dedup **conflicts** with shipping entropy-coded bytes. Unpriced |
| `zipnn` | model compression | lmz leads on ratio by 1.1 points (34.7% vs 33.6%) against a 35.0% ceiling — real, and not a moat. ZipNN publishes **1.66 GB/s single-threaded** (Xeon 8480+, 224 cores) against our **0.95** with CRC on / **1.09** with it off per thread — about 1.5–1.75× on a different CPU, not the 3.5× this row used to imply. Still not like-for-like |
| `ZipLLM` (research) | hub-scale dedup + delta | 54.1% across 43 TB, because 99.64% of hub models are fine-tunes. This is our Phase 3, published first by someone else |
| `DietGPU`, `nvCOMP` | GPU entropy decode | 250–600 GB/s and ~480 GB/s bracket lmz's 418. GPU decode is a commodity |

**One thing we have that none of them do:** they all assume streaming or
compression is better. We measure the machine and sometimes say no. That is a
real differentiator and it is shipped — with one honest amendment, that
filesystems have made a *ratio*-based version of this call for decades
(ZFS/LZ4 early abort). Nobody makes the **rate** call, which is the one that
can say *"on this machine your archive loads at 0.74× the speed of the plain
file"* — measured warm on this box, against 0.95× predicted, with about 20%
unaccounted and being chased.

**Distance to someone actually using it**, honestly:

- a laptop on a SATA SSD or a network mount — **one integration away**;
- vLLM cold start from S3 — **not close**, and the gap is native code plus cloud
  integration plus deployments behind their numbers, not arithmetic;
- as a diagnostic — **available now**, since `plan` costs the user nothing.

### The supply-side problem is mostly not a problem

It was listed as a blocker — *"the value needs the model to already be an lmz
archive, and nothing is"* — and that was overstated. It survives in one narrow
case and dissolves everywhere else.

**Where it dissolves: every local re-read.** Compress on first fetch and cache
it. The first load runs at normal speed, the compression happens off the
critical path, and every later load is faster. Nobody publishes anything and
nobody has to trust us in advance. With a **stdlib** codec there is not even a
dependency to install, so the ecosystem problem becomes a local cache decision
— which is the kind of decision this package already makes and can already
justify with a measured number. The router decides whether to build the cache
at all: on a fast NVMe it should not bother, and it already knows that.

**Where it is real: transfers.** You cannot send fewer bytes over a wire unless
the sender already has fewer bytes. Local caching does nothing for a first
download, so for §2b the supply side is genuine — but it splits the same way
§2b already splits it. On **your own object storage** you are the supply side
and there is no ecosystem to persuade. Only the **public hub** case needs
someone else to act, and that one was already filed as worth wanting and not
worth blocking on.

**The one real cost, which is not an ecosystem problem either:** a compressed
cache beside the plain file costs disk until the plain one is dropped, and
people will not let a tool delete out of their model cache. So the cache wants
to be opt-in and to say what it will use.

That reorders the roadmap: **`load_file()` → cache-on-first-fetch → object
storage** is the shortest path from correct to used, and it comes before the
phases below.

## 5. Roadmap

### Phase A0 — a codec that needs nothing installed *(§4b)*

Python 3.14 ships zstd in the standard library. Measured here on real BF16
weights: **21.1% saved, 0.77 GB/s decode single-threaded** and 1.69 at 8–16
threads where it stops scaling, against lmz's
32.9% and ~1.05 — so a stdlib codec captures **1.27× of the available 1.49×**,
about 85% of the win, with **no dependency at all**. It clears the gate
comfortably everywhere the gate matters: 2.3× on a 0.26 GB/s mount, 4.7× on a
gigabit link, overwhelmingly on a download.

That changes what this package *is*: standalone rather than a companion, with
lmz as an optional upgrade from 0.789 to 0.671 rather than a prerequisite. It
also makes Phase B real out of the box, since the compression needs nothing
installed.

The cost is honest: lmsluice would own a small container of its own — chunk
boundaries and an index. It stays inside the boundary rules as a second adapter
beside `lmzcodec.py`, and it would finally exercise a `Codec` protocol that
currently has one implementation and is therefore unproven.

### Phase A — be droppable *(days, and it gates everything else)*

`load_file(path) -> dict[str, torch.Tensor]`, matching safetensors' signature,
built on the tensor index that already exists and importing torch only if the
caller has it. Until this exists the measured wins are unreachable by anyone.

### Phase B — cache on first fetch *(§4b)* — **done, 2026-08-30**

`zstdcodec.py` and `cache.py`. Measured 1.80× from compression alone on a slow
link and 3.85× with the cache on local disk, byte-identical, with nothing
published and nothing installed. The gate decides whether to build it at all, so
a fast NVMe is told to keep the plain file. Numbers, and the decomposition of
the two effects, in `MEASURED.md`.

Left open: the decision samples with zstd but `build` may choose lmz, so a
verdict quoted before building understates the result when lmz wins the choice.
Harmless today — it errs toward not caching — but it should measure with the
codec it will use.


### Phase 0 — prove it where it pays *(days; mostly already possible)*

The evidence gap is embarrassing and cheap to close: **every measurement in this
repository was taken where the tool loses.** First win recorded today — 1.23×
over the 9p mount, byte-identical, five runs.

- Large models over the slow links already on this machine (9p, and a USB or SD
  device if one is to hand). The 151 MB win was only 1.23× against a predicted
  2.14× because at 0.3 s the fixed costs dominate; **the win needs size**, and
  showing that is part of the result.
- HTTP over a real network rather than localhost — and note that the 9p mount is
  a **sample of the sub-1-GB/s class**, not a target. It is a WSL2 artefact; the
  finding is the link rate and the gate, not the filesystem, and it must reach the
  README in that form.
- The **write path over a slow link**: measured at 0.239 GB/s written properly
  against a 1.5 GB/s encoder — a gate of **6.3×**, saturated, so the ratio is
  the only limit.

*Falsified if:* the win fails to grow with model size, or fixed costs dominate at
every size on slow links.

**Done, 2026-08-30 — and it passed.** Constant-ratio BF16 fixtures at 64 / 256 /
1024 MiB over the 9p link: loading is **1.26× → 1.21× → 1.45×** against a 1.49×
ceiling, so at a gigabyte the coded route reaches 97% of what the ratio allows,
and the plain route stays flat while the coded one climbs. Saving is **1.10× →
1.18× → 1.27×**. The win needs size and saturates by about a gigabyte. Full
numbers, and the two measurement traps that would have inflated them, in
`MEASURED.md`.

**Two findings that change later phases.** The write path leaves ~15% on the
table because encode and write do not overlap — roughly three quarters of the
encode time is exposed rather than hidden under the write, where the read path
pipelines properly. That is a concrete Phase 2 item and it is now the largest
single known inefficiency in the package. And the earlier 12.8× write gate was
wrong: it came from a badly-patterned write measurement, corrected above.

### Phase 0.5 — the download, where both ends are yours *(§2b)*

The best numbers in this document, on machinery that mostly exists, with no
ecosystem to persuade. Object storage to compute nodes, cluster distribution, CI,
appliance updates.

- End-to-end `lmsluice get` from real object storage over a real network, at the
  **directory** ratio rather than the shard ratio — which means the multi-file
  archive path, dedup included, not a single shard.
- Report bytes transferred beside seconds, because on a metered or shared link the
  first number is the one that is paid for.

*Falsified if:* the ranged-GET machinery cannot hold a real link's bandwidth, or
dedup across a real checkpoint directory does not reproduce the 64.6% figure.

### Phase 1 — order-aware delivery *(the differentiator)*

Let the consumer say what it needs and when, and deliver in that order.

- `stream(order=[...])` taking an execution order, not a file order.
- Time-to-first-tensor and time-to-Nth-layer as first-class outputs, beside GB/s.
- `plan` prices both, because they can disagree.

This is what turns a loader into a *facilitator*: the transport arranges itself
around the consumer's schedule instead of the file's layout.

### Phase 2 — the write path

Checkpointing is transport, the gate is `encode > write`, and it is **more
favourable than the read gate on every device**, because sustained writes are
slower than reads almost everywhere. Training checkpoints block training, so the
value is immediate and measurable.

- `put_model()` with the write gate measured per machine.
- Delta against the previous checkpoint — lmz already has delta chunks and
  nothing in this package asks for them.
- Overlap encode and write with training compute rather than stopping for it.

### Phase 3 — many models, not one archive

Everything here assumes one file. Real deployments have registries, fine-tunes
sharing a base, and checkpoints differing slightly from their predecessors.

- Dedup and delta **across** models, not only within one archive.
- Swap-in / swap-out timing for serving several models on one device.

### Phase 4 — device decode, conditionally

Keep what exists; extend it only where measurement says it pays — unified memory
first, since the upload stage that currently bounds it does not exist there.

## 6. What to stop

- **Stop expanding `gate.py`'s device table.** Every row a probe cannot fill is a
  one-sided bound, and more rows of those is not more knowledge.
- **Stop polishing the device route** until Phase 0 says where it pays.
- **Stop boundary and process work** unless it is blocking something. It is
  finished; it was worth doing; it is not the product.

## 7. What would change this plan

- A measurement showing the win does not grow with model size on slow links —
  that would make this a niche tool rather than a general one.
- Storage getting uniformly fast enough that the gate is lost everywhere. The
  trend is the opposite: models grow faster than consumer storage.
- A decoder an order of magnitude faster on the CPU, which would move the
  crossover up and widen the market rather than narrowing it — good news that
  would still not change the ordering above.
