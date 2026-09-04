# lmsluice — the model transport facilitator

Moves model weights from wherever they are to wherever they are needed, over
the route that is actually faster **on the machine it is running on**. It
decides by measurement rather than assumption, and it says what it decided and
on what numbers. Pure standard library, and nothing about the bytes changes:
every weight arrives byte-identical.

## Who this is for

The decision it makes turns on one comparison — is the decoder faster than the
link it is feeding from — and the two sides of that vary very differently. A
decoder varies by more than a factor of ten once you count what it actually
depends on — codec choice alone is 6.1× on one CPU, and thread count another 8×.
**The link varies by a factor of a hundred**, which is still the bigger term and
still the one that decides.

| link | GB/s | verdict | |
|---|---|---|---|
| NVMe, warm in the host cache | 6.2 | 0.74× — a modest tax | **measured** |
| **9p / network mount** | **0.26** | **1.45× — pays** | **measured** |
| NVMe, first touch | 2.4 | pays | *projected* |
| UFS 4.0 (phone flash) | 4.0 | pays | *projected* |
| SATA SSD | 0.55 | pays, ratio-capped | *projected* |
| eMMC / SD / USB | 0.30 | pays, ratio-capped | *projected* |
| 1 Gb/s network | 0.125 | pays, ratio-capped | *projected* |
| a download | 0.01 – 0.1 | pays, see below | *projected* |

**Two rows are measured end to end; the rest are the gate's arithmetic and are
marked as projections.** They are not measurements and should not be read as
any. The projection is `min(1/f, decode/link)` with decode = 5.95 GB/s measured
through this package's own transport, which puts the crossover at **5.95 GB/s of
link** — every projected row sits below it. The one measured row above the
crossover behaves as predicted in direction and is worse than predicted in
degree: 0.74× where the arithmetic says 0.95×, so about 20% is unaccounted and
is being chased.

So the value is very nearly a function of one variable: **how slow the link
is** — and every link below the crossover above is one where compression is
free or better.

**A correction, because this page used to say otherwise.** It claimed an archive
"costs you 6× on load time" on a fast NVMe. That figure paired the fastest plain
measurement with the slowest coded one, and the coded one used a chunk size that
reaches lmz's conditioned codec — which decodes **6.1× slower** than its field
split on the same data. With the chunk size this package now writes, the same
machine measures **0.74×**: still a tax on a warm local NVMe, but a modest one,
and it still saves 33% of the disk. `MEASURED.md` has both numbers and the
comparison that was wrong.

Note that 8.3 GB/s appears in `MEASURED.md` as the decoder's rate **from RAM**,
with no I/O in it. It is not a crossover and this page does not use it as one:
the crossover is 5.95, measured through the transport that actually does the
work.

**A second correction, still settling.** The destination buffer a load decodes
into cost more than the transport that filled it: `bytearray(n)` zero-fills,
which faults every page eagerly and writes a full pass of zeroes the decoder
immediately overwrites — 0.55 s against 0.31 s of transport on a 1.87 GB
checkpoint. It is now a private anonymous mapping advised for huge pages, which
is 4.5× cheaper and needs nothing installed. **The end-to-end figures on this
page predate that change and are pessimistic by up to ~2× on this box.** They
are left as they are rather than restated, because the replacements have only
been taken page-cache-warm and this page's numbers are cold-protocol ones; the
two are not interchangeable and quoting across them is how the 6× above
happened. No gate verdict moves either way — both routes paid the allocation
equally. `MEASURED.md` has the fault counts and the arithmetic.

**Measured, not asserted** — a 1 GiB BF16 model over a 9p mount, cache dropped
before every run, byte-identical. That mount is one sample of the sub-1-GB/s
class and not a target; the finding is the link rate, not the filesystem:

| | plain | coded | | runs |
|---|---|---|---|---|
| loading | 0.447 GB/s | **0.646** | **1.45×**, against a 1.49× ceiling | 5 |
| saving | 4.50 s | **3.55 s** | **1.27×** | 3 |

The win grows with model size and saturates by about a gigabyte; below a
hundred megabytes fixed costs eat most of it. `MEASURED.md` has the sizes, the
spread, and the two measurement traps that would have inflated both numbers.

## Downloading a model

The largest case, and the only one where the answer needs no measurement.

Over a 10–100 MB/s link the decoder is two orders of magnitude faster, so it
disappears entirely under the transfer and **you simply move fewer bytes** —
arithmetic, not a gate. It also pays twice, in bytes not transferred *and* in
load time not spent, at lmz's directory-level ratio rather than its shard-level
one, because a checkpoint directory ships the same tensors more than once:

    lmsluice get https://host/model.lmz ./model.safetensors

Ranged GETs with a connection kept alive per thread, decoded as they land, so
the archive never touches the disk in coded form. A server that refuses ranges
is fetched once rather than refused.

Exercised against a local range-serving server and a range-refusing one, both
byte-identical. **Not yet run over a real wide-area link** — the arithmetic
above is the link speed and the ratio, not a measurement of this path at that
speed.

## Compress on first fetch

The reason you do not have to convert anything first.

    lmsluice cache ./model.safetensors            # would it pay here?
    lmsluice cache ./model.safetensors --build    # then build it

The first load reads the plain file at its normal speed. Afterwards the model is
compressed into a local cache, and every later load takes the coded route —
measured at **1.80× from compression alone** on a slow link, **3.85×** with the
cache on local disk, byte-identical, with nothing published and nothing
installed.

**Nothing installed** is literal: `lmsluice/zstdcodec.py` uses the standard
library's zstd, so a machine with no lmz still gets a cache. lmz is preferred
where present because it compresses better; it is an upgrade rather than a
prerequisite.

**And it is a decision, not a policy.** On a fast NVMe the right cache entry is
the plain file, and it will tell you so rather than making every later load
slower. Nothing is cached unless asked, nothing is deleted, and an entry bound to
a file that has changed misses rather than serving stale weights.

## Buckets

    lmsluice get s3://my-bucket/model.lmz  ./model.safetensors
    lmsluice get gs://my-bucket/model.lmz  ./model.safetensors
    lmsluice get az://account/container/model.lmz ./model.safetensors

S3, Google Cloud Storage and Azure Blob, **with no SDK**. All three authorise an
HTTPS range GET with HMAC-SHA256 over a canonical form of the request, and
`hmac` and `hashlib` are in the standard library, so the whole of it is
`lmsluice/sign.py` — AWS Signature V4, Azure Shared Key, SAS tokens and bearer
tokens. `boto3` pulls botocore, s3transfer, jmespath, python-dateutil and
urllib3; on a phone, a CI container with no wheel cache or an air-gapped box
that is frequently the whole obstacle.

**All three have been read from for real**, anonymously, from public buckets —
S3, GCS and Azure, with concurrent ranges checked byte-for-byte against the
same bytes read contiguously (`MEASURED.md` has the objects and the 4.72 MB it
cost). Signed requests against a real store are the one thing still unverified:
they need a bucket and credentials.

**Signing is checked against each vendor's own published example**, not against
a fake I also wrote. A signature is byte-exact or worthless, and a wrong
canonical form produces a well-formed header that is always rejected with no
clue which of a dozen rules was misread — so the tests pin AWS's worked
`GET Object` signature to the hex digit and Microsoft's documented
string-to-sign byte for byte. Round trips then go through a local fake that
re-derives every signature it receives.

**Every command works on a bucket URL and none of them was changed.** `get`,
`info`, `load`, `stream`, `plan` and `bench` never asked what a source is —
they ask how fast it delivers — so an object store is a third implementation of
`pread` beside a local file and a web server. That is the test of the
abstraction rather than a claim about it. S3-compatible stores (MinIO,
Cloudflare R2, Backblaze B2, Ceph RGW, Wasabi) need no code of their own: set
`LMSLUICE_S3_ENDPOINT`.

**Credentials come from where each cloud already puts them** — environment,
`~/.aws/credentials` with profiles, a connection string, a SAS, `gcloud`'s
application-default refresh token — so a configured machine needs nothing new,
and **a public bucket needs nothing at all**. No flag anywhere takes a secret,
because argv is readable by every process on the machine out of `/proc`, and
every credential-bearing header and every signature in a URL is redacted from
every error. A SAS token *is* a signature carried in the query string, which
makes a URL itself a credential and error messages the place they leak from.

**The SDKs are extras, and narrower than they look.** `lmsluice[s3]`,
`[gcs]`, `[azure]` are for credential *sources* the standard library cannot
reach — instance-metadata and IAM roles, service-account JWTs needing RS256,
managed identity. The request is signed and sent by this package either way,
and each source reports its mode as `stdlib` or `sdk`.

**Uploading works too**, and is deliberately the second half:

    lmsluice put ./model.lmz s3://my-bucket/model.lmz

All three stores chunk a large upload the same way — begin, send numbered
parts, commit a manifest naming them. S3 calls it a multipart upload, GCS's XML
API implements the same one, and Azure calls the parts blocks and the manifest
a block list; what differs is two URLs and one XML vocabulary. A part-way
failure **cancels the upload** rather than leaving it, because the parts of an
abandoned multipart are stored, billed, and absent from a listing — a charge
whose cause is hard to find later.

Signing costs 12.7 µs per request for SigV4 and 5.9 µs for Shared Key
(9800X3D, Python 3.14). At the 4 MiB default chunk that is a ceiling of ~330
GB/s, which is to say free against any link a bucket is reached over — and the
number says where it would not be: very small chunks over a very fast one.

## Encryption at rest

A checkpoint on a shared filesystem, in a bucket, or on a laptop that leaves the
building is readable by anyone who can read the file.

    lmsluice seal --make-key key.bin              # 32 random bytes, mode 600
    lmsluice seal model.lmz model.sealed --key-file key.bin
    LMSLUICE_KEY_FILE=key.bin lmsluice get model.sealed out.safetensors

AES-256-GCM, authenticated, **with nothing installed**: OpenSSL's libcrypto is
reached through `ctypes`, the same way `cuda.py` reaches the CUDA driver. Any
Python that can open an HTTPS connection has already loaded it. Nothing here
invents cryptography — no keystream from `hashlib`, no XOR, no home-made
construction — and where no library can be reached, encryption reports `none`
and refuses to write a file that would look protected and not be.

**The key is always a path, never a value.** Not a flag holding a key and not an
environment variable holding one: on Linux any process can read another's
command line out of `/proc`, and an environment is inherited by children. The
key is never logged, never printed, and never written into the archive.

**It wraps the archive rather than living inside a codec.** Encryption changes
when the threat model changes, not when the format does, so it sits in the
transport layer as an envelope — which is also the only way it could cover an
lmz archive, whose container lmz writes and this project does not edit. One
mechanism, every codec, and no codec that mentions it; a test enforces that.

**Structure stays readable, and is still protected.** Tensor names, shapes and
the chunk index are in the clear, so `lmsluice info` works for someone who
cannot read the weights and a loader can plan a partial read before it can
decrypt one — tensorizer makes the same trade. They are covered by an HMAC
under a separate subkey, because an attacker who cannot read a weight could
otherwise still edit the index that says where the weights are. A wrong key is
refused when the file is opened; a tampered unit is refused by its tag and
named; nothing half-authentic is ever returned.

**What it costs, measured rather than assumed** (9800X3D, 16 logical cores,
OpenSSL 3.5.5, 403 MB BF16 checkpoint lmz-coded, warm page cache, best of 5):
**3% of load throughput on one fetch thread, 21% on sixteen.** The stage is
priced in `plan.py` like any other and `limited_by` will name `decrypt` where
it binds. The thread number is the interesting one and it is not about AES:
the same work run across processes instead of threads scales from 15 to 99
GB/s, so what caps a threaded fetch pool is the interpreter, not the cipher.
A sealed archive therefore asks for a *small* fetch depth where an unsealed one
wants a large one, which is the opposite of the usual advice and is why the
default is chosen after the file has said whether it is sealed.

The one thing encryption takes away is `mmap`: there is no arrangement of page
tables that decrypts, so a sealed archive can never take the zero-copy route
however fast the disk is.

**Writing one costs a second pass, and `lmsluice write --seal` says so first.**
The envelope wraps a finished archive, so the file is written and then read
back and rewritten sealed: both copies exist at once. That is the price of
covering an lmz archive without editing lmz, and the plan states both terms —
the extra wall clock as its own phase, and the disk high-water as a column of
its own — so a 70 GB checkpoint sees the cost before it starts rather than when
the volume fills.

## The one ratio

Compression is free on a path when the decoder is faster than the link it is
feeding from. Everything here follows from that. For N plain bytes and an
archive whose coded size is r times its plain size, with the stages overlapped:

    plain    N / link
    coded    max(rN / link, N / decode)
    speedup  min(1/r, decode / link)

The gate is the second term and **it does not mention r at all**. Compression
pays exactly when the codec beats the link; the ratio then decides how much,
up to a ceiling of 1/r. Writing is the same statement with `encode` in place
of `decode`, which is why the answer is often the opposite on the way out.

Three consequences, each of which is easy to get wrong:

- **A fast disk is a reason not to compress.** Above the gate the coded route
  loses, and by more the faster the disk is. A loader that always takes the
  compressed path is choosing to be slower on the machines that would
  otherwise be fastest.
- **The ratio cannot rescue a slow codec.** `min` is a floor, not a sum.
  Saving 40% of the bytes buys nothing where the decoder is already the
  bottleneck.
- **Expanding first is a third route, and it is far worse than both.** Without
  a tool like this, a user with a compressed checkpoint runs `decompress` and
  then opens the result — paying the link twice and a write in between. It is
  priced here so the comparison people actually face is visible.

## Where the verdict flips

The gate is a curve, and both sides of it are machines people own. With a
decoder running at D and an archive at ratio f, storage slower than D pays and
storage faster than D is taxed — and below D·f the win saturates at 1/f,
because the ratio is all there is left to win.

Taking this box's measured decode of **5.95 GB/s** through the transport, with
lmz's field-split codec at 1 MiB chunks and f = 0.673, the crossover sits at
about **6 GB/s of storage** — not the ~1 GB/s an earlier version of this page
gave, which used the conditioned codec's rate. All projections:

| storage | GB/s | projected verdict |
|---|---|---|
| NVMe Gen5 | 12.0 | tax, 0.50× |
| NVMe Gen4 | 6.34 | break-even, 0.94× |
| UFS 4.0 (phone) | 4.0 | **pays, 1.49× — saturated** |
| **SATA SSD** | **0.55** | **pays, 1.49× — saturated** |
| **eMMC / SD / USB** | **0.30** | **pays, 1.49× — saturated** |

So the same code and the same archive reach opposite verdicts, decided
entirely by the machine in front of them. A laptop with a SATA SSD gets the
whole of lmz's ratio as load speed with nothing but the CPU; the desktop this
was written on pays a modest tax for it — **0.74×**, not the 6× this page used
to claim. Neither is *the* answer, which is the reason the decision is made at
run time and not written down here.

Move the decoder instead of the disk and it flips again:

| decoder | GB/s | basis | against 6.34 GB/s NVMe |
|---|---|---|---|
| lmz CPU, 16 threads | 8.3 | decode only, from RAM | 1.31× |
| lmz CPU, through this transport | 5.95 | end to end | 0.94× |
| lmz CUDA, per-chunk table, RTX 5080 | 111 | decode only | free; 1.49× by the ratio |
| lmz CUDA, shared table, RTX 5080 | 418 | decode only | free; 1.49× by the ratio |
| a 2-CU display iGPU, **predicted** | **≥ 3.4** | `gate.py`, projection | ≥ 0.54×, a floor not a ceiling |

### The last row used to be an interval this page could not narrow

It said 4 – 36 GB/s, because the decoder's compute cost per byte was known from
**one** point on **one** card, and one point fits a compute-bound and a
bandwidth-bound reading equally well. The two agreed on the RTX 5080 that
produced them and differed 8× on a 2-CU integrated GPU — the class this project
exists for. `gate.py` carried both readings and refused to pick.

**lmz 1.3.0 separated them, and not by one reading being wrong.** Both are true,
and *which one binds is a property of the device*. The separation came from
measuring `k` across a grid sweep rather than at a single launch: a sweep across
residency is precisely what tells a latency-bound cost from a bandwidth ceiling,
because the two predict different slopes as blocks are added. `k` also moved,
230–330 → 217–248 lane-cycles per decoded byte, and *why* it moved is lmz's own
account in `cost_model()["provenance"]["supersedes"]`.

The defect was **not** that the old value was wrong, nor that its sweep was
bandwidth-bound. It was **mixed**, and one constant was fitted across rows
governed by different resources: at 64 threads a block the kernel is
compute-bound (245.7 GB/s against a 264 ceiling), and from 96 threads up it is
bandwidth-bound (417.3 against 791 at 384). One compute measurement and six
memory ones, averaged into a single interval. That explains what a simpler story
cannot — why the old interval's low end was nearly right, carried by the one good
row, while its high end was not. 217–248 is the same quantity taken only where
compute binds: a refinement of a badly-conditioned fit, not a contradiction, and
nothing about the kernel changed between them.

So the gate picks now, on the codec's published grounds rather than on our
inference:

| device | binding term | margin |
|---|---|---|
| RTX 5080 | **bandwidth** | 1.4× below its own compute ceiling |
| 2-CU iGPU | **compute** | 13.4× below its own bandwidth ceiling |

On a small integrated part the bandwidth reading is simply not live. That is
what collapsed the 8×.

### What is still not known, stated as plainly as the old refusal was

**A tighter interval is not a measurement.** Everything above was measured on an
RTX 5080. **Nobody has run a 2-CU device.** The row stays a projection, it stays
one-sided — `≤ 7.8` with no floor, because that device's occupancy is assumed
rather than read — and two things stay genuinely unverified:

1. **Whether decode stays linear in resident lanes down to 2 CUs**, or meets a
   cache, scheduler or driver wall first. The linearity holds over an 84× range,
   but all of it on one card and none of it near this size.
2. **How a unified-memory part behaves with the CPU contending for the same
   bus.** Every bandwidth figure here assumes the decoder has the memory system
   to itself, which on an integrated GPU it does not.

### Residency needs two numbers, and this page kept using one

lmz publishes the rule as `residency_formula`:

    blocks = floor(pool / shm) if shm <= cap else 0

The **cap** decides whether one block is *legal*; the **pool** decides how many
*fit*. On discrete NVIDIA parts they agree within a kilobyte — this card reports
101,376 and 102,400 — so conflating them is invisible there. They diverge on
integrated parts, which is the device class this project is for. This page has
now made that conflation twice, in opposite directions, and stated the 2-CU row
three different ways in one day: 4 – 36, then ≤ 7.8, then ≤ 1.9, and now what
follows.

**Every device fact in that row is now queried rather than assumed.** Through
Vulkan on the Windows side, since the adapter is not reachable from Vulkan inside
WSL2 at all — enumeration there returns a software rasteriser and nothing else:

| quantity | value | source |
|---|---|---|
| cap | 32,768 B | `maxComputeSharedMemorySize` |
| compute units | **2** | `VK_AMD_shader_core_properties`, and `activeComputeUnitCount` |
| threads per CU | 2,048 | 16 wavefronts × 2 SIMD × 64 wide |
| **pool** | **unobtained** | core Vulkan has no such query, and neither AMD shader-core extension — both present here, both queried — carries an LDS field |

So the row still gets **no compute term**, the same as any unprobed device. But
the pool is not unbounded below: **`pool ≥ cap` is forced**, because a unit that
could not hold one legally-sized workgroup could never schedule one. That is a
proof, not an estimate, and it turns the pessimistic substitution into a *floor*
where it had been serving as a ceiling.

The floor is best stated without a clock, because the clock is the weakest input
here — 2.2 GHz derived from an FMA measurement, not queried — and the compute
term is linear in it across the whole relevant range (bandwidth would only bind
above 16.8 GHz):

> **The floor is 1.55 GB/s per GHz of engine clock, and it exceeds the CPU
> decoder's 2.0 GB/s for any clock above 1.29 GHz.**

Every AMD integrated part of the last several generations boosts well clear of
that, so the claim is robust rather than contingent — and a reader can check it
against their own adapter without trusting our clock. At 2.2 GHz it is 3.4 GB/s,
1.7× the CPU decoder.

**The floor is conservative twice over, and both point the same way.** The pool
is taken as *equal* to the cap, its smallest legal value; and the cap itself is
under-reported, because on the reference card Vulkan says 49,152 where CUDA's
opt-in cap is 101,376 — Vulkan reports the non-opt-in limit. If this adapter's
real opt-in cap is 64 KiB rather than the 32,768 Vulkan gives, residency doubles
again. So the true value sits above this floor rather than scattered around it.

**That floor is arithmetic, not evidence.** It inherits every assumption above
it, and two are unverified at exactly this scale — whether decode stays linear in
resident lanes down to 2 CUs, and how a unified-memory part behaves with the CPU
on the same bus — while the clock is derived from an FMA measurement rather than
queried. It is a floor on the *model*, and the model is what is untested here.

**What would settle it is not a better query.** The pool is not exposed by
anything this adapter offers, so it can only be established by running the kernel
on the device. That makes it a measurement question rather than a query one, and
it is what a Vulkan backend would actually buy: not a tighter estimate, an end to
estimating.

What does not survive, either way, is reading any one desktop's NVMe as *the*
disk.

## Into VRAM

The second hop, built on the same two ideas. Two links now instead of one, and
the route decides how many bytes cross each:

    plain          max(N/L,  N/P)              N over both
    host-decode    max(fN/L, N/Dh, N/P)        fewer off disk, all of it over PCIe
    device-decode  max(fN/L, fN/P, N/Dd)       fewer over BOTH, decoded where it lands

`device-decode` is the one worth having: the only route that shortens both
links, and the only one whose decode is not competing with the host for memory
bandwidth. It is **built** — `lmsluice/devdecode.py` drives lmz's shipped
device decoder and adds the plane merge lmz stops short of — and on an fp32
checkpoint it is the fastest route into VRAM — **2.48× plain at the median of
nine cold runs** (1.39 against 0.56 GB/s; worst device run still 1.47× the best
plain one) — moving 0.427 of the bytes over PCIe.

**BF16 checkpoints decode on the GPU**, at **101 GB/s** against lmz's own
published 111 for the same kernel. That took one writer-side choice: lmz only
reaches its conditioned `CODEC_BF16C` — which no GPU kernel can read, because
its per-bucket streams have unequal lengths and the batch ABI needs them equal
— when a chunk holds a million elements or more. Compress at
`chunk_size=1 MiB` and it emits `CODEC_BF16` instead, which the kernel reads.
The cost is **0.2 points of ratio**: 32.9% saved becomes 32.7%.
`lmsluice.gpu_chunk_size()` returns the threshold and `lmsluice plan` says so
on any archive that needs it.

What still bounds the route is stream count and this box's cache, not the
decoder: an lmz stream is 8 lanes wide however large it is, so a small model
cannot fill a GPU (a 151 MB model gives 72 streams and 1 GB/s; a 1.87 GB one
gives 3574 and 101 GB/s). And "cold" on this machine is the Windows host cache
at 5.8 GB/s, fast enough that a 33% byte saving does not pay for the dispatch —
against genuinely cold storage the gate says device-decode wins by 1/f.
`MEASURED.md` has it in full.

```python
buf = m.to_device()                              # a CUDA allocation, filled
weights = torch.as_tensor(buf, device="cuda")    # zero copy; lmsluice imports no torch
```

The buffer carries `__cuda_array_interface__`, so torch, cupy and numba adopt it
without a copy and without lmsluice depending on any of them. It reaches the
device through the CUDA **driver** API via `ctypes` — the driver ships with the
GPU, the runtime ships with the toolkit, and requiring a toolkit to load a model
would repeat the mistake the probe exists to avoid.

Measured on this box, cold before every run, every route SHA-256 identical to
the source after copying back out of VRAM. **Nine runs each, median with the
best/worst spread**, because one of these stages varies by more than the
differences between them:

| model | route | median GB/s | spread over 9 runs |
|---|---|---|---|
| Llama BF16 1.87 GB | plain → VRAM | **3.10** | 2.53 – 3.16 (1.25×) |
| | host-decode → VRAM | **2.22** | 2.16 – 2.29 (1.06×) |
| | device-decode → VRAM | **1.36** | 1.12 – 1.71 (1.53×) |
| whisper fp32 151 MB | plain → VRAM | **0.56** | 0.47 – 0.62 (1.33×) |
| | device-decode → VRAM | **1.39** | 0.91 – 1.47 (1.62×) |

The device route has the widest spread of the three, and that is not noise about
nothing: it is the only route whose dominant stage — the host-to-device upload —
sits below its own crossover size, so it inherits that stage's variance.
`MEASURED.md` characterises the upload as a curve with a per-call overhead of
54.6 µs and a 0.87 MB crossover, and shows the defect getting **worse on a Gen5
link** and vanishing entirely on a unified-memory host.

PCIe measured 28.45 GB/s pinned against Gen4 x16's 28.8 theoretical, and 17.84
pageable — 1.59× apart, which is why the staging buffers are page-locked.

**This is where the plan is wrong for the first time.** By **29%** on the
host-decode route of an 8 MiB-chunk BF16 archive — predicted 1.05 GB/s against a
measured median of 0.75 over nine cold runs (0.65–0.79). It was first reported
as 30% from best-of-three and survives the stricter measurement unchanged. The
reason generalises: `max()` assumes overlapping stages are free, and they are not when
they share a resource. Here the CPU decodes into one pinned buffer while the DMA
engine reads the other, and both are on the memory bus — which lmz's decoder was
already saturating. On a unified-memory machine every stage shares that bus, so
what this box shows mildly, an Apple or Strix Halo part would show loudly.
`MEASURED.md` has the numbers and the reason it has not been "fixed" in
`plan.py`.

## Platforms

| | state |
|---|---|
| **Linux** | everything, on WSL2 |
| **Windows** | read path and cold measurement **run and verified** on NTFS; no CUDA path tried there |
| **macOS** | written, never executed — `fcntl(F_NOCACHE)`, and no CUDA at all |

Windows cold reads work through `CreateFileW` with `FILE_FLAG_NO_BUFFERING` —
the other half of the idea `posix_fadvise` gives on Linux, reachable from
`ctypes` with nothing installed. Running it there found that **`os.pread` does
not exist on Windows**, so every read in the package had been failing, not just
the cold measurement; the fix gives each thread its own descriptor rather than
locking a shared position, because a lock would turn the fetch stage's queue
depth into one. `MEASURED.md` has the numbers and the two further bugs that
only running it could surface.

## What it is

**A router.** `lmsluice plan` reads the machine's measured profile and prices
every route. `--link` and `--decode` answer the question a laptop cannot
measure — *would this pay over the link I will deploy on?* — with the asker's
numbers, labelled as theirs.

**A transport.** Two stages, sized independently, overlapped across a bounded
queue. Fetching wants queue depth; decoding wants the opposite, because lmz's
own measurement is that past two threads the interpreter between native calls
turns into contention. `lmz.decompress` reads and decodes inside one worker,
so one setting has to serve both. Splitting them is what this adds, and
`Report.limited_by` names which stage the run waited on.

**Three shapes of access**, because consumers differ. `map()` is an mmap and
moves nothing until touched — the fastest partial access there is, and it has
no coded form, which the tool says rather than faking. `load()` reads for real
into one buffer. `stream()` does the same under a memory ceiling, yielding
tensors as they land, for the machines where the model does not fit twice.

**Any source.** A local file and an HTTPS URL are two implementations of
`pread`, so the pipeline above them is unchanged, and an lmz archive can be
opened and decoded over range requests without landing on disk first. Servers
that refuse ranges are fetched once instead of refused. Connections are kept
alive per thread, because a handshake per megabyte costs more than the
megabyte.

**A probe that measures the right things.** Cold rather than warm, per storage
device rather than per machine, and against the archive that will actually be
read rather than a synthetic one — decode rate moves with chunk size and with
which codecs the encoder chose, so it is a property of the file, not the
build. Where the platform cannot drop a cache, the number is reported as warm
rather than passed off as cold.

## What this is not

**Running the model on the iGPU.** `vram/`'s roofline settles it: inference
decode does one multiply-add per byte of BF16 against about 4,132 needed,
*"short by a factor of four thousand, and no scheduling recovers it."* The
integrated GPU should decode the archive, not run the network.

**A second codec.** lmz produces bytes and turns them back into bytes; it does
not decide when. That line does not move because the device changed.

**A benchmark that flatters itself.** `lmsluice bench` runs both routes for real
and prints the predicted speedup beside the measured one. Twice so far the
plan has been checked this way, on a local disk and over HTTP, and it was
right to the second digit both times — see `MEASURED.md`. When it is wrong,
one command says so.

## Where the boundary is

**[`docs/boundary.md`](docs/boundary.md) is the canonical statement**: two
charters, the three-question test for any disputed piece of work, the ownership
table, five invariants, the codec interface, and the loan register. Read it
before adding a file to either tree. In one line — **lmz is the
compressor/decompressor; lmsluice is the AI-model transport facilitator.**


| | whose |
|---|---|
| the coded stream, its tables, its blocks, its index | **lmz** |
| a standalone GPU decoder with a stable ABI — CUDA, Metal, Vulkan | **lmz** |
| tensor → block addressing | **lmz** |
| which device and which route on this machine, and the probe that decides | **lmsluice** |
| dispatch, queue depth, staging buffers, overlap with I/O | **lmsluice** |
| sources — local, HTTP, and whatever comes next | **lmsluice** |
| adapters — safetensors-compatible open, PyTorch, llama.cpp | **lmsluice** |
| training residency across VRAM / RAM / SSD | `vram/` |

**The boundary is now published on both sides.** `docs/boundary.md` states it
here — two charters, a three-question test, an ownership table, five invariants
and a register of what is still on loan — and lmz has shipped the interface it
asked for. `ArchiveIndex.chunks()` and `.decode(ref, payload, out=)` are public
and, in lmz's own words, do *"No I/O, no threads"*, which is the invariant this
package needed; `lmz.capabilities()` declares which decoders can read an archive
instead of leaving the transport layer to reproduce an encoder threshold.

Every use of lmz is in `lmsluice/lmzcodec.py` and nowhere else, behind the
interface in `codec.py`, and a test walks the package source to keep it that
way. Three decode paths are probed in order and `Archive.route` names the live
one, so a benchmark can never quietly measure a fallback: `public` on a current
lmz, `direct` on one that predates `ArchiveIndex`, `mapped` on any lmz at all.
Since lmz 1.3.0 reached PyPI, **`pip install lmzip` gives the `public` path** —
it is no longer something you get only from a branch. The three-route probe
stays: a 1.2.0 consumer still exists, and a package that silently measured a
fallback would be worse than one that fails.

**The cost model is now read rather than restated, and that closed a real gap.**
`gate.py` used to hold lmz's expansion ratio, its sustained fraction of peak
DRAM, and its kernel's shared-memory shape as literals, because lmz published
the last of those only as prose in a provenance string. They are fields now, so
`CodecCost` carries them and the curve reads them. Those literals were correct
the day they were written, which is exactly the problem: a copied constant stays
right until it silently is not.

**The per-group table is the clearest thing the boundary has produced.** This
package's shared-memory finding and lmz's own kernel analysis arrived at the
same lever from opposite ends, and it is now priced in published fields: at the
same 192-thread block, the shared-table kernel's **640 bytes per group** fits
**7 blocks per SM**, where the per-chunk kernel's **5,760** fits **1**. A 9×
difference in one field, and the residency it buys is most of the gap between
418 GB/s and 111. Neither side had to tell the other a number to get there —
one published a shape, the other read it.

**What is still on loan is `merge.cu`** — a CUDA byte-plane transpose that is
lmz's work held here, because the device decoder returns plane-major bytes
rather than plaintext. It is registered with the interface change that retires
it and marked as such in its own header.

## What it does not do yet

Stated plainly, because a page that lists only what works is not a description.

- ~~**It does not hand you tensors.**~~ **It does now**, and this was the single
  biggest thing between the numbers above and anyone using them:

      -from safetensors.torch import load_file
      +from lmsluice.torchadapter import load_file

  Same call, same returned `dict[str, torch.Tensor]`, `device="cuda:0"`
  included. Torch lives in that one module and nothing else in the package
  imports it, so the core still needs nothing installed — a test walks the
  source and a second imports the whole core with torch forced to fail.

  A plain local file is **mapped, not read**, the same mechanism safetensors
  uses, and measures at parity with it (0.089 s against 0.085 s on a warm
  1.87 GB BF16 checkpoint, fresh process, n=5 median). An archive is decoded
  through the transport, which on that warm cache is 6.5× slower — the gate
  says so too, since warm the "link" is RAM at 22 GB/s and no CPU decoder
  competes with that. Cold, the verdict reverses; `MEASURED.md` has both.
- **vLLM can use it**, as `--load-format lmsluice`, through the registration
  mechanism vLLM actually has: a `BaseModelLoader` subclass reached from a
  `vllm.general_plugins` entry point, so installing the package adds the option
  and uninstalling removes it. It streams a window at a time rather than
  handing over a dict, because vLLM's interface is a generator so a large
  checkpoint never exists twice.

  **Checked against vLLM 0.28.0's released source**, not against the branch it
  was written from: the `BaseModelLoader` signatures, that `get_model_loader`
  instantiates with `load_config` positionally, that `LoadConfig.load_format`
  is `str | LoadFormats` rather than a closed enum, and that the CLI emits
  `metavar` instead of `choices` for exactly that reason — which is what makes
  a custom `--load-format` name parse at all. The registration, the
  abstract-method contract and the generator hand-off are tested against a stub
  built to that source.

  That check earned its keep: vLLM's `"Loading weights took N seconds"` line
  lives in `DefaultModelLoader`, not the base class, so this loader reported
  nothing until it was given one. Comparing the two load formats on vLLM's own
  metric would have had a figure on one side and silence on the other.

  **Verified against a real vLLM 0.28.0 install**, not only against its source.
  `get_model_loader` returns the class through vLLM's own plugin machinery, and
  vLLM logs `Registered model loader … with load format lmsluice` itself. On a
  1.87 GB BF16 checkpoint at temperature 0, three runs generate **identical
  token ids**: vLLM's own loader on the plain file, ours on the plain file, and
  **ours on the compressed archive**.

  **Cold, on a slow mount, it is 3.60× faster** than vLLM's default loader
  (median, n=5, distinct never-before-read copies, vLLM's own reported
  weight-loading time). That figure is compound and this page will not quote it
  bare, because 3.60 exceeds the 1.48× that 0.674-of-the-bytes can buy. A third
  arm separates it: **2.62× is the transport** — parallel reads against mmap
  page-faults on a high-latency mount — and **1.38× is the compression**, just
  under its own ceiling, which is the good case because it means the decoder
  keeps up.

  Warm on local NVMe the order reverses: vLLM's loader 0.20 s against our 2.90 s
  on the archive. That is the gate's prediction, measured inside an inference
  server rather than argued — warm, the link is RAM and no CPU decoder competes.

  Two flags are needed to run vLLM on WSL2 at all, neither ours:
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (vLLM disables pinned memory there by default,
  and the engine dies with "UVA is not available" without it) and
  `VLLM_USE_FLASHINFER_SAMPLER=0` (flashinfer JITs a sampling kernel and its
  arch check rejects an sm_120 card).
- **It does not speak object storage.** Plain HTTP with range requests, no S3,
  GCS or Azure, no auth. Which is awkward, because the download is the case with
  the best arithmetic in this repository.
- **It is not packaged.** No PyPI name, no remote, no version anyone can install.
- **The stdlib codec does not scale with threads.** Measured: one thread
  2.02 GB/s, sixteen 1.69. It caps near 2 GB/s however many cores are present,
  which is fine below the gate and a reason to prefer lmz above it.
- **The decoder is Python-bound at about 1 GB/s.** Fine below the gate, which is
  where this is aimed, and not competitive above it.
- **One machine of evidence.** No WAN, no object storage, no cluster, no second
  GPU, no Apple silicon. Every number names its box for that reason.

Where it stands against the alternatives is in
[`docs/competition.md`](docs/competition.md) — six lanes of competing products,
their published specs, and the three places the field has already taken ground
this repository was counting on. What would change the plan is in
[`docs/strategy.md`](docs/strategy.md).

## Status

Runs. 66 tests, no dependencies beyond the standard library and lmz.

```
python3 tests/test_lmsluice.py       # builds its own fixtures; skips what needs lmz
./lmsluice-cli doctor                # what is measured and what is active
```

What exists:

- `lmsluice/transport.py` — the two-stage overlapped pipeline, bounded by a queue
- `lmsluice/plan.py` — the routing arithmetic, and the margin that stops a
  decision being made on noise
- `lmsluice/probe.py` — cold reads, sustained writes, and decode measured on the
  real archive with the thread count swept rather than assumed
- `lmsluice/source.py` — local files and HTTP range requests behind one `pread`
- `lmsluice/codec.py` — what this package needs from *a* codec, as an
  interface rather than as lmz
- `lmsluice/lmzcodec.py` — the lmz adapter: the only module that imports
  lmz or knows a codec ID
- `lmsluice/archive.py` — runs, coalescing and placement: the I/O half
- `lmsluice/model.py` — `map` / `load` / `stream` / `to_device`, one API over
  every route
- `lmsluice/cuda.py` — the CUDA driver through ctypes: device buffers,
  pinned staging, streams, and `__cuda_array_interface__` for zero-copy handoff
- `lmsluice/cli.py` — `probe`, `plan`, `info`, `bench`, `get`, `write`, `doctor`
- `lmsluice/gate.py` — the device-side model: decode rate bounded from two
  sides, the codec's `k` carried as an interval with its provenance, and still
  runnable with no dependencies on a machine that has nothing installed
- `probe/igpu-bench.ps1` — the Direct3D 11 compute benchmark that feeds it,
  measuring whether a given device *could* decode

[`docs/strategy.md`](docs/strategy.md) is where this is going and why — the
argument that the value is nearly a function of link speed alone, what that
says about the GPU work, and the phased plan with what would falsify each step.
[`docs/boundary.md`](docs/boundary.md) is the division against lmz.
[`docs/transport-handover.md`](docs/transport-handover.md) is the full record
of how this was built and measured — the design decisions that were not
obvious, the two corrections that changed it after it was written, the traps,
and what each open item is blocked on.

What is blocked on lmz, re-derived against `main` rather than carried forward.
The v7 item that used to head this list has moved rather than vanished, and the
distinction is narrow enough to be worth stating exactly:

- **lmz's kernel accepts a shared frequency table** — `gpu.decode_batch_dev`
  takes one as `hdr` — so the *capability* is shipped and is no longer lmz's to
  add.
- **But a `--shared-tables` archive does not decode on the GPU today**, through
  this package or any other. `devdecode.py` refuses `CODEC_SPLIT_ST` chunks and
  passes `header=None`, so supplying the table is **our** work, not lmz's. It
  has left the list below because it is no longer blocked on someone else, not
  because it works.

1. **A device decode that returns plaintext.** `gpu.decode_batch_dev` takes
   device pointers, a caller's stream and a shared table, and returns
   plane-major bytes; the plane merge is still this package's to do. Retiring
   `merge.cu` is what closes it.
2. **The Metal port**, written but never run — `lmz/scratchpad/gpu/metal/`.
3. **Vulkan.** No longer merely wanted: the cycles model in `gate.py` needs a
   device's clock and shared-memory budget, and Direct3D 11 can report
   neither — its shared-memory figure is fixed by the API rather than by the
   hardware. Vulkan reaches the AMD and Intel iGPUs that make the difference
   between a one-sided bound and a real prediction.

When any of those lands, it enters here as another codec rate in the profile
and another row in the plan. Nothing above `plan.py` changes, which is the
point of putting the decision in one place.
