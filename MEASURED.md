# Measured — the box these numbers came from

*Conditions first, because two of the figures below moved by more than 2×
when a BIOS setting changed, and a bandwidth number without its memory clock
is not a measurement.*

## The machine

| | |
|---|---|
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads, 96 MB V-Cache |
| Board | ASUS PRIME B650M-A AX6 II, BIOS 9021 (2025-03-06) |
| Memory | 2 × 16 GB Team Group UD5-6000, **DDR5-6000 with EXPO I enabled** |
| dGPU | NVIDIA GeForce RTX 5080, 16 GB GDDR7, driver 610.57.01 / CUDA UMD 13.3 |
| iGPU | AMD Radeon Graphics, PCI `0x13C0` — Granite Ridge, **2 RDNA2 CUs**, 486 MB UMA carve-out |
| OS | Windows 11 with WSL2; the probe runs on the Windows side |

**EXPO is load-bearing and was off until 2026-08-26.** Every bandwidth figure
taken on this box before that date was taken at DDR5-4800 JEDEC defaults
against a 6000-rated kit, and the difference is not a rounding error:

| iGPU streaming read | at DDR5-4800 | at DDR5-6000 |
|---|---|---|
| 64 MB working set | 49.7 GB/s | 59.1 GB/s |
| 256 MB | 50.3 GB/s | 59.3 GB/s |
| 384 MB | **22.2 GB/s** | 59.4 GB/s |
| 1 GB | 23.5 GB/s | 59.5 GB/s |
| 2 GB | — | 59.1 GB/s |

At the JEDEC clock the iGPU appeared to fall off a cliff once the working set
outgrew its 486 MB carve-out, and the obvious reading — that the carve-out is
the limit — was wrong. At the rated clock the cliff does not exist: the iGPU
holds **59.1–59.5 GB/s flat across a 32× range of working set**, which is 62%
of the DDR5-6000 bus and *more than the 16-thread CPU manages out of cache*.
Raising the UMA frame buffer would have been a wasted trip to the BIOS.

## The gate numbers

Streaming coalesced reads and register FMA chains, D3D11 compute,
`probe/igpu-bench.ps1`, 512 MB buffer:

| | read GB/s | FP32 GFLOP/s | FLOP per byte |
|---|---|---|---|
| RTX 5080 | 822 | 53,691 | 65 |
| Radeon iGPU, 2 CU | 59.4 | 567 | 9.5 |
| CPU, 16 threads | 54.3 | — | — |

Two sanity checks that the harness is measuring what it claims. The 5080's
822 GB/s is 86% of its 960 GB/s theoretical and its 53.7 TFLOPS is 96% of
peak — both where a clean streaming benchmark should land. And the iGPU's
567 GFLOP/s is within 0.4% of 2 CU × 64 lanes × 2 × 2.2 GHz, which is the
arithmetic, so the FMA kernel is compute-bound as intended.

**The CPU column decays with working set and the iGPU column does not**,
which is the 96 MB V-Cache and is worth stating so nobody quotes the wrong
row:

| working set | CPU GB/s | iGPU GB/s |
|---|---|---|
| 64 MB | 112.9 | 59.1 |
| 128 MB | 111.5 | 59.4 |
| 256 MB | 70.1 | 59.3 |
| 512 MB | 58.5 | 59.4 |
| 1 GB | 54.3 | 59.5 |

Model weights are gigabytes, so **54.3 GB/s is the CPU row that matters** and
the 112 is an artefact of a buffer that fits in cache. The .NET array cap
means the CPU baseline tops out at a 1 GB buffer; that is already well clear
of the cache, so the steady state is unaffected.

## What the decoder actually achieves

Not measured here — carried from lmz, whose own benchmarks produced them:

| | GB/s | source |
|---|---|---|
| lmz CPU decoder, 1 thread AVX2 | 0.48 | `lmz/scratchpad/gpu/README.md` |
| lmz CPU decoder, 4–8 threads | **2.0** | `lmz/README.md` — memory-bus-bound, not CPU-bound |
| lmz CUDA decoder, per-chunk table | 111 | `lmz/docs/gpu-residency-handover.md` |
| lmz CUDA decoder, shared table | **418** | same, re-measured after the 580 → 610.88 driver update |
| NVMe, 64 KiB reads at QD64 | 6.34 | `vram/probe` |
| PCIe Gen4 x16 | 28.8 | `vram/probe` |

## Conditions and departures

- The probe is a **streaming read** benchmark. It is an upper bound on what a
  decoder can be fed, not a decoder benchmark. The two are related by the
  FLOP-per-byte gate and nothing stronger.
- FP32 FMA is a **proxy** for rANS decoding, which is integer work with
  dependent shared-memory loads. It is the right order of magnitude and the
  wrong unit; treat the estimated decode rates in the README as a reason to
  measure, never as a result.
- **The one decode number here does not identify which resource bound it.**
  418 GB/s on the 5080 is simultaneously 84% of that card's bandwidth bound
  and exactly its compute bound at `k = 128` FLOP-equivalents per decoded
  byte. A single point on a single device cannot separate the two, and the
  separation is worth 8× on a small iGPU. `../gate.py` carries both readings
  as an interval rather than picking one; pinning `k` needs a decode
  measurement on a low-FLOP/byte device, or a sweep that moves one axis
  (clocks against memory clocks, or a stream with a different `f`) on this
  one.
- Decode timings quoted anywhere in this tree should be **medians of several
  runs**. A single decode on this box varied by 40% between runs while other
  things were running.
- The 780M, Apple, Strix Halo and phone rows in the README are **arithmetic
  from public specifications**, not runs — no Radeon 780M and no Apple
  silicon has been through this probe. Earlier they were compute-scaled from
  the measured CUDA kernel, which additionally assumed that kernel was
  compute-bound; `../gate.py` no longer assumes it, and the rows are now
  intervals.

## How to re-run it

    powershell -ExecutionPolicy Bypass -File probe\igpu-bench.ps1
    powershell -ExecutionPolicy Bypass -File probe\igpu-bench.ps1 -SizeMB 512 -Reps 25
    powershell -ExecutionPolicy Bypass -File probe\igpu-bench.ps1 -Adapter 1

Nothing to install: it drives Direct3D 11 through `d3d11.dll` and compiles its
HLSL at runtime with `d3dcompiler_47.dll`, both present on every Windows box
since 8.1, via inline C# that PowerShell compiles itself.

## The lmsluice runtime, measured — 2026-08-30

*Everything below is this box on this day. None of it is compiled into the
tool: `lmsluice probe` takes the same numbers on whatever machine it runs on, and
a rate it could not take is reported as absent rather than defaulted.*

### The path, at four temperatures

Same 1.87 GB BF16 safetensors file, `posix_fadvise(DONTNEED)` between reads,
1 MiB requests:

| | GB/s | what it is |
|---|---|---|
| first touch, ever | **2.38** | the only genuinely cold number |
| cold to the guest, at depth 16 | 6.38 | the Windows host's cache under the vhdx |
| cold to the guest, sequential | 6.19 | same, one request at a time |
| warm in the guest page cache | 33–40 | not a disk measurement at all |
| sustained write, fsync included | 1.18–1.46 | what a checkpoint save costs |

**The gap between rows one and two is a trap with a general shape.** Dropping
the guest page cache does not drop the hypervisor's, and on WSL2 the vhdx sits
on NTFS with Windows caching it. The same file measured 2.38 GB/s on first
touch and 6.38 GB/s an hour later — a 2.7× error, in the direction that makes
compression look worse than it is. The same applies to any VM, any container
on a host filesystem, any network filesystem with a client cache, and any
drive with DRAM on it.

`lmsluice probe` detects this by reading one range cold twice and comparing, and
reports `layered`. What it **cannot** do is rule it out: the test only fires
when the lower cache was itself cold, so a file the host already holds gives
two equally fast reads and no detection. `layered` is therefore True or None
and never False, and the cold rate is an upper bound wherever it stayed None.

### The codec, on this machine and this archive

lmz 1.2.0, `native:avx2`, zstd 1.5.7 from the standard library, 16 threads
available. Measured on the archive that would actually be read, payloads
pre-fetched so only the codec is timed, median of three at each thread count:

| | GB/s of plain bytes | threads |
|---|---|---|
| decode, 8 MiB blocks (default archive) | **1.03–1.07** | 8–16 |
| decode, 64 KiB blocks (`--mapped`) | 0.83–0.89 | — |
| decode, one thread | 0.39–0.48 | 1 |
| encode, whole 1.87 GB file, default blocks | 1.25 | — |
| encode, `--mapped` | 0.15 | — |

Ratio 0.671 on this model — 32.9% saved.

**Two things the sweep found that are worth carrying.** Passing the payload as
`bytes` rather than as a `memoryview` slice measured 23% faster into lmz's
decoder, and CRC verification costs about 40% of decode. Neither is a reason
to change anything here — the checksum is why the bytes can be trusted — but
both are large enough that a decode rate quoted without saying which was in
force is not a measurement.

**The thread count is swept, not assumed.** It came out at 8–16 for 8 MiB
blocks and lower for small ones, which matches lmz's own finding that below
512 KiB the interpreter between native calls turns into GIL contention. On a
free-threaded build or another platform it will land somewhere else, which is
why it is measured per archive rather than written down.

### The gate, here

| direction | codec | link | gate | verdict |
|---|---|---|---|---|
| read, local disk | 1.05 decode | 6.38 | **0.16×** | a 6× tax |
| read, local disk, truly cold | 1.05 decode | 2.38 | 0.44× | still a tax |
| write, local disk | 1.25 encode | 1.46 | 0.86× | plain is faster |
| read, localhost HTTP | 1.05 decode | 1.26 | 0.83× | a tax, narrowly |
| read, a 1 Gb/s link | 1.05 decode | 0.125 | 8.4× | free; 1.49× by the ratio |
| read, NVMe → VRAM, CUDA | 418 decode | 28.8 PCIe | 14.5× | free; 1.49× by the ratio |

The last two rows are arithmetic on rates measured elsewhere, not runs on this
box: the link rate in the first is given, and the decoder in the second is
`lmz/docs/gpu-residency-handover.md`'s. They are what `lmsluice plan --link` and
`--decode` print, and they are labelled as given rather than measured wherever
they appear.

### Checking the prediction against a real run

The point of `lmsluice bench`. Both routes run for real, best of three, and the
predicted speedup is printed beside the measured one.

| source | predicted | measured | off by |
|---|---|---|---|
| local disk, 1.87 GB model, cold each rep | 0.16× | **0.16×** | 0% |
| localhost HTTP, same model, best of two | 0.83× | **0.83×** | 0% |

Two independent checks on two kinds of source. That is two points, not a
validation, and the interesting case — a link slow enough that the coded route
wins — has not been run on this machine because this machine does not have
one. Until it is, the winning half of the table is arithmetic.

**Localhost HTTP is not a network**, and the row above says so twice over: it
delivered 1.26 GB/s of coded bytes with the fetch stage saturated, which is
Python's HTTP stack rather than any link, and it is *faster* than this box's
decoder, so the gate correctly called it a tax. A real 1 Gb/s link is ten
times slower and lands on the other side of the gate.

### Conditions and departures

- **Take the memory clock before the bandwidth**, unchanged from the section
  above: EXPO was off on this box until 2026-08-26 and it moved the iGPU
  figure by 2.7×.
- **Decode timings swing 30–40% run to run** on a desktop with other things
  open. Everything above is a median or a best-of-three and is labelled which.
  The routing decision carries a 1.15× margin for exactly this reason: below
  it, the answer is "too close to call" and the plain route wins the tie.
- **The encode rate depends on the sample size.** A 128 MiB sample measured
  0.53 GB/s at r=0.787 where the whole 1.87 GB file gave 1.25 GB/s at r=0.671,
  because a small sample amortises lmz's startup over less work and finds less
  redundancy. `lmsluice probe --encode` should be given a large sample or its
  answer read as a lower bound.
- **The write rate includes `fsync`.** Without it the same test reports
  5–6.5 GB/s, which is the page cache and would make every compressed write
  look like a win.
- **No GPU decode has been run through this tool.** `lmz.gpu` exists and is
  measured in lmz's own documents; nothing here has driven it yet, so the
  CUDA row above is lmz's number in this tool's arithmetic and not a
  measurement of the two together.
- **One model, one dtype.** Every figure is BF16 weights from a single 1.87 GB
  checkpoint. Ratio, and therefore the ceiling on any win, moves with dtype —
  lmz's own results table has FP8 at 17% and Q4_K at 5%, where the ceiling is
  1.05× and nearly nothing is available to win.

### How to re-run it

    ./lmsluice-cli probe <a model or archive>     # measures this machine
    ./lmsluice-cli doctor                         # what it found
    ./lmsluice-cli bench <archive> --against <plain> --cold --reps 3
    python3 tests/test_lmsluice.py                # 33 tests

The probe writes its profile to the platform cache directory and refuses to
use one whose fingerprint says it came from a different machine, so a home
directory that travels does not carry the wrong numbers with it.

## The RAM → VRAM path, measured — 2026-08-30

*Added when the device path was built. Same box, same fixture. The CUDA rows
in earlier sections were lmz's numbers quoted in this tool's arithmetic; these
are this tool driving the hardware.*

### The link

`cuMemcpyHtoDAsync` to an RTX 5080 (sm_120, 17.1 GB), driver 610.88, through
the CUDA **driver** API via ctypes — no toolkit, no runtime library, no torch.
Median of three, 128 MiB per copy:

| | GB/s | |
|---|---|---|
| pinned (`cuMemHostAlloc`) | **28.45** | 98.8% of Gen4 x16's 28.8 theoretical |
| pageable | 17.84 | the driver stages it through its own pinned buffer |

Pinned is **1.59×** pageable here, and it is also the only form that can
overlap, so the staging buffers are pinned and the profile carries both numbers
rather than one called "PCIe".

### The path, cold, best of three

1.87 GB BF16 checkpoint, cache dropped before each run, 128 MiB double-buffered
staging windows cut on block boundaries:

| route | predicted | measured | |
|---|---|---|---|
| plain → VRAM | 3.01 GB/s | **3.35 GB/s** | read-limited |
| host-decode → VRAM | 1.05 GB/s | **0.81 GB/s** | decode-limited, 23% short |
| device-decode → VRAM | — | **not built** | see below |

Every route verified **SHA-256 identical** to the source checkpoint after
copying back out of VRAM.

*The 1.05 GB/s prediction is a retired figure and is kept because this is a
record of a prediction against its own outcome. It was `CODEC_BF16C`'s rate;
the archive here is an 8 MiB-chunk one, so the prediction was right about
**this** archive and wrong as a statement about lmz. See "Retired figures".*

### The first place the plan is materially wrong, and why

The ratio of coded to plain into VRAM was predicted at 0.35× and measured at
0.24× — **30% off**, where the two host-RAM predictions were right to the
second digit. The error is entirely in the host-decode route, which came in
23% below its predicted rate, and the cause is a limit of the model rather than
a bug in it:

**`max()` assumes overlapped stages are free, and they are not when they share
a resource.** On the storage path the stages use different hardware — the disk
fetches while the CPU decodes — so overlapping them costs nothing and the model
holds. On this path the CPU decodes into one pinned buffer while the copy
engine DMAs out of the other, and **both are on the memory bus**. lmz's decoder
is already memory-bus-bound at about 2 GB/s by its own README, so bandwidth
taken by the DMA comes straight out of the decode.

The general form, which is what transfers to another machine: the pipeline
reaches `max(stages)` when the stages bind on different resources and degrades
toward `sum(stages)` as they bind on the same one. Two stages that both saturate
memory bandwidth do not overlap, whatever the queue between them says. A
unified-memory machine has this everywhere — there is one bus and every stage is
on it — so the discrepancy measured here is the *mild* version of what an
Apple or Strix Halo part would show.

This has not been corrected in `plan.py`, deliberately: the fix is a contention
term, and one measurement on one machine is not enough to fit one. It is
recorded here, and `bench --to-device` reproduces it in one command.

### What is not built

**device-decode** — copy the *coded* bytes to VRAM and decode them there — is
priced by `vram_plan` and reports `unknown`, because the rate is unmeasured and
the path is unbuilt. It is the route worth having: it is the only one that
shortens **both** links, and against a 111 GB/s device decoder it should be
ratio-capped at 1.49× rather than losing.

What blocks it is not CUDA. `lmz.gpu` is available on this box
(`cuda:NVIDIA GeForce RTX 5080 sm_120 84 SMs`, already built, nothing written
into lmz), and its ABI has the right entry point —
`lmz_gpu_decode_batch_dev(hdr, streams, offsets, nstr, plane, out, stream, tpb)`
takes device pointers and does not synchronise. What is missing is the layer
that can tell it *which* streams: feeding it requires decomposing a chunk into
its per-plane rANS streams, and lmz exposes decoding a file and reading a byte
range, neither of which is that. lmz's own `gpu-residency-handover.md` names
this gap in the same terms — *"nothing can yet ask an archive where a tensor's
blocks are"* — so it is lmz's to close, and it is one piece of work rather than
research.

The Python entry point that does exist, `lmz.gpu.decode_batch`, is host-to-host
and PCIe-bound by construction. Routing through it would send the coded bytes
up and the plain bytes back down, which is slower than every route above.

### Conditions

- Pinned staging is two 128 MiB buffers. Page-locking is a kernel operation and
  a large one can fail; the size is a parameter (`to_device(window=)`) and the
  default is a compromise, not a measurement.
- The primary context is *retained*, not created, so a buffer handed to torch
  through `__cuda_array_interface__` is adopted with no copy. A second context
  would have made every handoff a copy and hidden it.
- The driver is loaded in a child process first. A half-removed driver leaves
  `libcuda.so.1` on disk with a faulting initialiser that kills the process on
  `CDLL`, with no exception to catch.
- **No second GPU, no second architecture, no non-NVIDIA device.** Everything
  above is one RTX 5080. The AMD and Intel paths need Vulkan, which does not
  exist in lmz yet, and Apple needs Metal, which is written and never run.

## Windows, actually run — 2026-08-30

*Every earlier section of this file is Linux under WSL2. This one is the
Windows side of the same machine, reached by invoking `python.exe` from WSL.
It exists because "the code is standard library, so it should run on Windows"
turned out to be false, and the only way to find that out was to run it.*

Python 3.12.11, NTFS, **512-byte sectors** (queried, not assumed), 128 MB
sample, the same `lmsluice` read path as Linux.

| | GB/s | |
|---|---|---|
| cold, `FILE_FLAG_NO_BUFFERING` | **3.51** | what the router now uses |
| warm, buffered through the fallback read path | 3.17 | |

Cold reads **are** available on Windows after all. `CreateFileW` with
`FILE_FLAG_NO_BUFFERING` opens a handle that never touches the cache, which is
the other half of the idea `posix_fadvise` provides on Linux, and it is
reachable from `ctypes` with nothing installed. Verified to return the file's
real bytes at offset 0 and at 64 MiB, and to hold its rate across repeats
rather than climbing the way a cached read does.

### Three bugs that only running it could find

**`os.pread` does not exist on Windows.** Not the cold path -- *every* read in
the package, which failed with `AttributeError` before reaching any of the
interesting logic. The fix cannot be a lock around seek-then-read, because that
would serialise the fetch stage and silently turn its queue depth into one;
each thread gets its own descriptor and therefore its own file position, which
is the independence `pread` gives, bought with a handle per thread.

**The first version of that fix had a race.** It claimed the constructor's
descriptor by testing a list outside the lock, so several threads starting
together all took the same one -- reintroducing the shared file position, only
under concurrency. Caught on Linux by a test that forces the no-`pread` path,
which is now part of the suite: the fallback would otherwise only ever run on
the platform nobody tests on, which is how it came to be missing in the first
place.

**`INVALID_HANDLE_VALUE` is -1, and the handle arrives unsigned.** Declaring
`CreateFileW.restype = HANDLE` returns `0xFFFFFFFFFFFFFFFF` on failure, so
`== -1` never matched and a missing file produced a reader whose every read
failed confusingly instead of the clean `None` the caller checks for.

### The condition that limits this measurement

Cold (3.51) came out *above* warm (3.17), which on Linux would mean something
was wrong. Here it does not: the warm number goes through the Python read path
and the unbuffered one is measured with a copy out of an aligned buffer, and on
this volume **both land near the Python read ceiling of about 3.5 GB/s**. So
what this sample shows is that at 128 MB the disk is not the binding
constraint on Windows -- the interpreter is. A larger sample on a slower volume
would separate them; that has not been run.

`read_at` copies by default for exactly this reason. Discarding the bytes into
a reused aligned buffer measured **5.69 GB/s against a 3.16 GB/s warm read on
the same file** -- reporting the cache as slower than the disk -- because it
skipped the allocation every other rate in the profile pays. A rate that is not
measured the way the pipeline will actually be fed is not comparable to the
decode rate it will be divided by.

### Still not run anywhere

macOS. The `fcntl(F_NOCACHE)` path remains written and unexecuted, and the
Apple decoder is lmz's unrun Metal port. Nothing here has touched Apple
silicon.

## device-decode, built and measured — 2026-08-30

*Coded bytes cross PCIe and become plaintext in VRAM. Built after the host
routes; `lmsluice/devdecode.py` and `lmsluice/merge.cu`. Every number below is
this box, best of three, cache dropped before each run, verified SHA-256
identical to the source after copying back out of VRAM.*

| model | plain → VRAM | host-decode → VRAM | device-decode → VRAM |
|---|---|---|---|
| whisper_tiny, fp32, 151 MB, f=0.427 | 0.61 | 0.46 | **0.99–1.65** |
| Llama 1.1B, BF16, 1.87 GB, f=0.671 | 2.78 | 0.64 | **refused, see below** |

On the fp32 case device-decode is the fastest route — roughly **2× plain** —
and only **0.427 of the plain bytes cross PCIe**. That is the shape the
arithmetic predicted: the only route that shortens both links.

### It wins on the link, not on the decoder, and that is not what was expected

The device kernels measured **1.87–2.37 GB/s of plain bytes**, against the
111 GB/s lmz reports for the same kernel. The gap is this package's dispatch,
not lmz's decoder, and it was worth chasing because the answer is structural:

**An lmz stream is 8 lanes wide whatever its size.** 8 interleaved rANS states
means one stream occupies a quarter of a warp, so what fills a GPU is stream
*count*. An 8 MiB-block archive of a 151 MB model yields 19 chunks × 4 planes =
**72 streams ≈ 576 lanes on an 84-SM card**, which is close to idle. lmz's
111 GB/s came from a far larger batch and says so.

**More chunks does not fix it, because dispatch takes over.** The same model in
64 KiB blocks gives 2305 chunks and 9220 streams — and ran *slower*, at 0.60
GB/s. Batching every chunk's merge into one launch instead of one launch each
moved that only to 0.71, which rules the merge out: what remains is the
host-side cost of 2305 separate `pread`s and assembling 9220 payload slices in
Python before anything reaches the device.

So the route as built is bounded at about 2 GB/s by dispatch on one side and
stream count on the other, and it still wins — because at f=0.427 it is moving
less than half the bytes. **The 111 GB/s figure is not reachable through this
implementation**, and quoting it as though it were would be exactly the kind of
carried number this file exists to stop.

What would fix it, in order: assemble the stream blob without per-chunk Python
slicing; issue the reads through the existing transport rather than one at a
time; and pick a block size in the middle of the two extremes measured here,
since neither end is good. None of that is research.

### The flagship case cannot use it at all

A BF16 checkpoint is refused, all 224 chunks, and falls back to the host route
with the reason recorded. This is not missing plumbing:

**lmz's best codec for BF16 weights is one its own GPU kernel cannot read.**
`CODEC_BF16C` codes sign and mantissa per exponent bucket, so its sub-streams
are not independent rANS and the batch decoder has nothing to take. Measured on
real archives: an fp32 checkpoint comes out **19/19 `CODEC_SPLIT`**, a BF16 one
**224/224 `CODEC_BF16C`**. lmz picks the conditioned form whenever it wins on
size, which on real weights is always.

So the GPU decode path today serves fp32 and fp16 checkpoints and not the LLM
BF16 case this project is mostly about. Closing that is lmz's: either a kernel
that reads the conditioned form, or a way to ask for `CODEC_BF16` when the
archive is destined for a GPU. It is one decision and one kernel, not research.

### The merge kernel, and why it is here

`merge.cu` transposes plane-major output back into elements at **264 GB/s**.
It is on lmz's side of the boundary and is on loan; the alternative was the
CUDA driver's own strided copy, measured at **1.87 GB/s**, which would have made
the device route slower than the host one it exists to beat.

Two bugs in it were found by testing rather than by reading. The vectorised
stores assume the destination is aligned to the element width, and a chunk's
destination is a byte offset into a safetensors file — aligned or not depending
on how long the JSON header happens to be. Misaligned, the vector store faulted
the whole CUDA context; the byte-wise kernel now takes those. And the upload
accounting initially omitted stored planes, which flattered the one number the
route is sold on.

### Conditions

- PTX, not a cubin or a shared library: the driver JITs it for whatever card is
  present, so one artifact covers every architecture and loading needs no CUDA
  runtime. Built by nvcc on first use, gitignored, exactly the bargain lmz
  already makes.
- **One card, one architecture.** RTX 5080, sm_120. Nothing here has run on
  Ampere, Ada, Turing or Hopper, and there is no non-NVIDIA device path at all.
- Chunk-size effects above are two points, not a curve. The middle has not been
  measured.

## The BF16 blocker, fixed on our side — 2026-08-30

*The earlier section said lmz's conditioned codec was one its own GPU kernel
could not read, and left it there as lmz's problem. It is our problem, and it
is fixed without touching lmz.*

### The fix is a writer-side choice, and it costs almost nothing

lmz only *attempts* its conditioned BF16 codec when a chunk holds at least
`COND_MIN_ELEMS` = 1M elements — 2 MiB for BF16. Compress below that and it
never reaches the conditioned path, emitting `CODEC_BF16`: two independent
planes, which is exactly the shape the batch decoder takes.

| chunk size | ratio | saved | chunks | codecs |
|---|---|---|---|---|
| 8 MiB (lmz's default) | 0.6709 | 32.9% | 225 | 224 × `BF16C` |
| **1 MiB** | 0.6734 | **32.7%** | 1788 | **1787 × `BF16`** |

**Two tenths of a point of ratio buys the entire GPU decode path**, and the
encode ran *faster* at the smaller chunk size (1.39 against 1.13 GB/s).
`lmsluice.gpu_chunk_size()` returns the threshold, reading it from lmz rather
than hardcoding it, and `lmsluice plan` now says so on any archive that needs
it.

Decoding `CODEC_BF16C` itself remains genuinely impossible through the shipped
ABI, and that is worth stating precisely rather than as a complaint: the
conditioned form partitions sign and mantissa into per-bucket streams of
**unequal length**, and `lmz_gpu_decode_batch_dev` requires every stream in a
batch to decode to the same size. There is no arrangement of the arguments that
expresses it. Avoiding the codec is the only route from outside lmz, and it is
cheap enough that it is arguably the better one anyway.

### Two correctness bugs the byte-level check caught

**`CODEC_BF16` does not split on byte boundaries.** It splits on bfloat16's own
fields — plane A holds the 8 exponent bits, plane B the sign in bit 7 with 7
mantissa bits below. Reassembling those as a high and low byte produces output
of exactly the right length and entirely the wrong values, and passes every
shape check. The kernel now mirrors `merge_bf16_scalar` in lmz's `lmzcore.c`:
`w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F)`.

**On real weights, plane 1 is not coded at all.** Sign and mantissa measure
7.92 bits of 8, so lmz stores them raw — every BF16 chunk is one rANS plane and
one stored plane. An early version wove stored planes in afterwards with a
byte-wise scatter, which is meaningless for a bit-field split. Coded and stored
planes now arrive in two buffers and the merge takes both.

### Measured, BF16 checkpoint at 1 MiB chunks

Cold, best of three, all byte-identical to the source after copying out of VRAM:

| route | GB/s | |
|---|---|---|
| plain → VRAM | **2.76** | reads 1.87 GB |
| host-decode → VRAM | 1.88 | |
| device-decode → VRAM | 2.01 | **kernels 101 GB/s**, 100% on device, 1.26 GB over PCIe |

**The decoder is no longer the constraint.** 101 GB/s against lmz's own
published 111 for the same kernel — the gap that was 50× is gone, and what
closed it was fixing our dispatch rather than anything in lmz:

- reading 1787 chunks one at a time became a coalesced parallel fetch through
  the same transport every other route uses;
- the stored plane was being concatenated into a host buffer before upload,
  which cost 0.49 s of pure interpreter memcpy on a 1.87 GB model — reads now
  go into writable buffers via `preadv` so those bytes reach the device without
  a copy;
- the PTX module was being JIT-compiled **on every call to `to_device`**, about
  half a second each time, and is now built once per context.

### Why it still does not beat the plain file here

Device-decode moves 1.26 GB where plain moves 1.87 and decodes at 101 GB/s, and
is still slower. That is this box, not the arithmetic: **the "cold" read here is
the Windows host cache at about 5.8 GB/s**, because the guest page cache can be
dropped and the hypervisor's cannot. At that speed the 33% byte saving is worth
about 0.11 s and does not cover the route's remaining dispatch. Against this
machine's genuinely cold first-touch rate of 2.3 GB/s, or any disk slower than
that, the gate says device-decode wins by the full 1/f = 1.49×. That case has
not been run, because this machine cannot produce it twice.

The stream-count effect from the earlier section still holds and is visible
here in the other direction: the 151 MB fp32 model yields 72 streams and its
kernels manage 1 GB/s, while the 1.87 GB BF16 model yields 3574 and reaches
101. **An lmz stream is 8 lanes wide however large it is, so device decode is a
function of chunk count, and small models cannot fill a GPU.**

## The upload stage, as a curve — 2026-08-30

*Requested after the device route's rate moved between builds and the cause was
found to be within-build variance in one stage. This characterises that stage as
a mechanism with parameters rather than as a number, so it re-evaluates on a
machine nobody here has.*

### The measurement

512 MiB moved host-to-device per point, median of 5, RTX 5080 on Gen4 ×16.
Call size varied by changing how much each `cuMemcpyHtoDAsync` carries.

| per call | calls | pageable GB/s | pinned GB/s |
|---|---|---|---|
| 64 KiB | 8192 | 1.49 | 1.54 |
| 256 KiB | 2048 | 3.46 | 4.13 |
| **512 KiB** | 1024 | **6.63** | 9.15 |
| 1 MiB | 512 | 7.89 | 12.12 |
| 4 MiB | 128 | 13.75 | 20.62 |
| 16 MiB | 32 | 15.36 | 26.08 |
| 256 MiB | 2 | 15.90 | 28.20 |

### The parameters, which are what travels

Least-squares fit of `t_call = overhead + n / BW` over the whole sweep:

| | per-call overhead | asymptotic bandwidth | crossover `n = overhead × BW` |
|---|---|---|---|
| pageable | **54.6 µs** | 15.9 GB/s | **0.87 MB** |
| pinned | **47.6 µs** | 28.3 GB/s | **1.35 MB** |

The crossover is the transfer size at which per-call overhead stops dominating —
below it you are paying for calls, above it for bytes. **The device route sends
512 KiB per call**, because that is one stored plane of a 1 MiB chunk, and 512 KiB
is *below* both crossovers. That is the whole defect, stated as a size rather than
as a rate.

**Concurrency does not rescue it**, which was worth checking rather than assuming:
at the route's own 512 KiB call size, 1 stream gives 11.33 GB/s and 2, 4 and 8
streams give 7.66, 7.40 and 7.48. More streams is *worse*, so the overhead is
serialised inside the driver rather than being queue depth waiting to be filled.

### What it predicts elsewhere

- **A Gen5 ×16 link** doubles bandwidth and leaves per-call overhead alone —
  overhead is CPU and driver work, not link time. So the crossover moves **up**,
  to ~1.74 MB pageable and ~2.70 MB pinned, and **the defect gets worse on a
  faster link**: the same 512 KiB call falls further below the knee. A machine
  twice as capable would show a larger shortfall from the same code.
- **A unified-memory host — Apple silicon, Strix Halo — has no upload stage at
  all.** There is no discrete device memory to copy into, so both this defect and
  its fix are absent, and the device-decode route's arithmetic loses one of its
  three stages entirely. Anyone reading the numbers above as a property of the
  *design* rather than of *this link* will over-weight them; on roughly half the
  target envelopes in `blueprint.md` §1 this section describes nothing.
- **A slower link** (Gen3, Thunderbolt, an eGPU) moves the crossover **down** and
  makes the current call size relatively less bad.

### The fix, still not implemented

Batch the stored planes through a **pinned** staging buffer, which moves both
terms: pinned lowers per-call overhead 54.6 → 47.6 µs and nearly doubles the
asymptote, and batching raises the call size past the crossover. It is not done,
deliberately — it costs back a host-side copy separately measured at **0.49 s** on
1.87 GB, and choosing between those is a measurement job rather than part of the
restructuring this was queued behind. It remains the first thing to try.

## Route rates with their spread — 2026-08-30

*Every route claim in this file and in `README.md` rests on these. Nine runs each,
cache dropped before every run, on the box described at the top.*

| model | route | best | median | worst | spread |
|---|---|---|---|---|---|
| Llama BF16 1.87 GB, 1 MiB chunks | plain → VRAM | 3.16 | **3.10** | 2.53 | 1.25× |
| | host-decode → VRAM | 2.29 | **2.22** | 2.16 | 1.06× |
| | device-decode → VRAM | 1.71 | **1.36** | 1.12 | 1.53× |
| whisper fp32 151 MB | plain → VRAM | 0.62 | **0.56** | 0.47 | 1.33× |
| | host-decode → VRAM | 0.52 | **0.50** | 0.34 | 1.52× |
| | device-decode → VRAM | 1.47 | **1.39** | 0.91 | 1.62× |

All GB/s of plain bytes. **The device route is the widest-spread of the three on
both models**, which is consistent with the upload curve above: it is the only
route whose dominant stage sits below the crossover, so it inherits that stage's
variance.

**The fp32 device claim survives the error bar.** Median to median it is
1.39 / 0.56 = **2.48×**, and even the worst device run against the best plain run
is 0.91 / 0.62 = 1.47×. The README's "about 2× plain" is supported rather than
flattered by a lucky sample.

**The BF16 device route is genuinely slower than plain here** — 1.36 against 3.10
at the median — and that is the box, not the arithmetic: "cold" on this machine is
the Windows host cache at ~5.8 GB/s, so a 33% byte saving buys about 0.11 s and
does not cover a stage sitting below its crossover. Against the genuine
first-touch rate of 2.3 GB/s the gate predicts device-decode wins by 1/f = 1.49×,
and **that case has not been run**.

## The byte-identity checks passed for months while a race was live

*Written down because the reassuring reading of that sentence is the wrong one.*

`Model.to_device` stages through two pinned buffers used alternately, and guarded
reuse with `if pending is pin: stream.synchronize()`. `pending` held the **most
recent** buffer, which with two alternating buffers is never the one about to be
reused — **so the guard never fired.** Every staging window was written into a
buffer whose own host-to-device copy might still have been in flight, on both the
plain and host-decode routes into VRAM.

Both routes are covered by SHA-256 byte-identity checks, and those checks passed.
They passed because **the race loses almost every time**: reading and decoding the
next window takes longer than the copy of the previous one, so the corruption
window is narrow and the check is not sensitive to it. It surfaced once, as a
single `DIFFER` in a sweep that had passed immediately before and passed again
immediately after.

The lesson is about evidence rather than about CUDA. **A passing correctness check
is not evidence that a bug is absent — it is evidence that the bug did not fire
this time.** Where a failure is intermittent, re-running until green destroys the
only signal there was. The fix was to synchronise before reusing any buffer, and
the test that now covers it (`test_many_small_windows_still_land_intact`) drives
an 8 KiB window so that buffer reuse dominates rather than reading, making the
race likely instead of rare.

### The 29% prediction error, re-measured

The README's "the plan is wrong for the first time, by 30%" was originally
best-of-three. Re-measured under the nine-run discipline: host-decode into VRAM on
the 8 MiB-chunk BF16 archive predicted **1.05 GB/s**, measured median **0.75 GB/s**
over nine cold runs (0.65–0.79), an error of **29%**. The figure survives the
stricter measurement, so the sentence keeps its number.

## Phase 0 — the case where it pays, measured at last — 2026-08-30

*Every earlier number in this file was taken on a link where compression is a
tax. This is the first measurement of the case the project exists for. The link
is a 9p mount from WSL2 to the Windows host — **a sample of the sub-1-GB/s
class, not a target**; it is a WSL2 artefact and the finding is the link rate
and the gate, not the filesystem.*

Fixtures are synthetic BF16 with per-row lognormal scales, generated at three
sizes from one generator so the **ratio is constant (f = 0.673) and size is the
only variable**. 1 MiB chunks, so every chunk is GPU-readable `CODEC_BF16`.

### Loading from the slow link

Cache dropped before every run (`posix_fadvise` works on 9p), five runs, median.
All byte-identical.

| model | plain GB/s | coded GB/s | speedup |
|---|---|---|---|
| 64 MiB | 0.461 | 0.580 | 1.26× |
| 256 MiB | 0.502 | 0.606 | 1.21× |
| **1024 MiB** | 0.447 | **0.646** | **1.45×** |

Ceiling from the ratio is 1.49×, so at 1 GiB the coded route reaches **97% of
the maximum the ratio allows**. The plain route stays flat at 0.45–0.50 GB/s
while the coded route climbs with size, which is the fixed costs amortising:
**the win needs size, and it saturates by about a gigabyte.** That was the
falsification criterion for this phase and it passed.

### Saving to the slow link

Three runs, median, fsync included on both sides.

| model | plain s | coded s | speedup |
|---|---|---|---|
| 64 MiB | 0.29 | 0.26 | 1.10× |
| 256 MiB | 1.13 | 0.96 | 1.18× |
| **1024 MiB** | 4.50 | **3.55** | **1.27×** |

Same shape, lower ceiling reached: 1.27× against 1.48×. **The write path leaves
about 15% on the table and the arithmetic says why.** Writing the coded output
takes 0.723 GB / 0.239 = 3.02 s and encoding takes 1.074 / 1.5 = 0.72 s. Fully
overlapped that is 3.02 s; fully serial it is 3.74 s; measured is **3.55 s**, so
roughly three quarters of the encode time is exposed rather than hidden under
the write. The read path pipelines its stages and the write path does not. That
is a concrete piece of work, not a mystery.

### Two measurement traps, both of which would have inflated the result

**`shutil.copyfile` is not a fair plain baseline on this link.** It manages
0.106 GB/s where a 4 MiB-buffered write with fsync manages 0.239 — the link is
per-call bound, so the buffer size dominates. Measured against `copyfile`, the
write win reads as **2.99×**; measured against the best plain method it is
**1.27×**. Two thirds of the apparent win was the baseline's buffer, not
compression. A win that exceeds the ratio ceiling is the tell: 1/f is the most
that moving fewer bytes can buy, so anything above it is the comparison being
unfair somewhere.

| plain write buffer | GB/s |
|---|---|
| 64 KiB | 0.150 |
| 256 KiB | 0.192 |
| 1 MiB | 0.239 |
| 4 MiB | 0.246 |
| 16 MiB | 0.249 |

**And the link's write rate was wrong in the first place.** The 0.098 GB/s
figure quoted in `docs/strategy.md` came from a single-threaded sequential write
with a poor pattern. Written properly the link does **0.239 GB/s**, so the write
gate against a 1.5 GB/s encoder is **6.3×**, not the 12.8× the strategy claimed.
The conclusion is unchanged — 6.3× is still far above 1, so the write path is
saturated and the ratio is the only limit — but the number was wrong and is
corrected there.

### What this does not cover

No real network. No USB, SD or eMMC device. No phone. The 9p mount is one
sample of the sub-1-GB/s class and every conclusion above should be read as
"links of roughly this speed", not "this filesystem".

## Compress on first fetch — built and measured — 2026-08-30

*The supply-side answer, made real: `lmsluice/zstdcodec.py` (a codec that needs
nothing installed) and `lmsluice/cache.py` (the decision about whether to use
it). Same box, same 9p link as Phase 0.*

### The stdlib codec

Python 3.14 ships zstd, so this compresses and decompresses with nothing
installed at all. On a 151 MB fp32 checkpoint: **35.9% saved**, encode 0.58 GB/s,
and a self-measured decode rate written into the archive at build time.

**It does not scale with threads, and that is a finding rather than a
disappointment.** 16 blocks of 4 MiB, decoded on a pool:

| threads | GB/s |
|---|---|
| 1 | **2.02** |
| 2 | 1.51 |
| 4 | 1.78 |
| 8 | 1.79 |
| 16 | 1.69 |

More threads is *worse*. So the loader gains nothing from a pool for this codec
and the honest rate to feed the gate is the single-threaded one — which caps it
near 2 GB/s however many cores are present. That is a real difference from lmz's
native kernel, it is fine below the gate where a no-dependency codec is aimed,
and it is a reason not to reach for it above one.

**This corrected a defect in the cache decision.** The first version measured
decode on one thread over an 8 MiB sample — two blocks — and got 1.01 GB/s, and
the gate then declined to cache at 1.12×, inside its own margin. The second
version "fixed" it by measuring on a pool, on the assumption that the loader
would parallelise; the table above says it cannot. What was actually wrong was
the *sample*: two blocks is a sample of the safetensors header as much as of the
weights. Sampling 64 MiB from the middle gives 2.18 GB/s and the gate then says
**1.56× — build it**. Both wrong versions were wrong in the conservative
direction, which is not the same as being safe.

### The cache, end to end

151 MB model on the 9p link, cache dropped before every run, five runs, median,
every result byte-identical to the source:

| | seconds | against plain |
|---|---|---|
| plain, on the slow link | 0.343 | — |
| **coded, on the slow link** | **0.191** | **1.80×** — compression alone |
| **coded, in the local cache** | **0.089** | **3.85×** — compression *and* a faster device |

Ceiling from the ratio alone is 2.34×, so **the 3.85× is not all compression**
and must not be quoted as though it were. The cache lives on local disk while
the source is on the mount, so it delivers two effects at once: fewer bytes, and
fewer bytes from somewhere faster. Both are real and the second is a legitimate
property of putting a cache on fast storage — it is simply not the ratio.

The tell was the same one as the write-path artefact earlier in this file: **a
win above 1/f means the comparison is unfair somewhere**, because 1/f is the most
that moving fewer bytes can buy. That rule has now caught two of my own
measurements in one day.

### What the decision does

The gate that routes a read decides whether the cache is worth building, using
the link rate from the profile and the codec's rate measured on *this file's own
bytes*. On a fast NVMe it declines and says the plain file is the right cache
entry. Nothing is cached without being asked, nothing is deleted, and an entry is
bound to its source's path, size and mtime so an edited checkpoint misses rather
than serving stale weights.

## The decode rate this file published was one codec's, not lmz's — 2026-08-30

*A correction to numbers used throughout this file and the README. Found while
normalising a competitor comparison, which is the value of doing one.*

Same library, same machine, same 1.87 GB BF16 fixture, same transport. Only the
encoder's chunk size differs, and with it which codec lmz reaches:

| archive | codec | chunk | decode |
|---|---|---|---|
| `m.default.lmz` | `CODEC_BF16C` (conditioned) | 8 MiB | **0.97 GB/s** |
| `llama.cs1m.lmz` | `CODEC_BF16` (field split) | 1 MiB | **5.95 GB/s** |

**6.1× on a codec parameter** — larger than any difference between the products
this repository compares itself against. Every "lmz decodes at about 1.05 GB/s"
above is the first row, presented as lmz's rate.

### Decode alone, from RAM, threads swept

Payloads pre-loaded so no I/O is in these numbers; plain bytes per second.
lmz `CODEC_BF16`, 1 MiB chunks, 400 chunks / 418 MB:

| threads | CRC on | CRC off |
|---|---|---|
| 1 | 0.95 | 1.09 |
| 4 | 3.66 | 4.18 |
| 8 | 6.75 | 7.91 |
| 16 | **8.29** | **10.41** |

Near-linear to 16 threads; checksums cost 15–20%. Repeated writing into a
distinct 418 MB buffer rather than a reused 1 MiB scratch, in case the scaling
was cache residency: **8.23 GB/s, unchanged.** So it is not an artefact.

Through lmsluice's own transport on the same archive: **5.95 GB/s**, so the
transport delivers 72% of the codec — a real gap, and not the order of magnitude
the old figure implied.

For comparison, stdlib zstd on the same data: 0.77 GB/s at one thread, 1.69 at
eight and sixteen. Flat past eight, as the earlier section found.

### End to end, and the claim it corrects

Local NVMe, cache dropped each run, n=7, median, every result byte-identical:

| | seconds | GB/s | vs plain |
|---|---|---|---|
| plain safetensors | 0.967 | 1.94 | — |
| coded, BF16C 8 MiB | 2.508 | 0.75 | 0.39× |
| coded, BF16 1 MiB | 1.030 | 1.82 | **0.94×** |

On the machine this repository calls the case where compression is a tax, it is
**break-even** with the right chunk size — and still saves 33% of the disk.

**The published "6× tax" was worse than a stale number: it paired the fastest
plain measurement (6.2 GB/s, host-cache-warm) with the slowest coded one
(BF16C).** Two legitimate figures, one indefensible comparison. It is corrected
here and on the front page rather than defended.

### The crossover, as a formula

    decode(T) ≈ min(T × 0.52,  8.3) GB/s   lmz CODEC_BF16, CRC on, 1 MiB
    decode(T) ≈ min(T × 0.60, 10.4) GB/s   same, CRC off
    decode(T) ≈ min(T × 0.77,  1.7) GB/s   stdlib zstd, 1 MiB blocks
    decode(T) ≈ min(T × 0.06,  1.0) GB/s   lmz CODEC_BF16C, 8 MiB

    compression pays when decode(T_available) > link
    the win is min(1/f, decode/link)

**This box at 16 threads:** the crossover moves from ~1.0 GB/s (published, BF16C)
to **~8.3 GB/s**. This machine's storage reads 2.4 cold and 6.2 warm, so both now
sit below it.

The 1 MiB chunk size was chosen to keep archives GPU-decodable at 0.2 points of
ratio. It buys 6× on CPU decode as well, which was not known when it was chosen.

### Conditions

Decode-only, from RAM. An end-to-end load also reads the coded bytes, which is
why the measured 0.94× sits below the 1.49× the ratio alone allows. One machine,
one dtype, one checkpoint.

## Job 3 — pricing the Xet conflict — 2026-08-30

*Does publishing a coded archive to a content-addressed hub trade away
cross-revision dedup? `strategy.md` §2b assumed the public-hub download was a
clean win without pricing that. It mostly survives, and the reason is not the
one the question assumes.*

### CDC on plain bytes catches tensor duplication completely

`/mnt/d/Models/Llama-3.1-8b-Instruct` ships the same weights twice — 16.06 GB of
safetensors and a 16.06 GB `original/consolidated.00.pth`, which is a zip with
**all 295 members stored uncompressed**, so the tensor bytes are present
verbatim at different offsets. That is the sharpest available test of whether
byte-level chunking catches what tensor-level dedup catches.

Content-defined chunking at ~64 KB (windowed gear sum, 99% shift-resistant on a
one-byte insertion) over the largest `.pth` member and the safetensors shard
holding the same tensor:

| | chunks |
|---|---|
| `.pth` member `data/0`, 1.05 GB | 12,097 |
| safetensors shard 1, 4.98 GB | 57,731 |
| **shared** | **12,095 — 100.0%** |

The two misses are the boundary chunks at each end, where surrounding context
differs. **So Xet's dedup does catch cross-container duplication**, and lmz's
64.6% directory figure is not something CDC would miss.

*Method note:* a first chunker used a single-byte gear predicate and produced
**zero** boundaries — with a 256-entry table the condition depends on the byte
*value*, not the position, so either a value cuts everywhere or nowhere. It
would have reported "CDC finds no duplication" for a reason unrelated to the
data. Replaced with a windowed sum before any result was taken.

### But coded archives dedup across revisions too, which the question assumes away

The premise is that entropy-coded bytes do not chunk-dedup. That is false for
the case that matters. lmz chunks **fixed spans of plaintext**, so a
shape-preserving edit — the normal shape of a fine-tune or a re-quantisation —
leaves every untouched chunk byte-identical after coding. 151 MB model, edits
made in place:

| edit | plain bytes changed | coded chunks new |
|---|---|---|
| largest tensor replaced | 52.7% | **52.7%** |
| one byte flipped | 0.0% | 0.7% |
| 4 KB at the head of all 167 tensors | 0.3% | **32.2%** |

For a **localised** edit, coded chunks track plain changes one-for-one and lmz
dedups exactly as well as CDC. For a **scattered** edit it is worse, and the
mechanism is chunk granularity: touching one byte rewrites its whole coded
chunk, so the change is amplified by roughly the ratio of coded chunk size to
CDC chunk size.

### The amplification is a dial, and it is the encoder's chunk size

Same scattered edit (0.30% of bytes, across 167 tensors):

| lmz chunk | ratio | coded chunks new | amplification |
|---|---|---|---|
| **64 KiB** | 0.453 | **3.3%** | **10.8×** |
| 256 KiB | 0.435 | 12.6% | 42.1× |
| 1 MiB | 0.432 | 32.2% | 107.2× |
| 4 MiB | 0.434 | 50.0% | 166.5× |

Matching Xet's own granularity costs **2.1 points of ratio** and cuts the
amplification tenfold.

### The crossover

For `R` revisions sharing a base, plain change fraction `c` per revision, coded
ratio `f`, amplification `A`:

    plain on a CDC hub   N · (1 + (R-1)·c)
    coded archives       f·N · (1 + (R-1)·c·A)

    coded wins while     R  <  1 + (1 - f) / (c · (f·A - 1))

**This box's instance**, at `c` = 0.003 measured above:

| lmz chunk | f | A | coded wins up to |
|---|---|---|---|
| 1 MiB | 0.432 | 107 | **~5 revisions** |
| 64 KiB | 0.453 | 10.8 | **~48 revisions** |

So **the public-hub case survives**, and §2b does not need reordering. A first
download of any single revision is a clean win for coding regardless — Xet's
dedup buys nothing on a cold fetch, only on an update. The conflict is real only
for a user tracking many revisions of one model, and even then the encoder's
chunk size moves the crossover by an order of magnitude for two points of ratio.

### Conditions

One model family, one dtype, edits synthesised rather than taken from real
consecutive releases — `c` for actual fine-tune revisions is unmeasured and is
the input the crossover is most sensitive to. `/mnt/d` reads at 0.20 GB/s, which
is why this used bounded samples rather than all 32 GB.

---

## The destination buffer costs more than the transport — 2026-08-30

The finding that was held back pending fresh-process runs. It has them now: every
row below is a **separate process, one measurement each, median of 5–7**, because
a repeated in-process benchmark measures the allocator's state as much as the
allocation. (It turned out not to matter — see Conditions — but that could not be
known without checking.)

`Model.load()` allocates its destination with `bytearray(self.plain_bytes)` before
`_gather` reads a byte. On the 1.87 GB fixture that allocation is **0.55 s**, and
the transport it exists to serve is **0.31 s**.

### Where the cost is, and it is not where "allocation" suggests

`bytearray(N)` zero-fills. Minor-fault counts locate the work exactly:

| allocation | alloc | first write | total | minor faults | AnonHugePages | needs |
|---|---|---|---|---|---|---|
| `bytearray(N)` | 0.568 | 0.089 | **0.657** | 457,778 | 0 | stdlib |
| `mmap(-1, N)` (defaults) | 0.000 | 0.835 | 0.835 | 457,730 | 0 | stdlib |
| `mmap` + `MADV_HUGEPAGE` | 0.000 | 0.843 | 0.843 | 457,730 | 0 | stdlib |
| `mmap` **`MAP_PRIVATE`** + `MADV_HUGEPAGE` | 0.000 | 0.147 | **0.147** | 1,407 | 1.74 GB | **stdlib** |
| `numpy.empty` | 0.000 | 0.129 | **0.129** | 1,407 | 1.74 GB | numpy |

The buffer is 457,519 pages. `bytearray` faults **every one of them eagerly**, at
allocation, and writes 1.87 GB of zeroes that the decoder then overwrites — one
whole redundant pass over the destination. `mmap` defers the faults instead of
removing them, which is worse: 457,730 faults taken one at a time during the write
cost more than taking them in a batch.

The last two rows take **325× fewer faults** because the buffer is backed by 2 MiB
huge pages, confirmed in `/proc/self/smaps` rather than inferred: 1,828,864 kB of
`AnonHugePages` against 0 for every other row.

**`MAP_PRIVATE` is the whole difference between rows 3 and 4.** Python's
`mmap.mmap(-1, N)` defaults to `MAP_SHARED`; transparent huge pages do not back
shared anonymous mappings, so `MADV_HUGEPAGE` on one is accepted and silently does
nothing. That is why row 3 looks like a failed fix rather than a misconfigured one.

### What it costs the load path

Cold-protocol runs, fresh process each, n=5 median, the same 1.87 GB fixture and
its 1 MiB-chunk `CODEC_BF16` archive:

| route | destination | alloc | load | total | GB/s |
|---|---|---|---|---|---|
| plain | `bytearray` (what `load()` does) | 0.546 | 0.311 | 0.848 | 2.21 |
| plain | huge-page buffer via `into=` | 0.000 | 0.361 | 0.361 | **5.18** |
| coded | `bytearray` | 0.524 | 0.450 | 0.974 | 1.92 |
| coded | huge-page buffer via `into=` | 0.000 | 0.449 | 0.449 | **4.18** |

**2.4× on the plain route and 2.2× on the coded one, for a buffer.** No decoder,
no scheduler, no link involved.

### It does not move the gate, and it makes the coded route look slightly worse

Both routes pay it identically, so the verdict does not change — but removing a
*shared* fixed cost widens the proportional gap between them. Coded against plain
goes from 1.13× slower to 1.24× slower. The allocation was partly masking the
difference, and saying so is the point of recording it.

### What it predicts elsewhere

    cost  ≈  N / BW_mem_write  +  (N / page_size) × t_fault

This box: the already-faulted write runs at 21 GB/s, so of `bytearray`'s 0.568 s
only ~0.089 s is the zeroing itself and **~0.479 s is fault overhead**, about
1.05 µs per 4 KiB page.

The consequence that generalises: **this cost is a memory-subsystem cost, not a
link cost, so it does not shrink when the link gets faster — it takes over.** At
0.26 GB/s over 9p, 0.55 s is ~7% of the load. On a Gen5 array at 12 GB/s it is
~78% of it. The faster the storage, the more of lmsluice's headline number is a
buffer nobody asked for.

Across the target envelopes the *remedy* is less portable than the problem:

- **Linux with THP in `madvise` mode** (this box, most phones): the fix applies.
- **THP set to `never`**: no remedy, `bytearray`'s cost stands.
- **macOS/Apple silicon**: no `MADV_HUGEPAGE`, but 16 KiB base pages cut the fault
  count 4× natively, so the problem is smaller and the fix unavailable.
- **Windows**: large pages exist but need `SeLockMemoryPrivilege`, which a library
  should not require.

So it must be a runtime probe with a `bytearray` fallback, not an assumption —
same shape as everything else here.

### Conditions

9800X3D, 24 GB WSL2, ext4 on Gen4 NVMe, CPython 3.14, 4 KiB pages, 2 MiB huge
pages, `/sys/kernel/mm/transparent_hugepage/enabled` = `always [madvise] never`.
Fixture `/home/rog/.cache/lmz-bench/model.safetensors` (1,872,871,856 plain
bytes); archive `/home/rog/.cache/yfce-load-bench/llama.cs1m.lmz`.

**The read is page-cache warm, not cold.** `posix_fadvise(DONTNEED)` was issued and
the load ran at 5.9–6.0 GB/s, which is this box's warm rate — under WSL2 the Linux
call cannot evict the Windows host's cache. The allocation figures are unaffected
(they touch no file), but the *ratio* of allocation to transport is the warm one;
cold, the transport would be ~0.78 s at 2.4 GB/s and allocation would be ~41% of
the total rather than 64%.

**Fresh-process and in-process agree here**, which was the open question: seven
consecutive `bytearray(1.87 GB)` in one process cost 0.50–0.68 s each and faulted
all 457,520 pages every time, because glibc `munmap`s an allocation this large on
free rather than keeping it in an arena. No warm-allocator correction is needed at
this size; at sizes below the mmap threshold it would be.

**This supersedes the earlier 0.7–1.0 s figure**, which was a small in-process
sample. It also settles the ≤0.3 s bound inferred from the cold end-to-end runs:
that bound was wrong, because it assumed the published cold timings included the
allocation. They timed `_gather`, not `load()`.

### Implemented — 2026-08-30

`lmsluice/buffer.py`, wired into all four allocation sites in `model.py`. The
alignment dance turned out not to be load-bearing after all and was dropped: a
private anonymous mapping plus the advice is the whole fix, three lines.

Verified on the real load path, fresh process each, n=5 median, same fixtures:

| route | before | after | |
|---|---|---|---|
| plain | 0.873 s (2.15 GB/s) | 0.363 s (5.16 GB/s) | **2.40×** |
| coded | 0.982 s (1.91 GB/s) | 0.478 s (3.92 GB/s) | **2.05×** |

The threshold is `8 × Hugepagesize` read from `/proc/meminfo` — 16 MiB here,
matching the measured crossover — rather than a named 16 MiB, because arm64 alone
ships 64 KiB, 2 MiB, 32 MiB and 512 MiB huge pages. Machines that cannot back the
mapping get `bytearray`, which on them is the faster of the two; `lmsluice probe`
prints which it chose and why.

**Two of the three details fail silently**, so `TestDestinationBuffer` checks the
kernel's `AnonHugePages` accounting rather than the arguments passed. Both
regressions were introduced deliberately to confirm the guard fires: dropping
`MAP_PRIVATE` and dropping the `madvise` each fail it, and only it.

**Absolute end-to-end load figures elsewhere in this file predate this and are
~2× pessimistic on this box.** The gate verdicts do not move -- both routes paid
the allocation equally -- with the one exception noted above, that coded against
plain widens from 1.13× to 1.24× once the shared overhead is gone.

---

## Every "cold" number this project took on WSL2 was a host cache — 2026-08-30

The blocker named in the last section turned out not to be what it looked like.
`posix_fadvise(DONTNEED)` works here; it was never the problem.

### The mechanism, and the parameters it depends on

**There are two caches, and only the upper one is addressable from Linux.**
`mincore()` reports, page by page, whether a file is resident in the Linux page
cache. After `fadvise(DONTNEED)` it reports **0.00% of 457,245 pages resident** —
the drop worked — and the very next read still returns **~6 GB/s**. Bytes that
are not in Linux memory are not coming off an NVMe at 6 GB/s.

The layer below is the Windows host's cache of the VHDX backing `/dev/sdd`. It is
invisible to the guest, unaddressable from it, and does not evict on demand:

| unrelated distinct data read first | fixture then reads at |
|---|---|
| 0 GB | 4.7 – 5.4 GB/s |
| 48 GB | 3.19 GB/s |
| 112 GB | 3.75 GB/s |

112 GB of pressure does not reach it, and more is not better. **The parameters
this depends on** are: a guest filesystem on virtualised block storage, a host
with enough free RAM to cache the backing file, and no guest-side interface to
the host's cache. That is every WSL2 box, every VM on a host with a file-backed
disk, and every container on such a VM. It is not a property of this hardware.

**The consequence is not local to this repository.** Any "cold" storage figure
taken inside such a guest by dropping the guest's cache is measuring the host's
cache. This file's own earlier cold rates, and `probe/`'s, were taken that way.
They are upper bounds on cold, not cold.

**Read that as a bound with a direction, and not as more than that.** It says
those figures were *no slower* than the truth, so gate verdicts derived from them
were conservative — compression pays at least as often as they implied. It does
**not** say by how much, and nothing available here can: the drift in the next
section means the true rate was not pinned even once. A slower true link makes
this project look better on more machines, which is exactly the direction that
does not feel like an error, so the bound is stated and the magnitude is not.

**And for a workload actually running in such a guest, the host cache is not an
artefact — it is that machine's performance.** A user loading a model under WSL2
gets those 6 GB/s. The host cache is a measurement problem only when the thing
being characterised is the *storage device*, which is what the gate needs, since
the gate must predict for machines whose caches are cold. So both numbers are
real and they answer different questions: the guest-warm rate is what a WSL2 user
experiences, the device rate is what the gate must compare a decoder against.

### The protocol that does work, and what it does not evict

**It evicts nothing. It reads data nothing has ever read.** A copy written and
never subsequently read is cold in both caches, because neither has had a reason
to hold it:

    stage N+2 copies of the fixture, fadvise(DONTNEED) each as it is written
    read them oldest-first, one per measurement, never twice
    verify with mincore that Linux residency is 0% before each clock starts

The two newest copies are spares: the most recently written one reads at 2.56
GB/s against 1.61–1.70 for its predecessors, so roughly 2 GB of subsequent writes
is enough to clear the write path. Reading oldest-first does that for free.

**It consumes fixtures.** One 1.87 GB copy per measurement, so *n* = 5 on two
routes costs 12 copies and about 19 GB, and a fixture set buys exactly as many
measurements as it has copies. When it runs out, the answer is more disk, not
more reads — re-reading a copy makes it a warm measurement wearing a cold label.

**Conditions, not just contamination.** `/` is mounted with `discard`, so
deleting a staged set issues inline TRIM, and staging a new one on top of that
measures the SSD's garbage collector. One run taken straight after a
delete-and-restage gave plain 0.29 GB/s where a quiesced run gave 1.08. That is a
property of this box worth recording rather than only designing around.

### What it measured, and what it could not

**The rate is not a stable quantity here.** Same operation, identical
never-before-read files, Linux residency verified 0% every time, and the rate
drifts within a single run — 0.26 → 2.38 GB/s ascending in one run, 2.50 → 0.27
descending in another. Nine pairs, both directions, no protocol variation. So
**no single cold GB/s figure is published**, and the earlier ones on the front
page stay marked as predating the destination-buffer change rather than being
replaced by these.

**The paired ratio survives, because it has a validity check.** Routes were
interleaved so neither absorbs the drift, and a coded route that moves `f` = 0.673
of the bytes cannot beat plain by more than `1/f` = **1.49×** when both are
link-bound. Any pair above that is not a fast archive, it is an unfair comparison
— the same rule that caught two of this file's earlier measurements:

| pairs | plain GB/s | ratio | |
|---|---|---|---|
| 4 | 1.52 – 2.38 | **1.05 – 1.39×** | usable |
| 6 | 0.26 – 1.46 | 1.69 – 6.96× | **discarded: above 1/f, so impossible** |

The six discarded pairs are all from the low-rate end, where the plain route's
larger working set — a 1.87 GB destination plus 1.87 GB read, against 1.26 GB
read — is penalised nonlinearly by whatever is throttling the device. They
measure that, not the archive.

**Result: cold, the coded route wins, by 1.05 – 1.39× (median 1.14×, n = 4).**
Below the 1.49× the ratio alone allows, which is the same "predicted in
direction, short in degree" gap the front page already reports warm. The verdict
reverses from the warm case, and it reverses in the direction the gate predicts:
at a link this slow, compression pays.


---

## Methodological notes this file earned the hard way — from 2026-08-30

Recorded here rather than in `CLAUDE.md`, which is not mine to edit.

### A result that beats a hard ceiling is a broken measurement, not a good one

Where the model gives a bound that cannot physically be exceeded, exceeding it is
a fault in the comparison — and that can be known **before** anything explains
it, from one device, with no second machine to check against.

The instance: a coded route carrying `f` of the bytes cannot beat plain by more
than `1/f` when both are link-bound. At `f` = 0.673 that ceiling is 1.49×, and it
discarded six of ten measured pairs reading 1.69× to 6.96×. The explanation came
afterwards and was not needed to reject them.

It has now caught four separate errors in this repository — the "6× tax" that
paired the fastest plain read with the slowest coded one, a `shutil.copyfile`
baseline on 9p, a 2.99× write win, and those six pairs. Every one of them was an
error **in this project's favour**. That is the pattern worth internalising: a
number that flatters the thing you are building does not feel like a bug, so the
arithmetic ceiling has to do the work that suspicion would otherwise do.

### A mode set on a descriptor outlives the measurement that set it

Added 2026-09-03, from reviewing the macOS path before it had ever run.

"Uncache this file" is two different kinds of operation depending on the
platform, and code that treats them as one gets both of them wrong on the
second one. Linux's `posix_fadvise(DONTNEED)` is an **event**: it drops what is
cached and leaves the descriptor exactly as it was. macOS's `F_NOCACHE` is a
**mode**: it is set on the descriptor and stays set, so every subsequent read
bypasses the cache — and does not populate it either.

Two failures follow, and neither can appear on the platform the code was
written on:

- **The measurement that comes next is wrong.** A "warm" rate taken by
  re-reading after the mode was set is not warm, because that read bypassed the
  cache too. And clearing the mode is not enough on its own: nothing populated
  the cache, so the region has to be read once untimed before a timed read means
  anything.
- **The object is left changed.** A `Source` that had been probed kept bypassing
  the cache for the rest of its life, so every load after a probe would have
  been slower — on one platform only, for a reason nothing in the code says.

**The general form: ask whether an operation is an event or a mode, and if it is
a mode, whose lifetime it now belongs to.** An event needs no undo. A mode needs
one, needs it called, and needs the state it suppressed rebuilt before anything
is measured against it. The same shape appears wherever a measurement changes a
handle rather than acting on data — `O_DIRECT`, `TCP_NODELAY`, a scheduler or
governor setting, a `setrlimit`.

### Three questions this machine cannot answer, and two of them are one question

All are *different hardware*, not *more work*, and none is closable by effort
here:

| question | why this box cannot answer it |
|---|---|
| the 2-CU adapter's shared-memory **pool per compute unit** | Exposed by no interface that adapter offers: core Vulkan has no such query, and `VK_AMD_shader_core_properties` and `shader_core_properties2` — both present, both queried — carry no LDS field. Obtainable only by running the kernel on the device. |
| a true **cold storage rate** | Every read here is served by a host cache the guest cannot address or evict — and, as of 2026-09-03, not reliably even when nothing is done differently: the same protocol gave 1.6 GB/s one hour and 5.3 the next. Obtainable only on storage that is not behind such a cache. |
| whether the coded route pays at a **cold first-touch** link | Downstream of the row above. Its one measurement, 0.94×, is indistinguishable by rate from a warm read plus the allocation it timed. |

They bound the same claim from opposite ends — the first sets how fast a small
iGPU decodes, the others how slow the link it is racing actually is — and the
gate is the comparison between them. So the one measurement that would most
change what this project knows is not a better analysis of this machine. It is
any of these on a different one, and the last two are the *same* measurement on
storage that is not behind a host cache.

### A test that supplies its own observer cannot tell you the observation works

Added 2026-09-03, from verifying the vLLM loader against a real install.

The loader emits `"Loading weights took %.2f seconds"` so that
`--load-format lmsluice` can be compared against vLLM's default on vLLM's own
metric. A unit test asserted the record was logged, and passed. In a real vLLM
process the line **never appeared**: vLLM's `dictConfig` attaches its handler to
the logger named `vllm` and leaves the root logger alone, so a logger called
`lmsluice` had no handler and its INFO records went nowhere.

The module was correct. The test was correct. The observation was impossible.
`unittest.assertLogs` works by attaching *its own* handler to the logger under
test, so it proves a record was emitted and says nothing about whether anything
downstream would ever receive it. Every test of the form "assert we logged X"
has this shape.

**The general form: when the host owns the logging tree — a plugin inside a
server, an extension inside an application, anything loaded rather than run —
being visible is a property of the integration and cannot be tested from
inside.** It needs the host. Here the fix was to name the logger `vllm.lmsluice`
so it inherits the handler vLLM configured, which is what vLLM's own modules get
by passing `__name__`.

Worth pairing with what the two verification stages each caught on this one line:
reading vLLM's released source showed the timer **did not exist** (the counters
live in `DefaultModelLoader`, not `BaseModelLoader`), and running a real install
showed that once it existed it was **invisible**. Neither stage alone would have
found both, and the cheap stage came first.

---

## Retired figures — 2026-09-03

Every number below appeared in this repository's documents and is superseded.
It is kept as a list so that a reader meeting an old figure in a commit
message, a handover or someone's notes can look it up, and so the next sweep
greps one table instead of rediscovering the set.

**Why this exists.** Three superseded figures were found in three separate
places on one day, each fixed where it was noticed and none propagated. Fixing
them one at a time as they surface does not converge; a register does.

| retired | why it was wrong | replaced by |
|---|---|---|
| lmz decodes at **~1.05 GB/s** (also 1.03–1.07) | `CODEC_BF16C` at 8 MiB presented as lmz's rate | **5.95 GB/s** through the transport (`CODEC_BF16`, 1 MiB, 8T); **8.29** decode-only from RAM at 16T — a bound, not a crossover |
| **0.39–0.48 GB/s** single-threaded | same codec | **0.95** CRC on / **1.09** CRC off, per thread |
| crossover at **~1 GB/s** of storage | derived from 1.05 | **~5.95 GB/s of link** |
| **"costs you 6× on load time"**, and 0.17× / 0.09× | fastest plain read (host-cache-warm) against slowest coded (BF16C) | **0.74×** warm NVMe measured, against 0.95× predicted; ~20% unaccounted |
| **0.94× break-even** | earlier cold run with the destination allocation inside the timing | 0.74× is the front-page figure; 0.94× only with its conditions stated |
| stdlib zstd **0.59–0.71 GB/s** | earlier measurement | **0.77** at 1T, **1.69** at 8–16T, flat above that |
| "the decoder varies maybe **3×** across CPUs" | codec choice alone is 6.1× on one CPU; threads add ~8× | codec-and-thread dependent over more than an order of magnitude; gate with the through-transport figure |
| `hf_transfer` — "parallel ranges, no compression" | deprecated; the env var is a no-op | `hf_xet`: CDC at ~64 KB, adaptive 1→64 streams, chunk dedup |
| "entropy-coded bytes do not chunk-dedup" | false for shape-preserving edits | coded chunks dedup 1:1 for aligned edits; scattered edits amplified by (coded ÷ CDC) chunk size, which is a dial |
| "lmz beats ZipLLM" / any **> 54.1%**; also the pair "41.83% today / 51.53% once fixed" | different corpora for the first; the second was a real bug, now fixed | matched corpus: **lmz as shipped 51.55%** against ZipLLM's method at 40.82% — **+10.7 points**. The filename-order dependence was lmz's delta-source choice and was fixed 2026-09-03 (`77fd720`); the archive is invariant under member order |
| **any end-to-end ratio taken before `buffer.py` and the `mincore` check** — the general form, covering 0.94×, "first touch 2.4 GB/s" and anything else from a cold protocol predating them | the destination was allocated inside the timed region, which adds a fixed cost to both routes and so drags the ratio toward 1.0 — biased **up to 14% toward the coded route**; and its coldness is unestablished, because warm-plus-allocation at 1.8–2.1 GB/s is indistinguishable by rate from a genuine first touch at 1.6–2.0 | not replaced — **withdrawn pending storage that is not behind a host cache**. One entry rather than one per figure, because the defect is the method, not the numbers |

### Which figures this does *not* touch, and why

The doubt above is about link rates near the decoder, where a factor of two in
the link decides the verdict. It does not reach the slow-link results, and the
reason is arithmetic rather than assertion.

On the 9p mount the plain route measures 0.117–0.146 GB/s cold, and this box's
*warm* 9p rate is 0.26. Both are **more than twenty times** below the decoder's
5.95 GB/s, so `min(1/f, decode/link)` saturates at the ratio cap either way:
**the verdict cannot flip on cache state there.** The 1.45× and the vLLM
decomposition's 2.62× / 1.38× are therefore unaffected by whether anything was
cold, which is why they carry the roadmap while the first-touch row is withdrawn.

That is also the general test for whether a host-cache doubt matters to a given
row: if the *warm* rate is still far below the decoder, cache state changes the
size of the win and not its sign.

### One figure the sweep was told to retire and did not

**The 9p `1.45×` stands.** It was listed for retirement on the grounds that its
archive "was very likely `CODEC_BF16C`". It was not: that section states its own
conditions — *"1 MiB chunks, so every chunk is GPU-readable `CODEC_BF16`"* —
cache dropped before every run, five runs, median, byte-identical, reaching 97%
of the 1.49× the ratio allows.

It is also not superseded by the vLLM 3.60×, because they measure different
things. 1.45× is coded against plain **through the same transport**: the codec
alone. 3.60× is our coded route against *vLLM's* default loader: codec **and**
reader. The figure that corresponds to 1.45× is the vLLM decomposition's codec
term, **1.38×** — two harnesses, two link types, a week apart, 1.45 and 1.38.
That is a cross-check, and it is worth more than either number alone.

---

## The allocation experiment, and a worse thing it found — 2026-09-03

Run to settle whether `strategy.md` §1's first-touch row pays or taxes, by
redoing the 0.94× measurement with the destination supplied instead of
allocated inside the timed region. It did not settle that. It found two other
things, and the second is the one that matters.

### It could not be run cold, and that is now a demonstrated limit

Fifteen point seven gigabytes of never-before-read copies were staged on local
ext4 and read with `mincore` confirming **0.0% Linux residency** every time. The
plain route read at **5.31–5.64 GB/s**. That is the host-cache-warm rate; a
genuine first-touch read on this box is 1.6–2.0.

Restaging with the exact protocol that *did* produce cold reads earlier the same
day — `posix_fadvise(DONTNEED)` on each file immediately after writing it —
produced **5.34 GB/s** again. So the protocol is not the variable. **Whether a
read on this box is cold is a property of Windows host state that the guest can
neither control nor observe**, and it changed between one hour and the next with
no action from here. Cold first-touch on local NVMe is not reliably obtainable,
which is a stronger statement than the earlier "the host cache does not evict
under pressure" and supersedes any plan to get one by trying harder.

### The distortion the experiment was built to measure

Same files, same host-warm link, fresh process per run, n=4, the *only* variable
being whether the destination is inside the timed region:

| destination | plain | coded | ratio |
|---|---|---|---|
| supplied and pre-faulted outside the clock | 5.36 GB/s | 3.51 GB/s | **0.65×** |
| allocated inside the clock (`bytearray`) | 1.82 GB/s | 1.37 GB/s | **0.76×** |

**Allocating inside the timing flatters the coded route by 14%**, 0.65 → 0.76.
It adds a fixed cost to both routes, which drags any ratio toward 1.0 — so it
helps whichever route is losing. Every figure in this file taken with the
allocation inside the clock is therefore biased *towards* compression, which is
the direction that does not feel like an error.

### The worse thing: "cold" and "warm plus allocation" are the same number here

The plain route with allocation inside the clock reads **1.82 GB/s** on a file
demonstrably served from the host cache at 5.36. Check it against the parts:

    warm read, 1.872 GB at 5.36 GB/s          0.349 s
    + bytearray(1.87 GB), fresh process       0.550 s
    = 0.899 s                              -> 2.08 GB/s

The 0.94× run that contradicts §1 reported its plain route at **1.94 GB/s** and
took that for a cold read. A genuine cold read is 1.6–2.0 GB/s. Warm plus
allocation is 1.8–2.1. **The two are not distinguishable by rate alone**, and
this file has no record of `mincore` or any independent check being run on those
measurements to tell them apart.

So the claim is not that those figures are wrong. It is that **nothing in the
record establishes they were cold**, and the arithmetic offers an equally good
explanation in which they were warm and the allocation supplied the difference.
That applies to the 0.94×, to "ext4 on NVMe, first touch 2.4 GB/s", and to
anything else derived from a cold protocol that predates `buffer.py` and the
`mincore` check.

**What would settle it** is not more runs on this machine — the first section
above is why. It is the same answer as the two open questions already on the
list: different hardware. Storage not behind a host cache makes cold reads
obtainable and tells all three of these apart at once.
