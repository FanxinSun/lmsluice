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
