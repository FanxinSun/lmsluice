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
