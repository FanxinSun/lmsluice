# load — the model load path, on whatever silicon the box has

A runtime that gets model weights off storage and into memory as fast as the
machine allows, by decoding lmz archives on the fastest processor present —
a discrete GPU, the integrated one, or the CPU — and choosing between them by
measurement rather than by assumption. Pure software, and nothing about the
bytes changes: every weight arrives byte-identical.

The target is **loading**, not training and not inference. `vram/` owns
training residency and has ruled out inference on its own roofline. This is
the third case and the only one that happens on every machine, every time a
model is opened.

## The one ratio

Compression is free on a path when the decoder is faster than the tier it is
feeding from. Everything here follows from that:

    decode_rate  >  source_rate      →  the archive loads faster than the file

lmz already wins that comparison decisively on one path and loses it on
another, and the second case is why this project exists:

| path | source | decoder | verdict |
|---|---|---|---|
| NVMe → VRAM, CUDA decoder | 28.8 GB/s PCIe | **418 GB/s** | free by 14×; **1.46× faster** than plain safetensors |
| NVMe → RAM, CPU decoder | 6.34 GB/s NVMe | **2.0 GB/s** | **2.85× slower** than plain safetensors |

The second row is the whole problem — **against this box's disk**. On a
machine without an NVIDIA GPU and with a Gen4 NVMe under it, installing lmz
today costs about **2.85× on load time** in exchange for 34.7% less disk, and
that is a bad trade taken by default on most of the machines lmz runs on. It
will not be fixed by threads: lmz's own README records that past four threads
the CPU decoder *"runs into the memory bus at about 2 GiB/s."*

It is not fixed by a slower disk either, but it *inverts* on one: the same
2.0 GB/s decoder is a 1.53× win on a machine with a SATA SSD. Which disk the
archive is being read from is half the gate, and the next section makes it an
axis rather than a constant.

## The gate

Whether a given device can pay is decided by two numbers, and the first
version of this file had only one of them. An entropy decoder spends compute
to avoid reading bytes, so it is bounded from both sides — and the thing it
has to beat is not a constant, it is *the disk in the machine beside it*.

    decode  =  min( C / k , BW / (1 + f) )        the decoder

    speedup =  min( 1/f , decode / disk )         what that is worth

`C` is the device's compute, `BW` its streaming bandwidth, `f` the coded
fraction (0.653 on model weights), and `k` the decoder's compute cost per
decoded byte. `gate.py` evaluates it; `python3 gate.py` prints the tables
below for any device you add to it.

**`k` is not measured, and that is the honest state of this project.** One
number exists — lmz's CUDA decoder at 418 GB/s on an RTX 5080 — and it fits
both readings at once: it is 84% of that card's bandwidth bound (497), and it
is exactly `C / k` for `k = 128`. On the card that produced it the two
readings differ by 16%. On a 2-CU display adapter they differ by **8×**, and
that is the device this project's case rests on. Scaling that one point by
FP32 throughput — what this table did before — silently picks the compute
reading, which is to say it assumes what was never measured.

So every device gets an interval: the compute reading as its floor, the
bandwidth bound as its ceiling.

| device | FLOP/byte | compute floor | b/width ceiling | decode GB/s | source |
|---|---|---|---|---|---|
| RTX 5080 (dGPU) | 65 | 418 | 497 | **418–497** | measured, `MEASURED.md` |
| M4 Max (unified) | 31 | 112 | 278 | 112–278 | spec arithmetic |
| Strix Halo (unified) | 69 | 116 | 130 | 116–130 | spec + 215 GB/s measured, blueprint §1 |
| Radeon 780M (iGPU) | 72 | 33 | 36 | 33–36 | spec arithmetic |
| Phone SoC GPU (class) | 44 | 19 | 34 | 19–34 | spec class figure |
| Radeon 2-CU (iGPU) | 10 | 4.4 | 36 | **4.4–36** | measured, `MEASURED.md` |

**No shipping device in that table is bandwidth-bound at `k = 128`** — every
one sits below the 78 FLOP/byte crossover. If the decoder really costs what
the compute reading says, then decode is a compute problem *everywhere*, and
FLOP-per-byte is not the discriminator this file used to claim it was:
absolute compute is. That inversion follows from one unmeasured constant,
which is the argument for measuring it.

Then the second axis, the one entirely absent before — the machine's own
storage:

| device | NVMe Gen5 12.0 | NVMe Gen4 6.34 | UFS 4.0 | SATA SSD 0.55 | eMMC/SD 0.30 |
|---|---|---|---|---|---|
| RTX 5080 | pays | pays | pays | pays | pays |
| M4 Max | pays | pays | pays | pays | pays |
| Strix Halo | pays | pays | pays | pays | pays |
| Radeon 780M | pays | pays | pays | pays | pays |
| Phone SoC GPU | pays | pays | pays | pays | pays |
| Radeon 2-CU | measure 0.37× | measure 0.70× | pays 1.10× | pays | pays |

Bare "pays" is the saturated case: the decoder beats the disk by more than
`1/f`, so the archive's ratio *is* the load speed and nothing faster helps.
"measure" means the interval straddles that disk — undetermined until `k` is
pinned, which is not a negative result. **The bottom row changes verdict
three times across one table row**, and nothing about the device changed. A
gate calibrated against one machine's NVMe is not a gate.

**The founding claim survives the worst reading.** Even if the pessimistic
`k` is right, the smallest integrated GPU AMD ships decodes at 4.4 GB/s
against the CPU decoder's 2.0 — **2.2×, on a 2-CU display adapter**. That is
still the case for this project in one line. What does not survive is reading
that row's verdict against one desktop's NVMe as *the* verdict.

**FP32 FMA is a proxy and the wrong unit.** rANS decoding is integer work
with dependent shared-memory loads, so `C` is measured in FP32-FLOP
equivalents only because that is what `probe/igpu-bench.ps1` can measure on
any device with nothing installed. Every rate derived from it is a reason to
go and measure, never a result — which is what `probe/` is for, and pinning
`k` on one low-FLOP/byte device is the highest-value measurement this
repository can make.

## What this is not

**Running the model on the iGPU.** `vram/`'s roofline settles it: inference
decode does one multiply-add per byte of BF16 against about 4,132 needed,
*"short by a factor of four thousand, and no scheduling recovers it."* The
integrated GPU should decode the archive, not run the network.

**A second codec.** lmz produces bytes and turns them back into bytes; it does
not decide when. That line does not move because the device changed.

## Where the boundary is

| | whose |
|---|---|
| the coded stream, its tables, its blocks, its index | **lmz** |
| a standalone GPU decoder with a stable ABI — CUDA, Metal, Vulkan | **lmz** |
| tensor → block addressing | **lmz** |
| which device decodes on this machine, and the probe that decides | **load** |
| dispatch, queue depth, staging buffers, overlap with I/O | **load** |
| adapters — safetensors-compatible open, PyTorch, llama.cpp | **load** |
| training residency across VRAM / RAM / SSD | `vram/` |

The split against lmz is the one lmz's own `docs/gpu-residency-handover.md`
already draws, applied to a different consumer: lmz ships decoders, this
decides which one runs and when. The kernels stay in `lmz/gpu/` precisely so
there is one ABI and not two.

## Status

Nothing here runs yet. What exists:

- `gate.py` — the gate above as ~150 lines of dependency-free Python:
  devices and storage tiers as parameters, the interval, the verdict matrix.
  `python3 gate.py`. Add a machine to it rather than re-deriving the
  arithmetic in prose.
- `probe/igpu-bench.ps1` — a self-contained Direct3D 11 compute benchmark
  (PowerShell plus inline C#, no toolkit, no installs) that enumerates every
  DXGI adapter and measures streaming read bandwidth, FP32 FMA throughput and
  a multithreaded CPU baseline. It is the source of the table above and it is
  the gate: run it on a machine and it says whether decoding there can pay.
- `MEASURED.md` — this box's numbers and the conditions behind them.

**What is not blocked on anything**: pinning `k` on one low-FLOP/byte device
— a decode measurement, not an FMA proxy — which turns six intervals above
into six numbers and settles whether the 2-CU row pays against a fast disk.

What is blocked on lmz, in order:

1. **The CUDA kernel reading v7.** `--shared-tables` writes a format worth
   3.8× on the GPU decoder and nothing collects it yet.
   `lmz_gpu_decode_batch_dev` already takes a shared header in `hdr`, so this
   is wiring.
2. **The Metal port**, written but never run —
   `lmz/scratchpad/gpu/metal/`.
3. **Vulkan**, which is what actually opens this project up: AMD and Intel
   iGPUs, Windows and Linux, driver-level with no toolkit to find.

The reasoning and the measurements behind that order are in
`lmz/docs/portable-decoder-handover.md`.
