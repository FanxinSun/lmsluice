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

The second row is the whole problem. On any machine without an NVIDIA GPU,
installing lmz today costs about **2.85× on load time** in exchange for 34.7%
less disk. That is a bad trade taken by default on most of the machines lmz
runs on, and it will not be fixed by threads: lmz's own README records that
past four threads the CPU decoder *"runs into the memory bus at about 2
GiB/s."*

## The gate

Whether a given device can pay is decided by one number — how much compute it
has per byte of bandwidth. An entropy decoder spends compute to avoid reading
bytes, so it needs slack in exactly that ratio. Measured on this box
(`probe/igpu-bench.ps1`, and `MEASURED.md` for conditions):

| | GFLOP/s FP32 | GB/s | FLOP/byte | est. decode | vs 6.34 NVMe |
|---|---|---|---|---|---|
| RTX 5080 | 53,691 | 822 | 65 | 418 (measured) | 66× |
| Radeon 780M (12 CU) | ~4,300 | ~60 | ~70 | ~34 | 5× |
| Radeon iGPU, 2 CU | 567 | 59.4 | 9.5 | ~4.4 | 0.7× |
| CPU, 16 threads | — | 54.3 | — | 2.0 (measured) | 0.3× |

Read the last column as: how much faster than the disk this device can turn
an archive back into weights. Above 1, compression is free on the load path
and lmz's ratio becomes load speed. Below it, lmz is a tax.

**The estimates are compute-scaled from the measured CUDA kernel and are not
predictions.** rANS decoding is integer and shared-memory-latency bound, not
FP32-throughput bound, so the middle rows are an order of magnitude and a
reason to go and measure — which is what `probe/` is for.

**Even the worst iGPU in that table beats the CPU decoder by 2×**, and it is a
2-CU display adapter, the smallest integrated part AMD ships. That is the
case for this project in one line.

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

- `probe/igpu-bench.ps1` — a self-contained Direct3D 11 compute benchmark
  (PowerShell plus inline C#, no toolkit, no installs) that enumerates every
  DXGI adapter and measures streaming read bandwidth, FP32 FMA throughput and
  a multithreaded CPU baseline. It is the source of the table above and it is
  the gate: run it on a machine and it says whether decoding there can pay.
- `MEASURED.md` — this box's numbers and the conditions behind them.

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
