# lmsluice — standing rule for every agent working here

*The parent repository's `CLAUDE.md` (one level up, if you have it) carries
this in full, plus the environment, network and scope rules. This copy
exists because `lmsluice/` is a repository of its own.*

## Where work goes: `docs/boundary.md`

**lmz is the compressor/decompressor; lmsluice is the AI-model transport
facilitator.** Before adding anything to either tree, run the three-question
test in [`docs/boundary.md`](docs/boundary.md): does it change if the *format*
changes (lmz), if the *machine* changes (lmsluice), or both — in which case it
is two pieces and must be split. That document also carries the five invariants
and the loan register; the loans are load-bearing and must not be removed.

## This PC is the instrument, not the target — and here it is the whole point

This project's subtitle is *"on whatever silicon the box has"*. Its entire
value is that it decides, **per machine**, whether decoding an archive
beats reading a plain file — on a machine with no discrete GPU, no CUDA,
a 2-CU display adapter, a SATA SSD, UFS flash, or unified memory. A gate
calibrated to the 9800X3D / RTX 5080 box these numbers came from would be
the one bug this project cannot have.

Which makes the rule concrete and unusually sharp here:

- **The gate is a curve, not a table of devices.** `gate.py` holds it:
  `decode = min(resident_lanes × clock / k, peak_DRAM × achieved_fraction /
  (1 + 1/expansion))` against *that machine's* storage rate. Device rows
  are samples of the curve. Which term binds is a property of the device,
  and residency comes from the device's own shared-memory pool, not its cap.
- **`k` is per kernel, so it is per backend — and it has never been measured
  on a low-FLOP/byte device.** Two measurements exist, each by a grid sweep
  at fixed byte traffic:
  - lmz CUDA `k_shared` on an RTX 5080: 217–248 lane-cycles per decoded
    byte, bandwidth-bound at full occupancy.
  - lmz's Metal shared kernel on an Apple M4 (unified memory): 269–313 at
    the kernel's IOReport-measured clock. It is compute-bound at ~28% of
    DRAM, and its `k` *rises* with co-residency. A CPU streaming ~80 GB/s on
    the same bus still slowed it by up to 29%.

  Never carry one backend's `k` to another: CUDA's predicts the M4 26–44% too
  fast. `gate.py` predicts each row with its own backend's constants and marks
  borrowed ones; the Vulkan rows borrow today. A 2-CU iGPU, where the founding
  claim lives, is still unmeasured. Pinning `k` there remains the
  highest-value measurement this repository can make.
- **Compare against the machine's own disk.** The WSL2 box's Gen4 NVMe reads
  6.34 GB/s. On the Apple M4 Mac mini the internal NVMe reads 3.12 GB/s cold
  and a USB SSD 0.373. A SATA SSD is a tenth of the first, phone UFS a third,
  a Gen5 array twice it. The verdict flips inside that range for the same
  device. It also assumes the route delivers the kernel's rate: lmz's
  shipped Python GPU path delivers ~2 GB/s on the M4 against a 25.2 GB/s
  kernel.
- **FP32 FMA is a proxy and the wrong unit.** rANS decoding is integer work
  with dependent shared-memory loads. On the M4 one dependent threadgroup
  load costs ~69 cycles, and a decode step ~4 of them. `probe/` says so;
  keep saying so. An FMA rate may check an ALU count, never set a decode
  rate.
- **The probe must reach the machines that matter.** Direct3D 11 covers
  Windows (`probe/igpu-bench.ps1`). Metal covers Apple hosts with nothing to
  install (`probe/metal-bench.js`, JavaScript for Automation via `osascript`;
  a stock Mac has no Python without the Command Line Tools). Vulkan, for AMD
  and Intel iGPUs on Linux, is still missing. A probe that needs a toolkit
  cannot run on the machines it most needs to measure — that constraint is
  load-bearing, not incidental. A probe that prints *this* machine's numbers
  on another one is the same bug in a different place.

Label every number with its machine and conditions; mark projections as
projections; never present "works on this box" as completion.
