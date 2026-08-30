# load — standing rule for every agent working here

*The parent repository's `CLAUDE.md` (one level up, if you have it) carries
this in full, plus the environment, network and scope rules. This copy
exists because `load/` is a repository of its own.*

## This PC is the instrument, not the target — and here it is the whole point

This project's subtitle is *"on whatever silicon the box has"*. Its entire
value is that it decides, **per machine**, whether decoding an archive
beats reading a plain file — on a machine with no discrete GPU, no CUDA,
a 2-CU display adapter, a SATA SSD, UFS flash, or unified memory. A gate
calibrated to the 9800X3D / RTX 5080 box these numbers came from would be
the one bug this project cannot have.

Which makes the rule concrete and unusually sharp here:

- **The gate is a curve, not a table of devices.** `gate.py` holds it:
  `decode_rate = min(C / k, BW / (1 + f))` against *that machine's*
  storage rate. Device rows are samples of the curve, and the FLOP-per-byte
  ratio `C / BW` decides only which of the two terms binds.
- **`k` — the decoder's compute cost per decoded byte — is not measured.**
  One point on one device (418 GB/s on an RTX 5080) fits both readings:
  it is 84% of that card's bandwidth bound *and* it defines `k = 128` if
  compute binds. The two readings differ by 8× on a 2-CU iGPU. Scaling
  that point by FP32 throughput assumes the answer. Pinning `k` on one
  low-FLOP/byte device is the highest-value measurement this repository
  can make.
- **Compare against the machine's own disk.** 6.34 GB/s is this box's
  Gen4 NVMe under WSL2. A SATA SSD is a tenth of it, phone UFS a third,
  a Gen5 array twice it. The verdict flips inside that range for the same
  device.
- **FP32 FMA is a proxy and the wrong unit.** rANS decoding is integer
  work with dependent shared-memory loads. `probe/` says so; keep saying
  so, and treat every rate derived from it as a reason to measure.
- **The probe must reach the machines that matter.** Direct3D 11 covers
  Windows; Vulkan is what reaches AMD and Intel iGPUs on Linux, and Metal
  the Apple hosts. A probe that needs a toolkit cannot run on the machines
  it most needs to measure — that constraint is load-bearing, not
  incidental.

Label every number with its machine and conditions; mark projections as
projections; never present "works on this box" as completion.
