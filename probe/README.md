# probe — can this machine decode faster than its disk?

> **Planning reference — 2026-10-03:** this probe record is historical. Its
> proposed ports/calibration tasks are **retired as an independent work order**
> and remain held candidates under [SLUICE-OPT-0](../plan/PLAN.md).
> Arithmetic/occupancy estimates do not establish decoder throughput or native
> Apple GPU support. Historical macOS host measurements are in `MEASURED.md`;
> they do not certify this GPU probe or MPS placement.

One question, asked of whatever silicon is present, before any of it is
trusted with a load path.

## igpu-bench.ps1

A self-contained Direct3D 11 compute benchmark. It enumerates every DXGI
adapter — discrete, integrated and the WARP software rasterizer — and for
each one measures:

- **streaming read bandwidth**, coalesced, grid-strided. This sets decode
  speed, because turning an archive back into weights is bound by how fast
  the coded bytes and the output can move, not by arithmetic.
- **FP32 FMA throughput**, four independent chains in registers. This stands
  in for the compute an entropy decoder spends to avoid reading bytes.
- a **multithreaded CPU baseline** over the same buffer, because on an
  integrated GPU both engines draw from one DDR bus and the honest comparison
  is never iGPU against nothing — it is iGPU against CPU.

It then divides bandwidth by model size to project a decode ceiling for a few
model shapes, which is the same arithmetic the load path will do for real.

## Nothing to install

This is the constraint the tool was built under and it should stay that way:
a probe that needs a toolkit cannot run on the machines it most needs to
measure. It uses only what ships with Windows —

- `d3d11.dll` for the device, buffers and dispatch,
- `d3dcompiler_47.dll` to compile the HLSL kernels at runtime,
- `dxgi.dll` to enumerate adapters,
- and PowerShell's own C# compiler for the interop.

No CUDA, no Vulkan SDK, no Python, no packages. `powershell -File` and it
runs.

**COM is called through raw vtable slots rather than declared interfaces.**
Declaring `ID3D11DeviceContext` properly means 115 methods in exact order;
reading four slots off the vtable is far less to get wrong. The slot numbers
are from `d3d11.h` and `dxgi.h` and are commented at their use.

## Reading the result

The number that decides everything is **FLOP per byte** — compute divided by
bandwidth. A decoder spends the first to save the second, so it needs slack
in that ratio. An RTX 5080 has 65, a Radeon 780M about 70, and a 2-CU display
adapter 9.5, which is why the last one is the floor of the category rather
than a fair sample of it.

Compare the projected decode rate against the machine's storage. Above the
disk, compression is free on the load path and every point of lmz's ratio
becomes load speed. Below it, the archive is a tax.

## Traps

**Take the memory clock before taking the bandwidth.** On the box in
`../MEASURED.md`, the same iGPU measured 22 GB/s and 59 GB/s on either side
of enabling EXPO, and at the slower clock it also produced a convincing but
entirely spurious cliff at the UMA carve-out boundary.

**Get clear of cache.** A 128 MB buffer sits inside a 9800X3D's 96 MB
V-Cache and reports the CPU at twice its real DRAM bandwidth. Sweep the size;
the steady state is the answer.

**Take a median.** A single run on a desktop with a browser open varies by
40%.

## Not yet written

A Linux and macOS equivalent. The same three measurements through Vulkan
compute would cover both, and would share the shape of the eventual decoder
backend — which is an argument for writing it after that backend rather than
before it.

## Determining the shared-memory pool empirically — feasible, and how — 2026-08-30

`gate.py`'s largest open parameter is the 2-CU adapter's shared-memory **pool per
compute unit**. It decides whether the smallest integrated GPU AMD ships beats
the CPU decoder, and it is **exposed by no interface that adapter offers**: core
Vulkan has no such query, and `VK_AMD_shader_core_properties` and
`shader_core_properties2` — both present on the device, both queried — carry no
LDS field.

**It is nonetheless measurable here, with no downloads and no SDK**, because
`igpu-bench.ps1` already has every piece: a D3D11 device on a chosen adapter, the
runtime HLSL compiler in `d3dcompiler_47.dll`, buffers, UAVs, dispatch and
timing. All of it ships with Windows, which is the same constraint that made the
existing probe worth writing.

**The method.** A compute shader declaring `groupshared` memory the size of the
decoder's request, at the decoder's group size, doing enough work per group to
dominate dispatch overhead. Then sweep the *number of groups dispatched*: if `C`
groups are resident at once, elapsed time rises as `ceil(G / C)`, so the
staircase in time-against-`G` gives `C` directly. Divide by the adapter's
**2 compute units**, queried, and the pool follows as at least
`blocks_per_CU × 31,744`.

That is lmz's own grid-sweep method pointed at occupancy instead of throughput,
which is a reason to trust the shape of it.

**It must be calibrated on the RTX 5080 first, and that is not optional.** That
device's occupancy is independently known — lmz measured 3–4 blocks per unit, and
`gate.py` derives 3 from queried `sharedMemPerMultiprocessor`. A method that
cannot recover 3 there has not earned an answer on a device where nothing can
check it. Calibrate, then measure.

**What it cannot settle.** D3D11 caps `groupshared` at 32 KiB per group, so this
can find the pool but cannot test whether the adapter's *opt-in* cap exceeds the
32,768 Vulkan reports. The second of `gate.py`'s two conservatisms therefore
survives this measurement, and the floor stays a floor.

**Cost:** hours, in the existing script's style — not a backend port. Which is
the point: the parameter that decides the founding claim may be obtainable
without one.


## metal-bench.js (macOS)

A zero-install Metal benchmark for macOS (`probe/metal-bench.js`), serving as the macOS counterpart to `igpu-bench.ps1`.
It measures device facts across all enumerated Metal devices, DRAM streaming bandwidth, sustained FP32 FMA / int32 throughput, dependent threadgroup memory chains, and the occupancy staircase that independently cross-checks decoder grid sweeps.

### Design and Zero-Install Constraints

On a clean stock macOS installation, `/usr/bin/python3` triggers a Command Line Tools prompt, and developer toolkits (Xcode, `clang`, `xcrun`, `metal`, `swift`) are absent. `metal-bench.js` is written entirely in **JavaScript for Automation** (JXA, executed via `osascript -l JavaScript`):
- **Runtime Shader Compilation:** Metal Shading Language (MSL) source strings are compiled dynamically on the target device via `device.newLibraryWithSourceOptionsError()`.
- **Foreign Function Interface & Struct Bridging:** C and Objective-C APIs are bound without external wrappers. `MTLSize` structs are bridged using explicit 64-bit integer packing in `NSData` and dispatched via `NSInvocation`.
- **Dynamic Clock & P-State Residency:** GPU operating frequencies are measured without `sudo` by dynamically loading `/usr/lib/libIOReport.dylib` via `$.dlopen` and binding `IOReportCopyChannelsInGroup` / `IOReportCreateSamplesDelta` for the `GPU Stats` / `GPUPH` channels, cross-referenced with `pmgr` voltage states.
- **Stock Utility Confinement:** Operating system inspection is strictly confined to tools shipping with macOS (`ioreg`, `sysctl`, `sw_vers`, `pmset`, and `/usr/bin/perl` for memory contention tests).

### Verification and Hygiene

Every benchmark kernel implements bit-exact validation:
- **Streaming Read:** Compares the GPU-reduced per-threadgroup sum against the exact closed-form analytical sum $\sum_{i=0}^{N-1} i \pmod{2^{32}}$.
- **Copy:** Verifies sample words across memory buffers.
- **FP32 FMA & Int32:** Mathematically models convergence and compares register chains against single-precision JS thread emulation.
- **Dependent Chain & Staircase:** Emulates the pseudo-random permutation hops in JS and verifies the final state written by thread 0.
- **GPU Lock:** Coordinates via atomic lockfile (`m4-gpu`) with background GPU workloads, logging foreign AGX user client time deltas from `ioreg`.

### Calibration & Staircase Findings

The occupancy staircase measures wave-period transitions by dispatching $G = 1 \dots (cores \times 24)$ threadgroups:
- For the compiled direct kernel shape ($T = 128, S = 19,456$ B compiled static footprint), the curve displays a sharp step transition at $G = 40$ across 10 GPU cores, establishing **4 TGs per core**. This directly cross-checks and confirms the knee identified by the independent decoder grid sweep.
- Per-core threadgroup memory allocation bounds are derived from clean step transitions: for the M4, $77,824 \le \text{pool} < 83,968$ B (likely $\approx 80$ KiB), accommodating $4 \times 19,456$ B and $2 \times 32,768$ B blocks per core at maximum group memory.

### Usage

```bash
# Quick verification run (~15s)
osascript -l JavaScript probe/metal-bench.js --quick

# Full verified benchmark run writing raw curves to JSON
osascript -l JavaScript probe/metal-bench.js --json results/probe_m4/metal-bench-<timestamp>.json

# Proven zero-install execution in pristine environment without toolkits
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=$HOME DEVELOPER_DIR=/nonexistent osascript -l JavaScript probe/metal-bench.js --quick
```

## igpu-pool.ps1 — the 2-CU pool measurement, as a kit (Windows, D3D11)

`gate.py`'s largest open parameter is the smallest AMD iGPU's shared-memory **pool per compute
unit**; it decides whether that part beats the CPU decoder (1 resident block per CU, or 3). No
interface reports it. The method written up above is implemented here as a script that needs
nothing installed: Windows PowerShell 5.1, `d3d11.dll`, `d3dcompiler_47.dll`, `dxgi.dll`.

```powershell
powershell -File igpu-pool.ps1 -Adapter -1                  # list adapters, exit
powershell -File igpu-pool.ps1 -Adapter 1 -Units 2          # measure adapter 1, which has 2 CUs
```

`-Units` is typed in (D3D11 cannot query it). The script writes `igpu-pool-<n>-<time>.json` with
the **raw timing curves** for seventeen shapes (the decoder's own threadgroup requests, an S sweep,
a T sweep), each calibrated so one group runs ≥ 0.8 ms, each kernel output checked against the same
chain computed on the CPU. It prints a preliminary first-jump estimate only.

```
python3 igpu-pool-analyze.py run1.json run2.json --expect lmz_cuda_shared_dynamic=252
python3 igpu-pool-analyze.py --selftest
```

The analyzer (standard library only) applies one set of rules to every shape: a jump is measured
against the local plateau, capacity is an *interval* on the coarse grid, a shape is clean only with a
second step near twice the capacity and exactly one integer groups-per-unit inside the interval, and
the pool bound is the intersection over shapes that are **clean and equal in every run given**. It
also reads `metal-bench.js` output. Checked on the real M4 data: one run gives 81,920 ≤ pool < 83,968
with a loud single-run warning; with a second run in which the boundary shape drifts, as it did, that
shape is excluded and the bound is 77,824 ≤ pool < 83,968, which is what `gate.py` carries.

**Calibrate first.** Run it on an RTX 5080 with `-Units 84` and require the (192, 31,744) row's
capacity interval to contain 252 (3 groups per SM; lmz measured 3–4 and the queried pool of 102,400 B
gives 3). A method that cannot recover that has not earned an answer on a device where nothing can
check it.

**Status.** Run on 2026-11-13 in a Windows 11 VM (Parallels, PowerShell 5.1) on the Apple M4 Mac mini,
adapter "Parallels Display Adapter (WDDM)", `-Units 10`: the C# interop compiled, the HLSL compiled
through `d3dcompiler_47`, all kernel outputs matched the CPU check, and JSON was written, in `-Quick` and
in two full runs. Analysed together, the staircase reproduces the *native* Metal result on the same
hardware (4 groups per unit at 19,456 B, 3 at 24,576, 2 at 28,672 and 32,768; pool 77,824 <= pool <
86,016 B against 77,824 <= pool < 83,968 B measured natively): the method recovers an independently
known occupancy through a different API path. **That calibrates the method, not any other device**: it
is a virtualised GPU over Metal, the unit count was supplied by hand, and the shapes whose first wave
exceeds `-MaxGroups` (the S = 1,024 T-sweep) never leave it. **Not yet run on real AMD, Intel or NVIDIA
hardware.** Raw files: `results/vmtest/` (kept in git as evidence).
