# probe — can this machine decode faster than its disk?

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
