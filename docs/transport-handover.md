# The transport runtime — what was built, measured, and got wrong

*The whole of the session that turned `lmsluice/` from a README and a probe
into something that runs: what each piece is, why it is shaped that way, the
numbers it produced and the conditions they hold under, two corrections that
changed the design after it was written, and what is still open. Written so
that none of it has to be rediscovered.*

[← back to the README](../README.md)

## Why this exists

The ask was for an **I/O transporting facilitator** — the layer that moves
model bytes fast, next to the compressor that decides how many of them there
are. `lmz/` had a codec and no transport; `lmsluice/` had a scope document, a
Direct3D 11 device probe, `gate.py`, and the line *"nothing here runs yet."*

It runs now. ~2,700 lines of standard library plus lmz, 34 tests, both routes
verified SHA-256 identical to the original checkpoint.

## What is here

| file | what it is |
|---|---|
| `gate.py` | **not mine** — the device-side model, see §4 |
| `lmsluice/transport.py` | two stages, sized independently, overlapped across a bounded queue |
| `lmsluice/plan.py` | the routing decision, from measured rates |
| `lmsluice/probe.py` | cold reads, sustained writes, decode swept for thread count |
| `lmsluice/rates.py` | the measured profile, cached per machine, fingerprinted |
| `lmsluice/source.py` | local files and HTTP ranges behind one `pread` |
| `lmsluice/archive.py` | every use of lmz's internals, in one file |
| `lmsluice/model.py` | `map` / `load` / `stream`, one API over either route |
| `lmsluice/cli.py` | `probe plan info bench get write doctor` |

### 1. Splitting fetch from decode is the engineering claim

`lmz.decompress` reads a payload and decodes it **inside one worker**, so the
thread count that sets decode throughput also sets how many reads are
outstanding. Those two want opposite things: a device reaches its rated speed
only with a dozen requests in flight, while lmz's own measurement is that past
two threads the interpreter between native calls turns into GIL contention and
a third core subtracts.

`transport()` runs F fetch threads and P place threads across a
`queue.Queue(maxsize=inflight)`. The queue is the memory bound and also the
diagnosis: always full means placing is behind, always empty means fetching
is, and `Report.limited_by` reads that off the stall counts. Placement is
unordered because every job knows its own destination offset; a consumer
needing order would need a reorder buffer this does not have.

### 2. Blocks overhang tensors, and that is not an edge case

A block is the smallest decodable unit and it does not stop where a tensor
does. A caller that allocates only the bytes it asked for has allocated too
few and the decode runs off both ends. `Archive.cover(spans)` returns the
plain range the covering blocks actually fill, and `place(..., clip=)` decodes
a straddling block into scratch and copies only the part inside. Every partial
read hits this — it is not an edge case, it is the normal case, and it was
found by `tensor()` failing on the coded route after `load()` had passed.

### 3. Two measurements about lmz's decoder worth carrying

Both taken while tuning the place stage, single-threaded, on the fixture below:

- Passing a payload as `bytes` rather than a `memoryview` slice measured
  **23% faster** into `decode_chunk`.
- CRC verification costs about **40%** of decode.

Neither is a reason to change anything — the checksum is why the bytes can be
trusted — but a decode rate quoted without saying which was in force is not a
measurement.

### 4. `gate.py` and `plan.py` are separate on purpose

They state one identity, `speedup = min(1/f, decode/link)`, and they are two
files because they answer different questions:

- **`gate.py`** predicts a decode rate for a machine that is *not here*, from
  compute and bandwidth, bounded from two sides, carrying the unmeasured
  constant `k` as an interval rather than picking a reading. It has **no
  imports** so it can be copied onto someone else's box and run.
- **`plan.py`** predicts nothing. It prices routes from rates somebody
  measured on the machine in front of it.

Neither can import the other without losing what makes it useful, so the
agreement is pinned by `test_agrees_with_the_standalone_gate`, which asserts
them equal across 80 combinations of f, link and decode. **The duplication is
deliberate and verified; do not "fix" it by merging them.**

`plan.py` uses `gate.py`'s notation (`f` for the ratio) for this reason.

## What was measured

Full conditions in [`../MEASURED.md`](../MEASURED.md). Fixture is a 1.87 GB
BF16 checkpoint (`~/.cache/lmz-bench/model.safetensors`, 111 tensors) and its
lmz 1.2.0 archives in `~/.cache/yfce-load-bench/`. Machine is the WSL2 desktop
in MEASURED.md, which is **one sample point and an unrepresentative one**.

| | GB/s | note |
|---|---|---|
| read, first touch ever | 2.38 | the only genuinely cold number |
| read, cold to the guest | 5.6–6.4 | the Windows host's cache under the vhdx |
| read, warm | 33–40 | not a disk measurement |
| write, fsync included | 1.18–1.46 | what a checkpoint save costs |
| lmz decode, 8 MiB blocks | 1.03–1.10 | 8–16 threads, swept |
| lmz decode, 64 KiB blocks | 0.83–0.89 | `--mapped` archives |
| lmz encode, whole file | 1.25 | 0.15 for `--mapped` |

**The plan was checked against reality twice**, which is what `lmsluice bench`
is for:

| source | predicted | measured |
|---|---|---|
| local disk, cold each rep | 0.16× | **0.16×** |
| localhost HTTP | 0.83× | **0.83×** |

Two points, not a validation. The interesting case — a link slow enough that
the coded route wins — has **not been run**, because this machine has no such
link. That half of the table is arithmetic.

## The two corrections

Both changed shipped work. Both are the kind of thing this document exists to
stop happening twice.

**The name was `load` and it was unsound.** Every serious consumer already
uses `load` as a *function* (`torch.load`, `json.load`, `pickle.load`), so
`import load` then `load.load()` reads as nonsense and is shadowed by any
local variable of that name; it is also almost certainly unavailable as a
distribution name, and it named half the job. It is `lmsluice` — the `lm`
prefix puts it in `lmz`'s family, `sluice` is a channel whose gate decides the
flow, which is the design. The rename spared the ordinary verb throughout:
`m.load()`, "the model load path", "load-bearing" and "download" are
untouched.

**The README led with this box, which is `CLAUDE.md` rule 1's failure mode.**
It said the tool's main job on most machines is to say no. That is this
desktop's answer, not the curve's. Recomputed against `gate.py`'s storage axis
with the measured 1.05 GB/s CPU decode and f = 0.671, the crossover is ~1 GB/s
of storage and saturation at 0.70:

> **Retired figures, kept because this is a record of what was believed at the
> time.** 1.05 GB/s was `CODEC_BF16C` at 8 MiB chunks presented as lmz's rate;
> the field-split codec at 1 MiB does **5.95 GB/s** through the transport, which
> moves the crossover from ~1 GB/s of storage to ~5.95. The table below is
> therefore too pessimistic in every row. See "Retired figures" in
> `MEASURED.md`; the corrected table is `docs/strategy.md` §1.

| storage | GB/s | verdict, CPU decoder alone |
|---|---|---|
| NVMe Gen5 | 12.0 | tax, 0.09× |
| NVMe Gen4 (this box) | 6.34 | tax, 0.17× |
| UFS 4.0 (phone) | 4.0 | tax, 0.26× |
| **SATA SSD** | **0.55** | **pays, 1.49× — saturated** |
| **eMMC / SD / USB** | **0.30** | **pays, 1.49× — saturated** |

Same code, same archive, opposite verdicts, decided entirely by the machine in
front of them. A laptop with a SATA SSD gets the whole of lmz's ratio as load
speed with nothing but a CPU.

## Traps, carried forward

**A cache below the process cannot be ruled out, only detected.** Dropping the
guest page cache does not drop the hypervisor's. The same file measured 2.38
GB/s on first touch and 6.38 GB/s an hour later — a 2.7× error, in the
direction that flatters plain files. `Storage.layered` is therefore `True` or
`None` and **never `False`**, and the cold rate is an upper bound wherever it
stayed `None`. This generalises to any VM, container, network filesystem, or
drive with DRAM on it.

**Localhost HTTP is not a network.** It delivered 1.26 GB/s of coded bytes
with the fetch stage saturated — that is Python's HTTP stack, not a link, and
it is *faster* than this box's decoder, so the gate correctly called it a tax.
A real 1 Gb/s link is ten times slower and lands on the other side.

**The encode rate depends on the sample size.** A 128 MiB sample measured 0.53
GB/s at f = 0.787 where the whole 1.87 GB file gave 1.25 GB/s at f = 0.671. A
small sample amortises lmz's startup over less work and finds less redundancy.
`probe --encode` wants a large sample or its answer is a lower bound.

**Overlap is not free when two stages share a resource.** The pipeline reaches
`max(stages)` when they bind on different hardware — disk versus CPU — and
degrades toward `sum(stages)` as they bind on the same one. Decoding into
pinned memory while the DMA engine drains it put the host-decode-to-VRAM route
23% below its predicted rate, because both stages are on the memory bus that
lmz's decoder already saturates. This is the model's one measured failure and
it is recorded rather than patched: fitting a contention term to one machine
would be exactly the mistake rule 1 forbids.

**Decisions inside 1.15× are made on noise.** Decode timings swing 30–40% run
to run on a desktop with other things open, so `plan.MARGIN` refuses to switch
routes below that and plain wins ties. It is the route with no second failure
mode and no CPU spent.

**Do not trust `Accept-Ranges`.** Servers that serve ranges perfectly well omit
it — Python's own `SimpleHTTPRequestHandler` is one — so range support is
settled by asking for one byte and checking for a 206. This was found by a
test failing against a fixture that did support ranges.

## What is open

1. **`k` is unmeasured**, and `gate.py` says pinning it on one low-FLOP/byte
   device is the highest-value measurement this repository can make. This
   package cannot do it: it measures decode where the decoder runs, and the
   decoder does not run on an iGPU yet. Blocked on item 3.
2. **The CUDA kernel does not read v7.** `--shared-tables` writes a format
   worth 3.8× on the GPU decoder and nothing collects it.
   `lmz_gpu_decode_batch_dev` already takes a shared header, so it is wiring.
3. **Metal**, written but never run; then **Vulkan**, which is what reaches
   AMD and Intel iGPUs on Windows and Linux with no toolkit. Order and reasons
   in `lmz/docs/portable-decoder-handover.md`.
4. **The device-decode route is built.** *(updated 2026-08-30: `devdecode.py`
   drives `lmz_gpu_decode_batch_dev` on device pointers and `merge.cu` supplies
   the plane transpose lmz stops short of. Fastest route into VRAM on fp32,
   ~2x plain, byte-identical. Two limits, both measured: it wins on the link
   rather than the decoder — the kernels reach ~2 GB/s not lmz's 111, because
   an lmz stream is 8 lanes wide however large it is — and a BF16 checkpoint is
   refused outright because lmz codes it as `CODEC_BF16C`, which its own GPU
   kernel cannot read. That last one is lmz's to close and is the single
   highest-value thing left. See the device-decode section of `MEASURED.md`.)*
   The original entry follows.

   **The device-decode route is priced and unbuilt.** *(updated 2026-08-30:
   the RAM → VRAM path now exists — `lmsluice/cuda.py`, `Model.to_device`,
   `plan.vram_plan` — and plain and host-decode both land byte-identical in
   VRAM. PCIe is measured at 28.45 GB/s pinned through this tool.)* What is
   still missing is decoding **on** the device: `lmz.gpu` is available here and
   its ABI has the right entry point, but feeding
   `lmz_gpu_decode_batch_dev` means decomposing a chunk into its per-plane rANS
   streams, and lmz exposes decoding a file and reading a byte range, neither
   of which is that. lmz's own handover names the same gap. See the RAM → VRAM
   section of `MEASURED.md`.
5. **One model, one dtype.** Every figure is BF16 from a single checkpoint.
   The ceiling on any win is 1/f, and lmz's own results put FP8 at 17% and
   Q4_K at 5%, where the ceiling is 1.05× and there is almost nothing to win.
6. **lmz should promote a public "decode this chunk into this buffer" entry
   point.** That is the whole of the coupling. `archive.py` probes for
   `lmz.api._chunk_decoder` at open time and falls back to `MappedArchive`,
   which is public and stable, so a change in lmz makes this slower and not
   broken; `Archive.route` says which is live so a benchmark cannot quietly
   measure the fallback.

## How to verify it

    python3 tests/test_lmsluice.py         # 34 tests, builds its own fixtures
    python3 gate.py                        # the device curve, no imports
    ./lmsluice-cli probe <a model or archive>
    ./lmsluice-cli doctor
    ./lmsluice-cli bench <archive> --against <plain> --cold --reps 3

The probe writes to the platform cache directory and refuses a profile whose
fingerprint says it came from another machine, so a home directory that
travels does not carry the wrong numbers with it.

lmz is not installed on this box — it is a checkout at `../lmz`, found by
`lmsluice/_lmz.py`, which handles both that and a normal `pip install lmzip`.

## State when this was written

Nothing is committed. In `lmsluice/`: `README.md`, `MEASURED.md`, `CLAUDE.md`
and `gate.py` modified; `lmsluice/`, `tests/`, `docs/` and `lmsluice-cli`
untracked. In `YFCE/`: `CLAUDE.md` and `docs/decisions.md` updated for the
rename, and one line of `vram/spec/plugin.md` inside that submodule.

**`lmz/` is untouched** — read only, nothing staged, per `YFCE/CLAUDE.md`
rule 4.
