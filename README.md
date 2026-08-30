# lmsluice — the model load path, on whatever silicon the box has

A runtime that moves model weights from wherever they are to wherever they are
needed, over the route that is actually faster on the machine it is running
on. It decides by measurement, not by assumption, and it says what it decided
and on what numbers. Pure software, pure standard library, and nothing about
the bytes changes: every weight arrives byte-identical.

The target is **transport**, not training and not inference. `vram/` owns
training residency and has ruled out inference on its own roofline. This is
the third case, and the only one that happens on every machine, every time a
model is opened.

```
lmsluice probe  ./model.lmz          # measure this machine, once
lmsluice plan   ./model.lmz          # which route is faster here, and why
lmsluice bench  ./model.lmz --against ./model.safetensors   # check that answer
lmsluice get    https://host/m.lmz ./model.safetensors      # move it
```

```python
import lmsluice

m = lmsluice.open_model("model.lmz")   # or .safetensors, or an https URL
print(m.plan.explain())                # the route, the rates, the arithmetic
weights = m.load()                     # one buffer, filled in parallel

for name, view in m.stream(budget=1 << 30):   # bounded memory, overlapped I/O
    upload(name, view)
```

## The one ratio

Compression is free on a path when the decoder is faster than the link it is
feeding from. Everything here follows from that. For N plain bytes and an
archive whose coded size is r times its plain size, with the stages overlapped:

    plain    N / link
    coded    max(rN / link, N / decode)
    speedup  min(1/r, decode / link)

The gate is the second term and **it does not mention r at all**. Compression
pays exactly when the codec beats the link; the ratio then decides how much,
up to a ceiling of 1/r. Writing is the same statement with `encode` in place
of `decode`, which is why the answer is often the opposite on the way out.

Three consequences, each of which is easy to get wrong:

- **A fast disk is a reason not to compress.** Above the gate the coded route
  loses, and by more the faster the disk is. A loader that always takes the
  compressed path is choosing to be slower on the machines that would
  otherwise be fastest.
- **The ratio cannot rescue a slow codec.** `min` is a floor, not a sum.
  Saving 40% of the bytes buys nothing where the decoder is already the
  bottleneck.
- **Expanding first is a third route, and it is far worse than both.** Without
  a tool like this, a user with a compressed checkpoint runs `decompress` and
  then opens the result — paying the link twice and a write in between. It is
  priced here so the comparison people actually face is visible.

## Where the verdict flips

The gate is a curve, and both sides of it are machines people own. With a
decoder running at D and an archive at ratio f, storage slower than D pays and
storage faster than D is taxed — and below D·f the win saturates at 1/f,
because the ratio is all there is left to win.

Taking this box's **measured** CPU decode of 1.05 GB/s and f = 0.671, the
crossover sits at about **1 GB/s of storage**, saturating at 0.70:

| storage | GB/s | verdict with a 1.05 GB/s CPU decoder |
|---|---|---|
| NVMe Gen5 | 12.0 | tax, 0.09× |
| NVMe Gen4 (this box) | 6.34 | tax, 0.17× |
| UFS 4.0 (phone) | 4.0 | tax, 0.26× |
| **SATA SSD** | **0.55** | **pays, 1.49× — saturated** |
| **eMMC / SD / USB** | **0.30** | **pays, 1.49× — saturated** |

So the same code and the same archive reach opposite verdicts, decided
entirely by the machine in front of them. A laptop with a SATA SSD gets the
whole of lmz's ratio as load speed with nothing but the CPU; the desktop this
was written on pays six times over for it. Neither is *the* answer, which is
the reason the decision is made at run time and not written down here.

Move the decoder instead of the disk and it flips again:

| decoder | GB/s | against 6.34 GB/s NVMe |
|---|---|---|
| lmz CPU, this box, measured | 1.05 | 0.17× |
| lmz CUDA, per-chunk table, RTX 5080 | 111 | free; 1.49× by the ratio |
| lmz CUDA, shared table, RTX 5080 | 418 | free; 1.49× by the ratio |
| a 2-CU display iGPU, **predicted** | 4 – 36 | pays at the floor of the interval |

That last row is `../gate.py`'s job rather than this package's, and the
interval is the honest state of it: the decoder's compute cost per byte is
known from **one** point on **one** card, and that point fits a
compute-bound and a bandwidth-bound reading equally well. The two readings
agree on the RTX 5080 that produced them and differ by 8× on a 2-CU
integrated GPU — which is the class this project exists for. `gate.py` carries
both and refuses to pick; this package measures instead of predicting, and the
two are asserted to state the same identity in the test suite.

**What survives the pessimistic reading** is the founding claim: even at the
floor of the interval, the smallest integrated GPU AMD ships decodes about 2×
faster than the CPU decoder, on a device that is a display adapter rather than
an APU. What does not survive is reading any one desktop's NVMe as *the* disk.

## What it is

**A router.** `lmsluice plan` reads the machine's measured profile and prices
every route. `--link` and `--decode` answer the question a laptop cannot
measure — *would this pay over the link I will deploy on?* — with the asker's
numbers, labelled as theirs.

**A transport.** Two stages, sized independently, overlapped across a bounded
queue. Fetching wants queue depth; decoding wants the opposite, because lmz's
own measurement is that past two threads the interpreter between native calls
turns into contention. `lmz.decompress` reads and decodes inside one worker,
so one setting has to serve both. Splitting them is what this adds, and
`Report.limited_by` names which stage the run waited on.

**Three shapes of access**, because consumers differ. `map()` is an mmap and
moves nothing until touched — the fastest partial access there is, and it has
no coded form, which the tool says rather than faking. `load()` reads for real
into one buffer. `stream()` does the same under a memory ceiling, yielding
tensors as they land, for the machines where the model does not fit twice.

**Any source.** A local file and an HTTPS URL are two implementations of
`pread`, so the pipeline above them is unchanged, and an lmz archive can be
opened and decoded over range requests without landing on disk first. Servers
that refuse ranges are fetched once instead of refused. Connections are kept
alive per thread, because a handshake per megabyte costs more than the
megabyte.

**A probe that measures the right things.** Cold rather than warm, per storage
device rather than per machine, and against the archive that will actually be
read rather than a synthetic one — decode rate moves with chunk size and with
which codecs the encoder chose, so it is a property of the file, not the
build. Where the platform cannot drop a cache, the number is reported as warm
rather than passed off as cold.

## What this is not

**Running the model on the iGPU.** `vram/`'s roofline settles it: inference
decode does one multiply-add per byte of BF16 against about 4,132 needed,
*"short by a factor of four thousand, and no scheduling recovers it."* The
integrated GPU should decode the archive, not run the network.

**A second codec.** lmz produces bytes and turns them back into bytes; it does
not decide when. That line does not move because the device changed.

**A benchmark that flatters itself.** `lmsluice bench` runs both routes for real
and prints the predicted speedup beside the measured one. Twice so far the
plan has been checked this way, on a local disk and over HTTP, and it was
right to the second digit both times — see `MEASURED.md`. When it is wrong,
one command says so.

## Where the boundary is

| | whose |
|---|---|
| the coded stream, its tables, its blocks, its index | **lmz** |
| a standalone GPU decoder with a stable ABI — CUDA, Metal, Vulkan | **lmz** |
| tensor → block addressing | **lmz** |
| which device and which route on this machine, and the probe that decides | **lmsluice** |
| dispatch, queue depth, staging buffers, overlap with I/O | **lmsluice** |
| sources — local, HTTP, and whatever comes next | **lmsluice** |
| adapters — safetensors-compatible open, PyTorch, llama.cpp | **lmsluice** |
| training residency across VRAM / RAM / SSD | `vram/` |

The split against lmz is the one `lmz/docs/gpu-residency-handover.md` already
draws, applied to a different consumer. Every use of lmz's internals is in
`lmsluice/archive.py` and nowhere else; two of them are internals rather than
public API — decoding one chunk, and a v7 archive's shared table set — because
lmz exposes decompressing a *file* and reading a *byte range*, and neither is
the shape a pipeline wants. Both are probed at open time and fall back to
`lmz.MappedArchive`, which is public and stable, so a change in lmz makes this
slower and not broken. `Archive.route` says which one is live, so a benchmark
can never quietly measure the fallback.

**The one thing lmz should promote:** a public "decode this chunk into this
buffer" entry point. It is the whole of the coupling.

## Status

Runs. 33 tests, no dependencies beyond the standard library and lmz.

```
python3 tests/test_lmsluice.py       # builds its own fixtures; skips what needs lmz
./lmsluice-cli doctor                # what is measured and what is active
```

What exists:

- `lmsluice/transport.py` — the two-stage overlapped pipeline, bounded by a queue
- `lmsluice/plan.py` — the routing arithmetic, and the margin that stops a
  decision being made on noise
- `lmsluice/probe.py` — cold reads, sustained writes, and decode measured on the
  real archive with the thread count swept rather than assumed
- `lmsluice/source.py` — local files and HTTP range requests behind one `pread`
- `lmsluice/archive.py` — the lmz coupling, in one file
- `lmsluice/model.py` — `map` / `load` / `stream`, one API over either route
- `lmsluice/cli.py` — `probe`, `plan`, `info`, `bench`, `get`, `write`, `doctor`
- `../gate.py` — the device-side model: decode rate bounded from two sides,
  the unmeasured constant carried as an interval, no imports so it runs on a
  machine that has nothing installed
- `probe/igpu-bench.ps1` — the Direct3D 11 compute benchmark that feeds it,
  measuring whether a given device *could* decode

[`docs/transport-handover.md`](docs/transport-handover.md) is the full record
of how this was built and measured — the design decisions that were not
obvious, the two corrections that changed it after it was written, the traps,
and what each open item is blocked on.

What is blocked on lmz, in order, unchanged from
`lmz/docs/portable-decoder-handover.md`:

1. **The CUDA kernel reading v7.** `--shared-tables` writes a format worth
   3.8× on the GPU decoder and nothing collects it yet.
2. **The Metal port**, written but never run — `lmz/scratchpad/gpu/metal/`.
3. **Vulkan**, which is what opens this up: AMD and Intel iGPUs, Windows and
   Linux, driver-level with no toolkit to find.

When any of those lands, it enters here as another codec rate in the profile
and another row in the plan. Nothing above `plan.py` changes, which is the
point of putting the decision in one place.
