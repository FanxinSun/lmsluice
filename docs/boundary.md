# The lmz / lmsluice boundary

*Where a piece of work goes, and how to settle an argument about it without
asking anyone. If you are about to add a file to either tree, this is the
document to read first.*

## Two charters, one sentence each

**lmz is the compressor/decompressor.** Plain bytes in → coded bytes out; coded
bytes in → byte-identical plain bytes out, as fast as the hardware allows. It
decides what the bytes *are*. It never decides *when* to move them, *how many at
a time*, or *where they land*.

**lmsluice is the AI-model transport facilitator.** It decides *when, from where,
over which link, and into whose memory*. It measures the machine, picks the
route, and runs it with the stages overlapped. It never looks inside a coded
stream.

## The three-question test

For any disputed piece of work, in order:

1. Would it change if the **archive format** changed? → **lmz**.
2. Would it change if the **machine** changed — disk, link, device, core count? → **lmsluice**.
3. **Both?** Then it is two pieces that have not been separated yet. Split it;
   the format half goes to lmz, the machine half stays here.

Question 3 is the diagnostic. Every drift this document exists to correct
answered *both*, and landed wherever it was first needed rather than being cut
in two.

## Ownership

| concern | owner |
|---|---|
| container format, index, manifest, chunk/block layout, page alignment | **lmz** |
| entropy coder, byte-plane split, delta/ref chunks, frequency tables, codec IDs | **lmz** |
| every decoder — C, CUDA, and any future Vulkan or Metal one — **including the plane merge, all the way to plaintext** | **lmz** |
| the decoder's cost constants (`k` per decoded byte, lane count, bytes per symbol) and their provenance | **lmz** |
| tensor → block addressing; which blocks cover a span | **lmz** |
| container parsing (safetensors / GGUF / ONNX) done *to choose chunk boundaries* | **lmz** |
| per-chunk checksums | **lmz** |
| `lmz` CLI, `lmz.fuse`, `lmz.lmzfs`, `Store` | **lmz**, as *consumers* of its own core — never as things the core depends on |
| probing the machine: cold/warm storage rate, link rate, device capability, core scaling | **lmsluice** |
| the rate record, its cache, its machine fingerprint | **lmsluice** |
| the routing arithmetic and the verdict (plain / host-decode / device-decode / expand-first) | **lmsluice** |
| the gate for *unmeasured* devices — the curve, consuming lmz's published constants | **lmsluice** |
| sources: local file, HTTP range, network mount; `pread` concurrency | **lmsluice** |
| fetch/place split, queue depth, inflight bound, thread counts | **lmsluice** |
| CUDA context, device allocation, streams, staging buffers, `__cuda_array_interface__` | **lmsluice** |
| `open_model` / `map` / `load` / `stream` / `to_device`; the tensor index of a **plain** file | **lmsluice** |
| whether to write compressed at all on this machine | **lmsluice** |
| end-to-end verification that a *moved* model is byte-identical | **lmsluice** |
| placement solver, tier budgets, eviction, prefetch, compute overlap | **vram** |

One duplication is deliberate: lmz parses a safetensors header to decide **how to
code**; lmsluice parses one to **index a plain file it will never code**. Both are
correct and neither may grow into the other's job — lmsluice must not parse a
container to make a coding decision, and must take the tensor index of an
*archive* from lmz.

## Five invariants

- **I1 — lmz's decode API is pure: bytes in, bytes out, no I/O, no threads of
  its own.** Any entry point that reads a file has taken a scheduling decision,
  and scheduling is lmsluice's. lmz keeps file-level `decompress()` for
  standalone use, built on top.
- **I2 — a decoder returns plaintext, on every backend.** A decoder that stops
  at plane-major bytes is unfinished, and the missing step lands in the
  transport repo where lmz's tests, portability plan and future backends cannot
  see it.
- **I3 — the archive declares which decoders can read it.** Either every codec
  lmz emits is readable by every decoder lmz ships, or the manifest says
  per-chunk which are. The transport layer never holds a copy of an
  encoder-side threshold.
- **I4 — one rate model and one prober, and they are lmsluice's; one
  coded-stream parser, and it is lmz's.** No machine-measured policy inside lmz
  (no probe, no route choice, no adaptive thread count, no cache-size heuristic
  in the mounts). No coded-stream parsing inside lmsluice (no tables, no codec
  IDs, no kernel that understands the plane layout).
- **I5 — the caller owns the device.** lmz's GPU entry points take a context, a
  stream and device pointers from the caller; lmz never creates a second CUDA
  context, allocates device memory the caller did not ask for, or synchronises a
  stream it does not own.

## The interface

`lmsluice/codec.py` is the boundary in a form a test can check. It is written
against *a* codec, not against lmz, because this package's founding arithmetic
— `speedup = min(1/f, decode/link)` — mentions no codec, and the layer has to be
able to route a zstd or GGUF source on a machine where lmz is not installed.

    encode_options()           name -> Option, what this codec accepts
    encode(src, dst, **opts)   write an archive; options must be declared ones
    ratio()                    coded bytes / plain bytes
    units() / units_for(spans) units of coded work: (off, clen, dst, rlen)
    cover(spans)               the plain range the covering blocks actually fill
    decode_chunk(unit, payload, out, off)   pure: bytes in, bytes out    -> I1
    tensor_index()             name -> plain span, from the container's metadata
    members() / span_of()      what is inside, and where
    capabilities()             which decoders can read this archive       -> I3
    cost_model()               k, lanes, bytes/symbol, each with provenance
    device_decode(...)         device pointers + caller's stream, plaintext out -> I2, I5

`lmsluice/lmzcodec.py` is the lmz implementation and **the only module that may
import lmz**. `lmsluice/zstdcodec.py` is a second implementation over the
standard library's zstd — it owns its own small container and no other module
parses it. Two implementations is what stops this protocol being theoretical,
and the second one is the reason the package works with nothing installed. Anything this package does that is not one of
the operations above is a **loan** and is registered below.

Note what is *not* here: `fetch`. Reading bytes is a scheduling decision and
belongs to `archive.py` on this side, which is why lmz's public `decompress()`
and `MappedArchive.read()` are both the wrong shape despite doing the right
arithmetic — each takes the I/O decision away.

## The loan register

Three pieces of lmz's work are held here because lmz does not yet offer the
interface that would let them go home. **Do not remove them**; they are
load-bearing until the replacement ships. Each is marked in its own file with
the same three facts.

| loan | whose | why it is here | retired by |
|---|---|---|---|
| `lmsluice/merge.cu`, its dispatch in `devdecode.py`, and `Unit.opaque` | lmz | `lmz_gpu_decode_batch_dev` returns plane-major bytes, not plaintext; the plane split is part of the format, so undoing it is part of decode | a device decode entry point that returns plaintext (**I2**) |
| `lmsluice.gpu_chunk_size()`, and `Capabilities.advice` with it | lmz | the *remedy* for a non-device-readable archive: which encoder setting to change. lmz now declares the **problem** (I3, below) but not the fix | an encoder-side statement of which settings produce device-readable output |

### Retired, 2026-08-30

Two loans went home the day lmz shipped the interface they were waiting for
(`b426cc9`), and are recorded here rather than deleted, because a register that
only ever grows teaches nothing.

| loan | retired by | how it shows |
|---|---|---|
| the reach for `lmz.api._chunk_decoder` / `freqs.TableSet` | `ArchiveIndex.chunks()` + `.decode(ref, payload, out=)` — *"No I/O, no threads"*, which is **I1** verbatim | `Archive.route` now reports `public`; `direct` and `mapped` remain as fallbacks for a released lmz |
| deriving `Capabilities` from an encoder threshold | `lmz.capabilities(path)`, which declares `batch_decodable`, the readable byte fraction and the blockers by name, with its own `derived: false` — **I3** | `Capabilities.derived` now carries lmz's answer instead of a standing confession |

**And the expiry test for the first one did not fire.** It watched for
module-level `decode_chunk` / `decode_block` / `decode_unit`; lmz shipped a
*method*, `ArchiveIndex.decode`. So the replacement arrived and the register
went on claiming a live loan. The lesson, which generalises past this instance:
**an expiry test that guesses the shape of a future API only fires if the guess
was right — watch for the capability, not the spelling.** Both tests are now
inverted to guard the retirement instead: they assert the adapter is *on* the
public path, and fail if it silently falls back.

A fourth was retired by this work: `gate.py` used to hold `k` as a literal and
now takes a `CodecCost` from `cost_model()`, so the constant has an owner and a
provenance. It still refuses to pick between the two readings of `k`, which is
the whole point of it.

Two fields exist only to serve loans and expire with them: **`Unit.opaque`**
with the merge loan (I2) — it is the hole through which `devdecode.py` reads
codec IDs and plane flags, and it must come out when the kernel does — and
**`Capabilities.advice`** with the declaration loan (I3), since a format that
declares its own capabilities makes advice about encoder flags unnecessary.
Neither is general interface.

`tests/test_lmsluice.py` contains, for each loan, a test that **fails when lmz
ships the replacement** — so the register cannot quietly outlive its reasons.
The merge loan's test also asserts that nothing outside `devdecode.py` has
started reaching through `opaque`, so the two cannot drift apart.

## Adding something new

Run the three questions. If the answer is "both", split it before you write it;
if you cannot, write it here, register it as a loan with a retirement condition,
and say so in its header. A loan with a named retirement is a debt. An unmarked
one is a drift, and four of those is what this document cost.
