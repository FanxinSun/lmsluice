"""First run of lmsluice on macOS, and the cold measurement this project cannot take.

Two things this settles that the development box cannot, and they are related.

**Is a cold read obtainable?** On the WSL2 box it is not: `posix_fadvise` empties
the Linux page cache -- `mincore` confirms 0% residency -- and the read still
returns at 5.3 GB/s, because the Windows host caches the virtual disk beneath
and the guest can neither address nor evict it. Coldness there turned out to be
host state that changed between one hour and the next with nothing done from
inside. On bare-metal APFS there is no layer underneath: `F_NOCACHE` is honoured
by the kernel that owns the disk, so a cold number is a cold number.

**Does the macOS path work at all?** `README.md` says it is written and never
run. `source.py` takes the `F_NOCACHE` branch here for the first time, `_win`
declines, and `buffer.py` falls back to `bytearray` because there is no
`MADV_HUGEPAGE`. Reviewing it before this ran found two bugs that only appear on
this platform -- the "warm" rate was measured through a still-open `F_NOCACHE`
descriptor, and the mode was never cleared afterwards -- so treat a clean run as
evidence and not as a formality.

## What it does

Three stages, each useful on its own, so a failure late still leaves an answer.

    stage 1  platform, import, and whether F_NOCACHE actually bypasses the cache
    stage 2  the cold rate of this machine's storage, fresh process per read
    stage 3  plain against coded, cold, interleaved -- the row `strategy.md` §1
             has had to leave unsettled

Stage 3 needs a zstd: Python 3.14 has `compression.zstd` in the standard
library, and below that it needs `pip install zstandard` (~5 MB) in a virtual
environment of its own. Stages 1 and 2 need nothing beyond the package.

## What it writes

One temporary directory, reported before it is created and removed at the end
unless `--keep` is passed. The default fixture is 1 GiB and every copy of it is
counted in what the script prints, because on a laptop that is worth knowing
before rather than after.

    python3 experiments/macos_cold.py                 # 1 GiB, n=5
    python3 experiments/macos_cold.py --size-mb 256   # quicker
    python3 experiments/macos_cold.py --stage 1       # just the platform check
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# -- the fixture -----------------------------------------------------------
#
# Real BF16 weights compress because their exponent byte carries little
# entropy: the values cluster near zero, so a few exponents cover almost
# everything. Uniform random bytes have none of that and would give a ratio of
# 1.0, which would make stage 3 measure nothing. So the values are drawn from a
# normal distribution, which is what a trained weight matrix approximately is
# and what the fixture on the development box also is (`generator: lmz-tests`).

def _bf16_normal(count: int, seed: int) -> bytes:
    """BF16 with the byte structure that makes these codecs work.

    A bfloat16 is [sign 1][exponent 8][mantissa 7], so the high byte carries
    the sign and most of the exponent and the low byte is nearly all mantissa.
    In trained weights the values cluster near zero, which means the high byte
    takes few distinct values while the low byte is close to uniform — and that
    asymmetry is the whole reason a byte-splitting codec beats a general one.

    Built directly rather than by drawing floats: a Python loop over hundreds
    of millions of elements would dominate the run, and sampling the two bytes
    from their own distributions reproduces the property being measured. Each
    tensor gets its own seed so the file is not one block repeated, which would
    compress like nothing real.
    """
    import random

    rng = random.Random(seed)
    # Exponent/sign byte: a handful of values, as a near-zero normal gives.
    high_alphabet = [0x3B, 0x3C, 0x3D, 0x3E, 0xBB, 0xBC, 0xBD, 0xBE]
    weights = [6, 14, 22, 8, 6, 14, 22, 8]
    high = bytes(rng.choices(high_alphabet, weights=weights, k=count))
    low = bytes(rng.getrandbits(8) for _ in range(min(count, 1 << 16)))
    low = (low * (count // len(low) + 1))[:count]      # cheap, still high entropy
    out = bytearray(count * 2)
    out[0::2] = low                                     # little-endian: low first
    out[1::2] = high
    return bytes(out)


PURGE = False


def _maybe_purge() -> None:
    """Drop the whole buffer cache, if asked and if the platform has a way.

    macOS `purge` is the only reliable eviction it offers: there is no
    per-file equivalent of `posix_fadvise(DONTNEED)`, and `F_NOCACHE` governs
    future access rather than what is already resident. It is coarse — it
    drops everything, not this file — and it needs a password, which is why it
    is opt-in rather than the default.
    """
    if not PURGE:
        return
    try:
        subprocess.run(["sudo", "purge"], check=False, timeout=120)
    except Exception:                     # noqa: BLE001 -- never fatal
        pass


def _no_cache(fd: int) -> bool:
    """Ask macOS not to cache this descriptor's data. No-op elsewhere.

    Used on the descriptor the fixture is *written* through, which is the
    step that matters. `F_NOCACHE` declines to populate the cache; it does not
    evict what is already there. A file that has just been written is in the
    unified buffer cache because of the write, and no amount of read-side
    `F_NOCACHE` removes it — which is why the first Mac run measured 11.8 GB/s
    on a "bypassed" read of a 5-7 GB/s SSD and correctly refused to call it
    cold. Setting it on the writer keeps the data out from the start.
    """
    if hasattr(os, "posix_fadvise"):          # Linux: nothing to do here
        return False
    try:
        import fcntl

        fcntl.fcntl(fd, 48, 1)                # F_NOCACHE
        return True
    except (ImportError, OSError, ValueError):
        return False


def write_fixture(path: str, size_bytes: int, seed: int = 7) -> str:
    """A safetensors file of BF16 tensors with realistic exponent structure."""
    per = 1 << 22                      # elements per tensor, 8 MiB each
    n = max(1, size_bytes // (per * 2))
    names = [f"model.layers.{i}.weight" for i in range(n)]
    header, off, span = {}, 0, per * 2
    for name in names:
        header[name] = {"dtype": "BF16", "shape": [per],
                        "data_offsets": [off, off + span]}
        off += span
    # Labelled, because a synthetic fixture is not a checkpoint: its mantissa
    # byte is uniform where a real one has some structure, so it compresses a
    # little worse than real weights (f nearer 0.76 than the 0.673 measured on
    # a real Llama checkpoint). That makes the ceiling tighter, not looser, so
    # any win it shows is a lower bound on the real one.
    header["__metadata__"] = {"format": "pt", "generator": "lmsluice-macos-check"}
    blob = json.dumps(header).encode()
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        _no_cache(fh.fileno())
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for i in range(len(names)):
            fh.write(_bf16_normal(per, seed + i))
    return path


# -- stages ----------------------------------------------------------------

def stage1() -> dict:
    """Platform, import, and whether F_NOCACHE is honoured.

    The check that matters is the last one, and it is done by measurement
    rather than by trusting the return code: set the mode, read, clear it,
    prime, read again. If the second is much faster the first genuinely missed
    the cache.
    """
    from lmsluice import buffer as B
    from lmsluice.source import FileSource
    import lmsluice

    print("== stage 1: platform and the macOS path ==")
    print(f"  {platform.platform()}")
    print(f"  python {sys.version.split()[0]} on {platform.machine()}")
    print(f"  lmsluice {lmsluice.__version__} from {os.path.dirname(lmsluice.__file__)}")
    print(f"  os.posix_fadvise present: {hasattr(os, 'posix_fadvise')}  "
          f"(expected False on macOS)")
    how, why, floor = B.strategy()
    print(f"  destination buffer: {how} — {why}")

    d = tempfile.mkdtemp(prefix="lmsluice-stage1-")
    try:
        p = write_fixture(os.path.join(d, "probe.safetensors"), 64 << 20)
        src = FileSource(p)
        n = os.path.getsize(p)
        try:
            ok, how_un = src.uncache()
            print(f"  uncache(): {ok} — {how_un}")

            # Read into a buffer that already exists, in chunks, rather than
            # through `pread`. `pread` allocates a fresh object per call, and on
            # fast storage that allocation — not the disk — becomes the limit on
            # BOTH sides, which squashes the very ratio this is trying to read.
            # An Apple SSD at 5 GB/s and a cached read at 7 differ by 1.3x that
            # way and by much more when the allocation is taken out.
            chunk = 8 << 20
            buf = bytearray(chunk)
            fd = src._fd                    # the descriptor uncache() acted on

            def sweep() -> float:
                off, started = 0, time.perf_counter()
                while off < n:
                    got = os.preadv(fd, [memoryview(buf)[:min(chunk, n - off)]], off)
                    if not got:
                        break
                    off += got
                return off / (time.perf_counter() - started)

            _maybe_purge()
            cold = sweep()
            ok2, how_re = src.recache()
            print(f"  recache(): {ok2} — {how_re}")
            sweep()                                      # prime, untimed
            warm = sweep()
        finally:
            src.close()
        print(f"  bypassed read {cold/1e9:.2f} GB/s   cached read {warm/1e9:.2f} GB/s"
              f"   ratio {warm/cold:.1f}x")
        # With the allocation gone the two sides are limited by what they
        # actually read from, so a real bypass shows a wide gap. 1.5x is kept as
        # the bar; what changed is that the measurement can now reach it.
        honoured = warm > cold * 1.5
        # Name the mechanism this platform actually used, so the line is not
        # quietly wrong on the box it was developed on.
        mech = "F_NOCACHE" if not hasattr(os, "posix_fadvise") else "DONTNEED"
        print(f"  {mech} really does get past the cache: {honoured}"
              + ("" if honoured else
                 "  <-- so-called cold numbers here would NOT be cold"))
        return {"honoured": honoured, "cold": cold, "warm": warm}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _one(path: str, mode: str) -> dict:
    """One load in a fresh process. `mode` is 'supplied' or 'inside'."""
    prog = (
        "import json,os,sys,time\n"
        f"sys.path.insert(0,{HERE!r})\n"
        "from lmsluice.buffer import destination\n"
        "from lmsluice.model import open_model\n"
        "from lmsluice.source import FileSource\n"
        f"p={path!r}\n"
        "m=open_model(p)\n"
        "s=m._src if isinstance(getattr(m,'_src',None),FileSource) else "
        "getattr(m._arc,'source',None)\n"
        "ok,how=(s.uncache() if s is not None else (False,'no source'))\n"
        f"mode={mode!r}\n"
        "if mode=='supplied':\n"
        "    b=destination(m.plain_bytes)\n"
        "    mv=memoryview(b).cast('B')\n"
        "    for i in range(0,m.plain_bytes,4096): mv[i]=0\n"
        "    t=time.perf_counter(); m.load(into=b); el=time.perf_counter()-t\n"
        "else:\n"
        "    t=time.perf_counter(); m.load(into=bytearray(m.plain_bytes)); "
        "el=time.perf_counter()-t\n"
        "print(json.dumps({'route':m.route,'seconds':el,"
        "'gbps':m.plain_bytes/el/1e9,'cold_ok':ok,'how':how}))\n"
        "m.close()\n"
    )
    _maybe_purge()
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    if r.returncode:
        return {"error": r.stderr.strip()[-300:]}
    return json.loads(r.stdout)


def stage2(paths: list, reps: int) -> None:
    """One never-before-read copy per reading.

    An earlier version read `copies[0]` every time, which made every reading
    after the first a warm one — and worse, left that copy cached for stage 3,
    where it showed up as a plain route running at twice the machine's actual
    cold rate. A stage that contaminates a later stage is the kind of bug that
    looks like a result.
    """
    print("\n== stage 2: this machine's cold storage rate ==")
    print("  fresh process per read, one unread copy each, F_NOCACHE before each")
    rates = []
    for i in range(min(reps, len(paths))):
        got = _one(paths[i], "supplied")
        if "error" in got:
            print(f"  {i}: FAILED {got['error']}"); continue
        rates.append(got["gbps"])
        print(f"  {i}: {got['seconds']:6.3f} s  {got['gbps']:5.2f} GB/s"
              f"  ({got['how']})")
    if rates:
        rates.sort()
        print(f"  median {rates[len(rates)//2]:.2f} GB/s   "
              f"spread {rates[0]:.2f}-{rates[-1]:.2f}")


def stage3(plain: str, coded: str, reps: int, f_ratio: float) -> None:
    print("\n== stage 3: plain against coded, cold, interleaved ==")
    print(f"  ratio f = {f_ratio:.3f}, so the speedup cannot exceed "
          f"{1/f_ratio:.2f}x")
    ps, cs = [], []
    for i in range(reps):
        p = _one(plain, "supplied")
        c = _one(coded, "supplied")
        if "error" in p or "error" in c:
            print(f"  {i}: FAILED {p.get('error') or c.get('error')}"); continue
        ps.append(p["gbps"]); cs.append(c["gbps"])
        print(f"  {i}: plain {p['gbps']:5.2f}  coded {c['gbps']:5.2f}  "
              f"ratio {c['gbps']/p['gbps']:.2f}x")
    if ps:
        ps.sort(); cs.sort()
        mp, mc = ps[len(ps)//2], cs[len(cs)//2]
        print(f"\n  median plain {mp:.2f} GB/s   coded {mc:.2f} GB/s   "
              f"ratio {mc/mp:.2f}x")
        if mc / mp > 1 / f_ratio + 1e-9:
            print("  ABOVE THE RATIO CEILING — the comparison is unfair "
                  "somewhere, do not publish it")
        else:
            print("  within the ratio ceiling")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size-mb", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stage", type=int, default=0, help="run only this stage")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--purge", action="store_true",
                    help="run `sudo purge` before each timed read (macOS). "
                         "Drops the whole unified buffer cache, which is the "
                         "only reliable eviction macOS offers; asks for your "
                         "password")
    ap.add_argument("--force", action="store_true",
                    help="run stages 2 and 3 even if the cache "
                         "check fails; results are then warm and "
                         "must be reported as warm")
    a = ap.parse_args()
    global PURGE
    PURGE = a.purge
    if PURGE:
        print("--purge: `sudo purge` runs before every timed read. It drops "
              "the whole buffer cache, not just this file.\n")

    if sys.platform != "darwin":
        print(f"note: this is {sys.platform}, not darwin. It will run, but the "
              f"point of it is the macOS path.\n")

    got = stage1()
    if a.stage == 1:
        return 0
    if not got["honoured"]:
        mech = "F_NOCACHE" if not hasattr(os, "posix_fadvise") else "DONTNEED"
        print(f"\n{mech} did not get past the cache on this filesystem, so "
              f"stages 2 and 3 would report warm numbers as cold.")
        print("That is itself a result and worth reporting: it is what the "
              "development box does, and the reason this script exists.")
        if sys.platform == "darwin":
            print("On macOS this is expected when the file is already cached: "
                  "F_NOCACHE declines to populate the cache but does not evict "
                  "what a write already put there.")
            print("Try --purge, which runs `sudo purge` and drops the whole "
                  "buffer cache. It is the only reliable eviction macOS offers "
                  "and it will ask for your password.")
        if not a.force:
            print("Or pass --force to run them anyway; whatever they print is "
                  "then warm and must be reported as warm.")
            return 1
        print("--force given: continuing, and every number below is WARM.")

    size = a.size_mb << 20
    need = size * (2 + a.reps * 2 + 2) / 1e9   # +2 spares for stage 2
    d = tempfile.mkdtemp(prefix="lmsluice-cold-")
    print(f"\n  workspace {d}")
    print(f"  will write about {need:.1f} GB and remove it at the end"
          + (" (--keep given, will not remove)" if a.keep else ""))
    try:
        plain = write_fixture(os.path.join(d, "model.safetensors"), size)
        # Which codec is used decides what stage 3 can possibly show, so it is
        # chosen deliberately and named in the output. The stdlib codec decodes
        # at about 0.77 GB/s; any SSD worth measuring is several times that, so
        # with it the answer is "tax" on every fast machine and the stage
        # settles nothing. lmz's field-split codec does ~5.95 GB/s through this
        # transport, which is the same order as an SSD, and only then is the
        # comparison a contest rather than a foregone conclusion.
        f_ratio, coded, codec_name = 1.0, None, None
        try:
            from lmsluice.cache import _encoder

            codec_name, enc = _encoder("auto")
            ext = ".lmz" if codec_name == "lmz" else ".lmsl"
            coded = os.path.join(d, "model" + ext)
            enc.encode(plain, coded)
            f_ratio = os.path.getsize(coded) / os.path.getsize(plain)
            print(f"  codec: {codec_name}")
            print(f"  coded {os.path.getsize(coded)/1e9:.2f} GB of "
                  f"{os.path.getsize(plain)/1e9:.2f} GB, f = {f_ratio:.3f}")
            if codec_name != "lmz":
                print("  NOTE: this is the stdlib codec at ~0.77 GB/s. On a fast")
                print("  SSD it loses by arithmetic and stage 3 will say tax")
                print("  whatever the storage does. For a real contest install")
                print("  lmz:  pip install 'lmzip[zstd]'   (about 6 MB)")
        except Exception as exc:                  # noqa: BLE001
            print(f"  no codec available ({exc}); stage 3 will be skipped. "
                  f"Python 3.14 has compression.zstd in the standard library; "
                  f"below that, `pip install zstandard` in a venv of its own.")

        # Distinct copies so no read is ever a re-read, which is belt and braces
        # here -- F_NOCACHE should make that unnecessary -- and free insurance.
        def copy_uncached(src: str, dst: str) -> None:
            """Copy without either side entering the cache, where that is
            possible. `shutil.copyfile` caches both."""
            with open(src, "rb") as i, open(dst, "wb") as o:
                _no_cache(i.fileno())
                _no_cache(o.fileno())
                shutil.copyfileobj(i, o, 8 << 20)

        copies = []
        for i in range(a.reps):
            pc = os.path.join(d, f"p{i}.safetensors")
            copy_uncached(plain, pc)
            cc = None
            if coded:
                cc = os.path.join(d, f"c{i}" + os.path.splitext(coded)[1])
                copy_uncached(coded, cc)
            copies.append((pc, cc))

        # Stage 2 gets its own copies, never the ones stage 3 will read.
        if a.stage in (0, 2):
            spare = []
            for i in range(min(2, a.reps)):
                sp = os.path.join(d, f"s{i}.safetensors")
                copy_uncached(plain, sp)
                spare.append(sp)
            stage2(spare, len(spare))
        if coded and a.stage in (0, 3):
            print("\n== stage 3 uses one unread copy per measurement ==")
            ps, cs = [], []
            print(f"  ratio f = {f_ratio:.3f}, ceiling {1/f_ratio:.2f}x")
            for i, (pc, cc) in enumerate(copies):
                p = _one(pc, "supplied"); c = _one(cc, "supplied")
                if "error" in p or "error" in c:
                    print(f"  {i}: FAILED {p.get('error') or c.get('error')}")
                    continue
                ps.append(p["gbps"]); cs.append(c["gbps"])
                print(f"  {i}: plain {p['gbps']:5.2f}  coded {c['gbps']:5.2f}  "
                      f"ratio {c['gbps']/p['gbps']:.2f}x")
            if ps:
                ps.sort(); cs.sort()
                mp, mc = ps[len(ps)//2], cs[len(cs)//2]
                print(f"\n  median plain {mp:.2f}  coded {mc:.2f}  "
                      f"ratio {mc/mp:.2f}x   ceiling {1/f_ratio:.2f}x")
                print("  " + ("ABOVE THE CEILING — unfair somewhere, do not publish"
                              if mc/mp > 1/f_ratio + 1e-9 else
                              "within the ceiling"))
    finally:
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)
            print(f"\n  removed {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
