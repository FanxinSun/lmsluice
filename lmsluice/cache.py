"""Compress on first fetch, and only where it will pay.

The supply-side problem in one sentence: compression on the load path is worth
something only if the model is already compressed, and no model is. Asking
people to convert their checkpoints before they have seen a benefit is a
chicken-and-egg that every format loses.

This is the way out. The first load reads the plain file at its normal speed and
nothing is asked of anybody. Afterwards -- off the critical path -- the model is
compressed into a local cache, and every later load takes the coded route. The
supply side becomes *this machine*, there is nothing to publish, nothing to
trust in advance, and with `zstdcodec` there is not even a dependency to
install.

**It is a decision, not a policy, and that is the whole point.** A cache that
always stores the compressed form is exactly the "loader that always takes the
compressed path" this package criticises on its own front page: on a fast NVMe
the correct cache entry is the *plain file*, and building a coded one would make
every later load slower. So the same gate that routes a read decides whether the
cache is worth building at all, on the machine in front of it, and says which it
chose and why.

Three deliberate refusals, each because the alternative is a tool that surprises
people:

- **Nothing is cached unless asked.** Writing gigabytes as a side effect of
  opening a file is not a favour. An existing cache is used automatically; a new
  one is built only on request.
- **Nothing is deleted.** The compressed copy sits beside the original and costs
  disk until someone removes it. A tool that tidies a model directory on the
  user's behalf will eventually delete the wrong thing.
- **A cache entry is bound to the file that produced it** -- path, size and
  modification time -- so an edited checkpoint silently misses rather than
  silently serving stale weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass

from . import plan as _plan
from .rates import Profile, cache_dir as _root, storage_key

# How much of the file to sample when deciding whether caching pays. Enough for
# the compressor to see real structure, small enough that the decision is free
# next to the load it is attached to.
SAMPLE = 64 << 20

# Leave this much of the filesystem alone. A cache that fills a disk is worse
# than no cache.
KEEP_FREE = 2 << 30


def models_dir() -> str:
    return os.path.join(_root(), "models")


def key_for(path: str) -> str:
    """Identity of the *contents*, cheaply.

    Path, size and mtime rather than a hash of the bytes: hashing a 70 GB
    checkpoint to decide whether to read it faster is self-defeating, and the
    failure mode of this key -- a file rewritten within the same second at the
    same length -- is one a model directory does not produce.
    """
    st = os.stat(path)
    seed = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def entry_for(path: str) -> str:
    return os.path.join(models_dir(), f"{key_for(path)}.lmsl")


def find(path: str) -> str | None:
    """A cached coded form of `path`, or None. Never builds one."""
    try:
        got = entry_for(path)
    except OSError:
        return None
    return got if os.path.exists(got) else None


@dataclass(frozen=True)
class Verdict:
    """Whether caching pays here, and the numbers behind it."""

    worth_it: bool
    why: str
    source_rate: float | None = None
    decode_rate: float | None = None
    ratio: float | None = None
    speedup: float | None = None

    def explain(self) -> str:
        head = ("cache: worth building" if self.worth_it
                else "cache: not worth building")
        return f"  {head} — {self.why}"


def measure(path: str, sample: int = SAMPLE):
    """(ratio, encode B/s, decode B/s) for this file's own bytes.

    Sampled from the middle rather than the head: a safetensors file starts
    with a JSON header that compresses unlike anything after it, and judging a
    checkpoint by its header is how you get a ratio that is not the file's.

    **Decode is measured on one thread, and that is not a shortcut.** Measured
    on this box, 16 blocks of 4 MiB: one thread 2.02 GB/s, two 1.51, four 1.78,
    eight 1.79, sixteen 1.69. The standard library's zstd does not scale with
    threads and gets slower for trying, so the loader gains nothing from a pool
    here and the honest rate to feed the gate is the single-threaded one.

    That is a real difference from lmz, whose native kernel does scale, and it
    caps this codec at roughly 2 GB/s however many cores are present. Fine
    below the gate, which is where a codec that needs nothing installed is
    aimed, and a reason not to reach for it above one.

    Enough blocks are taken to be representative: an earlier version sampled
    8 MiB, which is two blocks, and two blocks is a sample of the header as much
    as of the weights.
    """
    from .zstdcodec import DEFAULT_CHUNK, DEFAULT_LEVEL, _zstd

    zstd = _zstd()
    size = os.path.getsize(path)
    take = min(sample, size)
    with open(path, "rb") as fh:
        fh.seek(max(0, (size - take) // 2))
        raw = fh.read(take)
    if not raw:
        return None, None, None

    # Cut the sample into blocks of the size the archive will really use, so
    # the ratio and the rate both describe the thing that would be built.
    blocks = [raw[i:i + DEFAULT_CHUNK]
              for i in range(0, len(raw), DEFAULT_CHUNK)] or [raw]
    t = time.perf_counter()
    coded = [zstd.compress(b, DEFAULT_LEVEL) for b in blocks]
    enc = len(raw) / max(time.perf_counter() - t, 1e-9)

    t = time.perf_counter()
    got = sum(len(zstd.decompress(b)) for b in coded)
    dec = got / max(time.perf_counter() - t, 1e-9)
    return sum(len(c) for c in coded) / len(raw), enc, dec


def worth_caching(path: str, profile: Profile | None = None,
                  margin: float = _plan.MARGIN) -> Verdict:
    """Would a later load of this file be faster from a coded cache?

    The read gate, applied to a file that is not compressed yet. It needs the
    link rate -- which the profile has if anyone has probed -- and the codec's
    rate on *this data*, which is measured here rather than assumed, because a
    checkpoint's ratio and a text file's are not the same number.
    """
    profile = profile if profile is not None else Profile.load()
    st = profile.storage.get(storage_key(path)) if profile else None
    source = st.source if st else None
    if source is None:
        return Verdict(False, "the link has not been measured, so whether a "
                              "cache would help is unknown — run `lmsluice "
                              "probe` on this path first")

    ratio, _enc, dec = measure(path)
    if ratio is None:
        return Verdict(False, "file is empty")

    d = _plan.read_plan(os.path.getsize(path),
                        int(os.path.getsize(path) * ratio),
                        source=source, decode=dec, margin=margin)
    speedup = d.speedup or 0.0
    if not d.coded:
        return Verdict(False,
                       f"{d.why}. On this link a plain file is the right cache "
                       f"entry", source, dec, ratio, speedup)
    return Verdict(True,
                   f"later loads would be {speedup:.2f}x faster: this codec "
                   f"reads at {dec / 1e9:.2f} GB/s against a link of "
                   f"{source / 1e9:.2f} GB/s, at a ratio of {ratio:.3f}",
                   source, dec, ratio, speedup)


def build(path: str, *, force: bool = False, level: int | None = None,
          chunk_size: int | None = None, codec: str = "auto") -> str:
    """Compress `path` into the cache and return the entry's path.

    `codec="auto"` prefers lmz where it is importable, because it compresses
    better, and falls back to the standard library so that a machine with
    nothing installed still gets a cache.
    """
    dst = entry_for(path)
    if os.path.exists(dst) and not force:
        return dst
    os.makedirs(models_dir(), exist_ok=True)
    free = shutil.disk_usage(models_dir()).free
    need = os.path.getsize(path)
    if free - need < KEEP_FREE:
        raise OSError(f"not enough room: caching needs about {need / 1e9:.1f} GB "
                      f"and {free / 1e9:.1f} GB is free; the cache keeps "
                      f"{KEEP_FREE / 1e9:.0f} GB clear")

    chosen, enc = _encoder(codec)
    opts = {}
    if level is not None:
        opts["level"] = level
    if chunk_size is not None:
        opts["chunk_size"] = chunk_size
    tmp = f"{dst}.{os.getpid()}"
    try:
        enc.encode(path, tmp, **opts)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _note(dst, path, chosen)
    return dst


def _encoder(which: str):
    if which in ("auto", "lmz"):
        try:
            from .lmzcodec import backend_info, encoder as lmz_encoder

            if not backend_info().get("version"):
                raise ImportError(backend_info().get("error", "lmz absent"))
            # 1 MiB blocks so the archive stays device-decodable: lmz reaches
            # its conditioned BF16 codec above a million elements a chunk, and
            # no GPU kernel can read that. Two tenths of a point of ratio.
            return "lmz", _WithDefaults(lmz_encoder(), {"chunk_size": 1 << 20})
        except Exception:                 # noqa: BLE001 -- fall back, do not fail
            if which == "lmz":
                raise
    from .zstdcodec import encoder as zstd_encoder

    return "zstd", zstd_encoder()


class _WithDefaults:
    """An encoder with options pre-set, so the caller still names none."""

    def __init__(self, inner, defaults):
        self._inner, self._defaults = inner, defaults

    def encode_options(self):
        return self._inner.encode_options()

    def encode(self, src, dst, **opts):
        return self._inner.encode(src, dst, **{**self._defaults, **opts})


def _note(dst: str, src: str, codec: str) -> None:
    """A sidecar saying what this entry is, so a cache directory is readable."""
    meta = {"source": os.path.abspath(src), "codec": codec,
            "plain_bytes": os.path.getsize(src),
            "coded_bytes": os.path.getsize(dst), "built_at": time.time()}
    with open(f"{dst}.json", "w") as fh:
        json.dump(meta, fh, indent=1)


def entries() -> list[dict]:
    """What is in the cache, for `lmsluice cache --list`."""
    out = []
    d = models_dir()
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        meta["entry"] = os.path.join(d, name[:-5])
        meta["present"] = os.path.exists(meta["entry"])
        out.append(meta)
    return out


def clear(source: str | None = None) -> int:
    """Remove cache entries. Returns how many. Touches nothing else."""
    n = 0
    for meta in entries():
        if source and os.path.abspath(source) != meta.get("source"):
            continue
        for p in (meta["entry"], meta["entry"] + ".json"):
            if os.path.exists(p):
                os.unlink(p)
        n += 1
    return n
