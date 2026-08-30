"""The command line.

Five verbs, and the order they are listed in is the order they are useful in:
measure the machine, ask what it implies, check the answer against reality,
then move some bytes.

`bench` is the one worth arguing for. A tool that predicts which route is
faster and never checks is a tool that can be wrong indefinitely, and this
package's whole claim is that the arithmetic in `plan.py` describes real
machines. So `bench` runs both routes for real and prints the predicted and
the measured speedup side by side. When they disagree the plan is wrong, and
that is a thing to be able to find out in one command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .plan import gate, write_plan
from .probe import SAMPLE, encode_rate
from .probe import run as probe_run
from .rates import Profile, age, cache_dir, storage_key


def _gb(x) -> str:
    return "  --  " if not x else f"{x / 1e9:6.2f}"


def _bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000.0
    return str(n)


def _profile(args) -> Profile | None:
    p = Profile.load(getattr(args, "profile", None))
    if p is None and not getattr(args, "quiet", False):
        print("no profile for this machine yet -- run `lmsluice probe <a model>` "
              "for a routing decision based on measurement\n", file=sys.stderr)
    return p


# -- verbs -----------------------------------------------------------------
def cmd_probe(args) -> int:
    started = time.perf_counter()
    profile = probe_run(args.target, sample=args.sample, write=not args.no_write,
                        profile=Profile.load(args.profile))
    if args.encode and os.path.isfile(args.target):
        rate, ratio = encode_rate(args.target, sample=args.sample)
        if rate:
            codec = profile.codecs.get("lmz-cpu")
            profile.codecs["lmz-cpu"] = type(codec or _blank())(
                **{**(codec.__dict__ if codec else _blank().__dict__),
                   "encode": rate, "ratio": ratio if ratio else
                   (codec.ratio if codec else None)})
    path = profile.save(args.profile)
    print(f"measured in {time.perf_counter() - started:.1f}s -> {path}\n")
    _show_profile(profile)
    return 0


def _blank():
    from .rates import Codec

    return Codec(name="lmz-cpu")


def _show_profile(p: Profile) -> None:
    h = p.host
    print(f"host    {h.get('system')} {h.get('machine')}, "
          f"{h.get('cpus')} cpus, python {h.get('python')}, "
          f"lmz {h.get('lmz')}")
    if h.get("lmz_backends"):
        b = h["lmz_backends"]
        print(f"        lmz kernel {b.get('kernel')}, entropy "
              f"{b.get('entropy')}, gpu {b.get('gpu')}")
    print(f"\nstorage {'key':<14} {'cold':>7} {'warm':>7} {'write':>7}  GB/s")
    for key, s in p.storage.items():
        note = []
        if s.layered:
            note.append("layered cache below: cold holds on first touch only")
        elif s.cold is not None:
            note.append("cold is an upper bound: a lower cache is not ruled out")
        if not s.method or s.cold is None:
            note.append("cold unavailable: " + (s.method or "no method"))
        note.append(f"depth {s.depth}")
        print(f"        {key:<14} {_gb(s.cold)} {_gb(s.warm)} {_gb(s.write)}"
              f"   ({'; '.join(note)})")
    print(f"\ncodec   {'name':<14} {'decode':>7} {'encode':>7} {'ratio':>7}  GB/s")
    for key, c in p.codecs.items():
        print(f"        {key:<14} {_gb(c.decode)} {_gb(c.encode)} "
              f"{(f'{c.ratio:.3f}' if c.ratio else '  --  '):>7}"
              f"   ({c.threads} threads, on {os.path.basename(c.archive)})")

    for key, s in p.storage.items():
        for c in p.codecs.values():
            g = gate(c.decode, s.source)
            if g is None:
                continue
            verdict = ("compression is free on this path"
                       if g > 1 else "compression is a tax on this path")
            print(f"\ngate    decode / link on {key} = {g:.2f}x -- {verdict}")
            if c.encode and s.write:
                gw = c.encode / s.write
                print(f"        encode / write        = {gw:.2f}x -- "
                      + ("compressing on the way out pays" if gw > 1
                         else "writing plain is faster"))
    return None


def _rate(text: str) -> float:
    """A rate like `450MB/s`, `1.2GB/s`, `125e6`, or a bare bytes-per-second."""
    t = text.strip().lower().removesuffix("/s").removesuffix("ps")
    mult = 1.0
    for suffix, m in (("gb", 1e9), ("mb", 1e6), ("kb", 1e3), ("g", 1e9),
                      ("m", 1e6), ("k", 1e3), ("b", 1.0)):
        if t.endswith(suffix):
            t, mult = t[:-len(suffix)], m
            break
    return float(t) * mult


def cmd_plan(args) -> int:
    from .model import open_model
    from .plan import read_plan

    with open_model(args.target, profile=_profile(args)) as m:
        print(f"{args.target}\n  {m.route} · {len(m.tensors)} tensors · "
              f"{_bytes(m.plain_bytes)} plain · {_bytes(m.coded_bytes)} stored"
              f" · r={m.ratio:.3f}")
        if args.link or args.decode:
            # A what-if, not a measurement, and labelled as one. The question
            # "would this pay on the link I will actually deploy over" is the
            # one a laptop cannot answer by measuring itself, and refusing to
            # answer it is worse than answering it with the asker's own
            # numbers and saying whose they are.
            source = _rate(args.link) if args.link else m.plan.source_rate
            decode = _rate(args.decode) if args.decode else m.plan.codec_rate
            given = ", ".join(x for x in (
                f"link {source / 1e9:.2f} GB/s" if args.link else "",
                f"decode {decode / 1e9:.2f} GB/s" if args.decode else "") if x)
            print(f"  what-if on given rates ({given}); the rest measured")
            d = read_plan(m.plain_bytes,
                          m.coded_bytes if m.route == "coded" else None,
                          source=source, decode=decode)
            print(d.explain())
        else:
            print(m.plan.explain())
    return 0


def cmd_info(args) -> int:
    from .model import open_model

    with open_model(args.target, profile=_profile(args)) as m:
        print(f"{args.target}\n  route {m.route} · {len(m.tensors)} tensors · "
              f"{_bytes(m.plain_bytes)} plain · r={m.ratio:.3f}")
        rows = sorted(m.tensors.values(), key=lambda t: -t.nbytes)
        n = len(rows) if args.all else min(len(rows), 20)
        print(f"\n  {'tensor':<52} {'dtype':>6} {'bytes':>10}  shape")
        for t in rows[:n]:
            print(f"  {t.name[:52]:<52} {t.dtype:>6} {t.nbytes:>10}  "
                  f"{list(t.shape)}")
        if n < len(rows):
            print(f"  ... {len(rows) - n} more (use --all)")
    return 0


def cmd_get(args) -> int:
    """Move a model to a local plain file, over whatever route it came by."""
    from .model import open_model

    started = time.perf_counter()
    with open_model(args.target, profile=_profile(args)) as m:
        if os.path.exists(args.dst) and not args.force:
            print(f"{args.dst} exists (use --force)", file=sys.stderr)
            return 1
        n = m.to_file(args.dst, overwrite=args.force)
    elapsed = time.perf_counter() - started
    print(f"{_bytes(n)} in {elapsed:.2f}s = {n / elapsed / 1e9:.2f} GB/s "
          f"plain, over the {m.route} route "
          f"({_bytes(m.coded_bytes)} actually moved)")
    return 0


def cmd_bench(args) -> int:
    """Run both routes for real and check the plan against what happens."""
    from .model import open_model
    from .source import FileSource

    profile = _profile(args)
    pairs = [(args.target, None)] if args.against is None else \
            [(args.target, args.against)]
    for target, other in pairs:
        results = {}
        for path in [p for p in (target, other) if p]:
            with open_model(path, profile=profile) as m:
                times = []
                for i in range(args.reps):
                    if args.cold and isinstance(getattr(m, "_src", None), FileSource):
                        m._src.uncache()
                    elif args.cold and m._arc is not None \
                            and isinstance(m._arc.source, FileSource):
                        m._arc.source.uncache()
                    buf = bytearray(m.plain_bytes)
                    t = time.perf_counter()
                    m.load(into=buf)
                    times.append(time.perf_counter() - t)
                    del buf
                best = min(times)
                results[m.route] = (best, m.plain_bytes, m.plan, times)
                print(f"{os.path.basename(path):<28} {m.route:<6} "
                      f"{m.plain_bytes / best / 1e9:5.2f} GB/s  "
                      f"best of {args.reps}  "
                      f"({', '.join(f'{x:.2f}s' for x in times)})")
        if len(results) == 2:
            pt = results["plain"][0]
            ct = results["coded"][0]
            measured = pt / ct
            predicted = results["coded"][2].decision.speedup
            print(f"\n  measured  coded is {measured:.2f}x the plain route")
            if predicted:
                print(f"  predicted            {predicted:.2f}x")
                off = abs(measured - predicted) / max(predicted, 1e-9)
                print(f"  the plan is {'right' if off < 0.25 else 'WRONG'} "
                      f"here, off by {off * 100:.0f}%")
    return 0


def cmd_write(args) -> int:
    """What compressing on the way out would cost, rather than on the way in."""
    profile = _profile(args)
    key = storage_key(args.target)
    st = profile.storage.get(key) if profile else None
    codec = profile.codecs.get("lmz-cpu") if profile else None
    size = os.path.getsize(args.target)
    if args.measure or not (codec and codec.encode):
        rate, ratio = encode_rate(args.target, sample=args.sample)
        enc, r = rate, ratio
    else:
        enc, r = codec.encode, codec.ratio
    d = write_plan(size, r or 1.0, sink=st.write if st else None, encode=enc)
    print(f"{args.target}  {_bytes(size)}  r={r:.3f}" if r else args.target)
    print(d.explain())
    return 0


def cmd_doctor(args) -> int:
    p = Profile.load(args.profile)
    print(f"lmsluice {__version__}")
    try:
        from ._lmz import lmz

        m = lmz()
        print(f"lmz  {m.__version__}  {m.backends()}")
    except ImportError as exc:
        print(f"lmz  not available: {exc}")
    print(f"cache {cache_dir()}")
    if p is None:
        print("\nno profile for this machine. run:  lmsluice probe <a model file>")
        return 0
    print(f"profile taken {age(p) / 3600:.1f} hours ago\n")
    _show_profile(p)
    return 0


# -- wiring ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lmsluice", description="Move model bytes at the rate the machine "
                                 "allows, over the route measurement picks.")
    ap.add_argument("--version", action="version", version=f"lmsluice {__version__}")
    ap.add_argument("--profile", help="profile file to use instead of the cached one")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="measure this machine and cache the result")
    p.add_argument("target", help="a model file or archive on the path to measure")
    p.add_argument("--sample", type=int, default=SAMPLE,
                   help=f"bytes per measurement (default {SAMPLE >> 20} MiB)")
    p.add_argument("--no-write", action="store_true",
                   help="skip the write measurement, which creates a temp file")
    p.add_argument("--encode", action="store_true",
                   help="also measure compression, by compressing a sample")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("plan", help="say which route is faster, and why")
    p.add_argument("target")
    p.add_argument("--link", help="ask about a link you do not have, "
                                  "e.g. 125MB/s, 1.2GB/s")
    p.add_argument("--decode", help="ask about a decoder you do not have, "
                                    "e.g. the 418GB/s CUDA one")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("info", help="what is in the model")
    p.add_argument("target")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("get", help="move a model to a plain local file")
    p.add_argument("target", help="path, lmz archive, or https URL")
    p.add_argument("dst")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("bench", help="run both routes and check the plan")
    p.add_argument("target")
    p.add_argument("--against", help="the other form of the same model")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cold", action="store_true",
                   help="drop the cache before each run, where the platform allows")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("write", help="price compressing on the way out")
    p.add_argument("target", help="a plain model file")
    p.add_argument("--measure", action="store_true", help="re-measure the encoder")
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("doctor", help="what is measured and what is active")
    p.set_defaults(func=cmd_doctor)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"lmsluice: {exc}", file=sys.stderr)
        return 1
