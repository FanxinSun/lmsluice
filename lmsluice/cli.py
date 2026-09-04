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
from . import buffer as _buffer
from .plan import gate_ratio, write_plan
from .probe import SAMPLE, SEALED_FETCH, encode_rate
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
    from .probe import decrypt_rate

    profile.decrypt = decrypt_rate()
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
    if p.decrypt:
        from . import crypt

        print(f"        decrypt {p.decrypt / 1e9:.2f} GB/s of ciphertext "
              f"({crypt.backend()[0]}, {SEALED_FETCH} threads)")
    if h.get("lmz_backends"):
        b = h["lmz_backends"]
        print(f"        lmz kernel {b.get('kernel')}, entropy "
              f"{b.get('entropy')}, gpu {b.get('gpu')}")
    print(f"\nstorage {'key':<14} {'cold':>7} {'warm':>7} {'write':>7}  GB/s")
    for key, s in p.storage.items():
        if key == "pcie":
            # Not a disk, and the cold/warm columns mean something else here:
            # pinned and pageable. Printed in its own row rather than being
            # squeezed into headings that would misname both numbers.
            print(f"        {'pcie':<14} {_gb(s.cold)} {_gb(s.warm)}"
                  f" {'  --  ':>7}   (pinned / pageable; {s.method})")
            continue
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
    how, why, floor = _buffer.strategy()
    print(f"\nbuffer  {how:<14} {why}")
    if how == "mapped":
        print(f"        {'':<14} used for destinations from "
              f"{floor >> 20} MiB up; smaller ones stay on the heap")
    else:
        print(f"        {'':<14} destinations are zero-filled on the heap, "
              f"which costs a full pass over them")
    print(f"\ncodec   {'name':<14} {'decode':>7} {'encode':>7} {'ratio':>7}  GB/s")
    for key, c in p.codecs.items():
        print(f"        {key:<14} {_gb(c.decode)} {_gb(c.encode)} "
              f"{(f'{c.ratio:.3f}' if c.ratio else '  --  '):>7}"
              f"   ({c.threads} threads, on {os.path.basename(c.archive)})")

    for key, s in p.storage.items():
        if key == "pcie":
            continue        # the PCIe gate belongs to the VRAM plan, not here
        for c in p.codecs.values():
            g = gate_ratio(c.decode, s.source)
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
        if m.route == "coded":
            try:
                from .devdecode import advice

                tip = advice(m._arc)
            except Exception:             # noqa: BLE001
                tip = ""
            if tip:
                print(f"\n  note: {tip}")
    return 0


def cmd_info(args) -> int:
    from .model import open_model

    with open_model(args.target, profile=_profile(args)) as m:
        print(f"{args.target}\n  route {m.route} · {len(m.tensors)} tensors · "
              f"{_bytes(m.plain_bytes)} plain · r={m.ratio:.3f}")
        store = _store_of(m)
        if store:
            print(f"  from {store}")
        enc = getattr(getattr(getattr(m, "_arc", None), "source", None),
                      "encryption", None)
        if enc:
            state = "verified" if m._arc.source.verified else \
                    "NOT verified (no key supplied)"
            print(f"  encrypted {enc['algorithm']} via {enc['backend']} · key "
                  f"id {enc['key_id']} · structure {state}")
        rows = sorted(m.tensors.values(), key=lambda t: -t.nbytes)
        n = len(rows) if args.all else min(len(rows), 20)
        print(f"\n  {'tensor':<52} {'dtype':>6} {'bytes':>10}  shape")
        for t in rows[:n]:
            print(f"  {t.name[:52]:<52} {t.dtype:>6} {t.nbytes:>10}  "
                  f"{list(t.shape)}")
        if n < len(rows):
            print(f"  ... {len(rows) - n} more (use --all)")
    return 0


def _store_of(m):
    """`s3 · stdlib · signed`, or nothing when the source is not a bucket."""
    src = getattr(getattr(m, "_arc", None), "source", None) or getattr(m, "_src", None)
    src = getattr(src, "_inner", src)     # a store that refused ranges
    if getattr(src, "cloud", None) is None:
        return ""
    from .cloud import describe

    return describe(src)


def cmd_get(args) -> int:
    """Move a model to a local plain file, over whatever route it came by."""
    from .model import open_model

    started = time.perf_counter()
    with open_model(args.target, profile=_profile(args)) as m:
        if os.path.exists(args.dst) and not args.force:
            print(f"{args.dst} exists (use --force)", file=sys.stderr)
            return 1
        n = m.to_file(args.dst, overwrite=args.force)
        # Read before the model closes: closing releases the source, and the
        # store's name goes with it.
        store = _store_of(m)
    elapsed = time.perf_counter() - started
    print(f"{_bytes(n)} in {elapsed:.2f}s = {n / elapsed / 1e9:.2f} GB/s "
          f"plain, over the {m.route} route "
          f"({_bytes(m.coded_bytes)} actually moved"
          + (f", from {store}" if store else "") + ")")
    return 0


def cmd_bench(args) -> int:
    """Run both routes for real and check the plan against what happens."""
    from .model import open_model
    from .source import FileSource

    profile = _profile(args)
    pairs = [(args.target, None)] if args.against is None else \
            [(args.target, args.against)]
    cold_warned = False
    for target, other in pairs:
        results = {}
        for path in [p for p in (target, other) if p]:
            with open_model(path, profile=profile) as m:
                times = []
                for i in range(args.reps):
                    if args.cold:
                        # Say so when the drop fails, rather than labelling the
                        # run "cold" because the flag was passed.
                        src = (m._src if isinstance(getattr(m, "_src", None),
                                                    FileSource)
                               else getattr(m._arc, "source", None))
                        if isinstance(src, FileSource):
                            ok, how = src.uncache()
                            if not ok and not cold_warned:
                                print(f"  --cold could not drop the cache: "
                                      f"{how}; these runs are warm",
                                      file=sys.stderr)
                                cold_warned = True
                    if args.to_device:
                        t = time.perf_counter()
                        dev = m.to_device()
                        times.append(time.perf_counter() - t)
                        dev.close()
                    else:
                        buf = bytearray(m.plain_bytes)
                        t = time.perf_counter()
                        m.load(into=buf)
                        times.append(time.perf_counter() - t)
                        del buf
                best = min(times)
                plan = m.device_plan() if args.to_device else m.plan
                results[m.route] = (best, m.plain_bytes, plan, times)
                enc = m.encrypted
                # Bytes beside seconds, always. On a metered link or a billed
                # bucket the first number is the one that is actually paid for,
                # and a rate alone hides it -- the coded route is faster *and*
                # moves a third less, and only one of those shows up on a bill.
                moved = (m.coded_bytes if m.route == "coded" else m.plain_bytes)
                store = _store_of(m)
                print(f"{os.path.basename(path):<28} {m.route:<6} "
                      f"{m.plain_bytes / best / 1e9:5.2f} GB/s  "
                      f"{_bytes(moved):>9} moved  "
                      f"best of {args.reps}  "
                      f"({', '.join(f'{x:.2f}s' for x in times)})"
                      + (f"  ·  {store}" if store else "")
                      + (f"  ·  {enc['algorithm']} via {enc['backend']}, "
                         f"{m._fetch_threads} fetch threads" if enc else ""))
        if len(results) == 2:
            pt = results["plain"][0]
            ct = results["coded"][0]
            measured = pt / ct
            got = results["coded"][2]
            predicted = (got.speedup if hasattr(got, "speedup")
                         else got.decision.speedup)
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
    # The encrypted row appears only when this machine can actually seal, and
    # is priced from rates already measured here rather than a new constant.
    from . import crypt
    from .plan import seal_rate

    seal = None
    if args.seal and crypt.available():
        seal = seal_rate(source=st.source or st.warm_bound if st else None,
                         sink=st.write if st else None,
                         encrypt=profile.decrypt if profile else None)
        if seal is None:
            print("  note: the encrypted row needs a read rate, a write rate "
                  "and a decrypt rate for this machine. Run: lmsluice probe",
                  file=sys.stderr)
    elif args.seal:
        print(f"  note: cannot price encryption here: {crypt.backend()[1]}",
              file=sys.stderr)
    d = write_plan(size, r or 1.0, sink=st.write if st else None, encode=enc,
                   seal=seal)
    print(f"{args.target}  {_bytes(size)}  r={r:.3f}" if r else args.target)
    print(d.explain())
    if seal:
        print(f"\n  the encrypted row is a second pass: the archive is written, "
              f"then read back\n  and rewritten sealed, so both copies exist at "
              f"once. Peak disk is the\n  GB-on-disk column, not the GB moved "
              f"one.")
    return 0


def cmd_cache(args) -> int:
    """Say whether a local coded copy would pay, and build it if asked."""
    from . import cache

    if args.list:
        rows = cache.entries()
        if not rows:
            print("cache is empty")
            return 0
        print(f"{'codec':<6} {'plain':>10} {'coded':>10} {'ratio':>6}  source")
        for m in rows:
            print(f"{m['codec']:<6} {_bytes(m['plain_bytes']):>10} "
                  f"{_bytes(m['coded_bytes']):>10} "
                  f"{m['coded_bytes'] / max(1, m['plain_bytes']):>6.3f}  "
                  f"{m['source']}")
        return 0
    if args.clear:
        print(f"removed {cache.clear(args.target)} entries")
        return 0
    if not args.target:
        print("cache: give a model, or --list / --clear", file=sys.stderr)
        return 1

    v = cache.worth_caching(args.target, _profile(args))
    print(v.explain())
    if not args.build:
        if v.worth_it:
            print("  (pass --build to make it)")
        return 0
    if not v.worth_it:
        print("  refusing to build a cache the gate says would not help; "
              "pass --codec explicitly if you want it anyway", file=sys.stderr)
        return 1
    started = time.perf_counter()
    entry = cache.build(args.target, codec=args.codec)
    print(f"  built in {time.perf_counter() - started:.1f}s -> "
          f"{_bytes(os.path.getsize(entry))} at {entry}")
    return 0


def cmd_put(args) -> int:
    """Upload a local file to a bucket.

    Chunked above one part, and it says how many bytes crossed the link --
    on a metered connection or a billed bucket that is the number paid for,
    and a rate alone hides it.
    """
    from .cloud import put

    if not os.path.isfile(args.src):
        print(f"{args.src}: not a file", file=sys.stderr)
        return 1
    size = os.path.getsize(args.src)
    shown = [0]

    def progress(done, total):
        if not args.quiet and done - shown[0] >= (32 << 20):
            shown[0] = done
            print(f"  {_bytes(done)} of {_bytes(total)}", file=sys.stderr)

    started = time.perf_counter()
    report = put(args.src, args.dst, part_bytes=args.part_size,
                 progress=progress)
    elapsed = time.perf_counter() - started
    print(f"{args.dst}: {_bytes(size)} in {elapsed:.2f}s = "
          f"{size / elapsed / 1e9:.2f} GB/s over the link "
          f"({report['parts']} × {report['method']}, to {report['store']})")
    return 0


def cmd_seal(args) -> int:
    """Wrap an archive in an encrypted envelope, or read one back out.

    The key is a *path*, never the key itself. argv is world-readable out of
    `/proc` on Linux and shows up in shell history everywhere, so a flag that
    took the key would hand it to the machine. `--key-file` and
    `LMSLUICE_KEY_FILE` both name a file.
    """
    from . import crypt, sealed

    if args.make_key:
        if os.path.exists(args.make_key) and not args.force:
            print(f"{args.make_key} exists (use --force)", file=sys.stderr)
            return 1
        fd = os.open(args.make_key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(crypt.new_key())
        print(f"wrote a new 32-byte key to {args.make_key} (mode 600). "
              f"Lose it and the archives it seals are gone.")
        return 0

    if args.src is None:
        print("nothing to do: pass an archive, or --make-key PATH",
              file=sys.stderr)
        return 2
    if not crypt.available():
        print(f"cannot encrypt: {crypt.backend()[1]}", file=sys.stderr)
        return 1
    if os.path.exists(args.dst) and not args.force:
        print(f"{args.dst} exists (use --force)", file=sys.stderr)
        return 1

    started = time.perf_counter()
    header = sealed.seal_file(args.src, args.dst, args.key_file)
    elapsed = time.perf_counter() - started
    n = os.path.getsize(args.src)
    grew = os.path.getsize(args.dst) - n
    print(f"{args.dst}: {_bytes(os.path.getsize(args.dst))} sealed in "
          f"{elapsed:.2f}s = {n / elapsed / 1e9:.2f} GB/s")
    print(f"  {header['algorithm']} via {header['backend']}, key id "
          f"{header['key_id']}, +{grew} bytes of tags and index")
    print(f"  structure stays readable without the key; the weights do not")
    return 0


def cmd_doctor(args) -> int:
    p = Profile.load(args.profile)
    print(f"lmsluice {__version__}")
    from .lmzcodec import backend_info

    codec = backend_info()
    if codec.get("version"):
        print(f"codec {codec['name']} {codec['version']}  {codec.get('backends')}")
    else:
        print(f"codec {codec['name']} not available: {codec.get('error')}")
    from . import cuda

    ok, why = cuda.available()
    if ok:
        info = cuda.device_info()
        print(f"cuda {info['name']} sm_{info.get('cc_major','?')}"
              f"{info.get('cc_minor','')} · "
              f"{info['total_bytes'] / 1e9:.1f} GB")
    else:
        print(f"cuda not usable: {why}")
    from . import crypt

    name, why = crypt.backend()
    if name == "none":
        print(f"encryption unavailable: {why}")
    else:
        print(f"encryption AES-256-GCM via {name} ({why})")
    from .cloud import aws_credentials, azure_credentials, gcs_credentials

    modes = []
    for label, found in (("s3", aws_credentials()), ("gcs", gcs_credentials()),
                         ("azure", azure_credentials())):
        if label == "s3":
            how = ("credentials found" if found.get("access_key")
                   else "anonymous only")
        else:
            how = found.get("mode", "anonymous only")
        modes.append(f"{label} {how}")
    print(f"buckets {', '.join(modes)}  (stdlib signing, nothing installed)")
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
    p.add_argument("--to-device", action="store_true",
                   help="load into VRAM rather than host RAM")
    p.add_argument("--against", help="the other form of the same model")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cold", action="store_true",
                   help="drop the cache before each run, where the platform allows")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("write", help="price compressing on the way out")
    p.add_argument("--seal", action="store_true",
                   help="also price encrypting it, wall clock and peak disk")
    p.add_argument("target", help="a plain model file")
    p.add_argument("--measure", action="store_true", help="re-measure the encoder")
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("cache", help="compress a model locally, where it pays")
    p.add_argument("target", nargs="?", help="a plain model file")
    p.add_argument("--build", action="store_true",
                   help="build the entry (otherwise only says whether it pays)")
    p.add_argument("--codec", default="auto", choices=("auto", "lmz", "zstd"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("put", help="upload a file to s3://, gs:// or az://")
    p.add_argument("src", help="the local file to upload")
    p.add_argument("dst", help="s3://bucket/key, gs://bucket/key, "
                               "or az://account/container/blob")
    p.add_argument("--part-size", type=int, default=8 << 20,
                   help="bytes per part (default 8 MiB; S3 needs at least 5 "
                        "MiB for every part but the last)")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("seal", help="encrypt an archive at rest (AES-256-GCM)")
    p.add_argument("src", nargs="?", help="the archive to wrap")
    p.add_argument("dst", nargs="?", help="where to write the envelope")
    p.add_argument("--key-file", help="file holding a 32-byte key or 64 hex "
                                      "characters; defaults to $LMSLUICE_KEY_FILE")
    p.add_argument("--make-key", metavar="PATH",
                   help="write a fresh random key and exit")
    p.add_argument("--force", action="store_true", help="overwrite the target")
    p.set_defaults(func=cmd_seal)

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
    except Exception as exc:              # noqa: BLE001
        # A missing or wrong key is something a user did, not something that
        # broke, and a traceback for it buries the one sentence that helps.
        from .crypt import CryptoUnavailable

        # `CloudError` needs no entry here: it subclasses `OSError` and is
        # caught above. Listing it as well would say it is not.
        if not isinstance(exc, CryptoUnavailable):
            raise
        print(f"lmsluice: {exc}", file=sys.stderr)
        return 1
