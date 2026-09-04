"""How fast this machine decrypts, and whether it ever becomes the binding stage.

Encryption adds a stage to the coded route, and `plan.py` prices it like any
other: coded bytes in, coded bytes out, on the fetch pool. That pricing needs a
rate, and a rate is a measurement, not a manufacturer's figure -- AES-NI is
present on every x86 worth shipping to and absent from plenty of the machines
this project exists for.

What this reports is deliberately *not* one number:

- **Per thread and aggregate**, swept, because the fetch pool has more than one
  thread and AES-GCM parallelises across independent chunks perfectly (each has
  its own nonce and its own tag) while sharing one memory system.
- **Against chunk size**, because a 16-byte tag on a 64 KiB chunk is a different
  proposition from one on 8 MiB, and because small chunks spend their time in
  the EVP call rather than in AES.
- **Labelled with the machine**, because "3 GB/s" means nothing without the core
  it ran on and whether that core has AES-NI.

Run:  python experiments/decrypt_rate.py [--seconds 2] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmsluice import crypt          # noqa: E402


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def has_aes_ni() -> bool | None:
    """None when it cannot be determined, which is not the same as False."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                return "aes" in line.split()
    except OSError:
        return None
    return None


def _run(key, blobs, seconds):
    """Decrypt round-robin for `seconds`; return bytes of ciphertext consumed."""
    done, i, deadline = 0, 0, time.perf_counter() + seconds
    n = len(blobs)
    while time.perf_counter() < deadline:
        for _ in range(8):
            idx, blob = blobs[i % n]
            crypt.open_chunk(key, idx, blob)
            done += len(blob)
            i += 1
    return done


# Ciphertext each thread gets to itself. Sharing one set across the pool made
# every thread read the same pages at the same time and turned a cipher
# measurement into a measurement of one shared working set: the first sweep
# fell from 53 GB/s on four threads to 17 on eight for that reason and no
# other. In the real fetch pool each thread holds a different chunk.
PER_THREAD = 8


def sweep(chunk: int, threads: list[int], seconds: float) -> list[dict]:
    key = crypt.subkey(crypt.new_key(), crypt.new_salt())
    rows = []
    for t in threads:
        sets = [[(j * PER_THREAD + i, crypt.seal(key, j * PER_THREAD + i,
                                                 os.urandom(chunk)))
                 for i in range(PER_THREAD)] for j in range(t)]
        # Warm once: the first EVP call in a process pays for symbol binding.
        _run(key, sets[0], 0.05)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=t) as pool:
            total = sum(f.result() for f in
                        [pool.submit(_run, key, s, seconds) for s in sets])
        elapsed = time.perf_counter() - started
        rows.append({"chunk": chunk, "threads": t,
                     "rate": total / elapsed,
                     "per_thread": total / elapsed / t})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    name, why = crypt.backend()
    if name == "none":
        print(f"no backend: {why}")
        return 1

    cores = os.cpu_count() or 1
    threads = sorted({1, 2, 4, min(8, cores), cores})
    machine = {"cpu": cpu_name(), "cores": cores, "aes_ni": has_aes_ni(),
               "backend": name, "library": why,
               "python": platform.python_version(),
               "platform": platform.platform()}
    print(f"{machine['cpu']}  ·  {cores} logical cores  ·  "
          f"AES-NI {machine['aes_ni']}")
    print(f"{crypt.describe()}\n")

    rows = []
    print(f"  {'chunk':>9} {'threads':>8} {'GB/s':>8} {'GB/s/thread':>12}")
    for chunk in (64 << 10, 1 << 20, 8 << 20):
        for r in sweep(chunk, threads, args.seconds):
            rows.append(r)
            print(f"  {r['chunk'] >> 10:>7} KiB {r['threads']:>8} "
                  f"{r['rate'] / 1e9:>8.2f} {r['per_thread'] / 1e9:>12.2f}")

    best = max(rows, key=lambda r: r["rate"])
    one = min((r for r in rows if r["threads"] == 1),
              key=lambda r: -r["rate"])
    print(f"\n  one thread   {one['rate'] / 1e9:.2f} GB/s at "
          f"{one['chunk'] >> 10} KiB")
    print(f"  aggregate    {best['rate'] / 1e9:.2f} GB/s at "
          f"{best['threads']} threads, {best['chunk'] >> 10} KiB")
    print(f"\n  These are ciphertext bytes -- coded bytes, not plain ones, "
          f"which is\n  the unit plan.py prices the decrypt stage in.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"machine": machine, "rows": rows}, fh, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
