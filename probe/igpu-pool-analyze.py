#!/usr/bin/env python3
"""Analyse a shared-memory-pool staircase: how many threadgroups are resident at once?

Reads the JSON written by `probe/igpu-pool.ps1` (D3D11) or by
`probe/metal-bench.js` (Metal), applies one set of rules to every shape, and
reports -- per shape and combined -- the per-unit residency and the bounds on the
per-unit shared-memory pool that follow from it. Standard library only.

    python3 igpu-pool-analyze.py run1.json [run2.json ...] [--units N] [--expect SHAPE=TOTAL ...]
    python3 igpu-pool-analyze.py --selftest

The rules, stated once because a "staircase" read by eye is an opinion:

  * A jump is a time >= LOCAL x the median of the (up to four) points just before it
    on the same plateau. Comparing with the LOCAL plateau, not with the one-group time,
    keeps a gradual slow-down inside a wave from being mistaken for a step.
  * wave-1 capacity C: the first jump (LOCAL = 1.35) puts it in [G_before, G_jump). The
    grid is coarse past G = 64, so the capacity is an interval, not a point.
  * a shape is CLEAN only if (a) that jump exists, (b) a second jump (SECOND = 1.2 over
    its own new plateau) follows near twice the capacity, and (c) with a unit count,
    exactly one integer groups-per-unit k has units*k inside [G_before, G_jump). Unit
    counts are typed in by the operator (D3D11 cannot query them); without one,
    per-unit values are skipped.
  * a clean shape with k groups of S bytes per unit says  k*S <= pool < (k+1)*S.
    The pool bound is the intersection over REPRODUCIBLE clean shapes. If the
    intersection is empty the shapes disagree and NO pool is reported -- the
    disagreement is the finding.
  * REPRODUCIBLE means clean, with the same per-unit value, in EVERY run given. Give
    this script at least two runs taken at different times. On the Apple M4 a boundary
    shape (S = 20,480 B) read 4 groups per core in one run and 3.4 in another: one run
    cannot tell a boundary shape from a clean one, and a bound built on it is wrong.

Nothing here knows any device. Every constant is a rule above, and every input is the
JSON's own.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

LOCAL, SECOND, WINDOW = 1.35, 1.2, 4


def _median(xs):
    return statistics.median(xs)


def _next_jump(ms, plateau_start, scan_from, factor):
    """First index i >= scan_from whose time is >= factor x the median of the (up to
    WINDOW) points before it on the current plateau, or None."""
    for i in range(max(scan_from, plateau_start + 3), len(ms)):
        level = _median(ms[max(plateau_start, i - WINDOW):i])
        if ms[i] >= factor * level:
            return i
    return None


def analyze_shape(g, ms, units=None, size=None):
    """One shape's staircase -> a dict with its capacity interval and verdict."""
    out = {"clean": False, "reason": "", "cap_lo": None, "cap_hi": None}
    n = len(g)
    if n < 8:
        out["reason"] = "too few points"
        return out
    t1 = _median(ms[:3])
    first = _next_jump(ms, 0, 3, LOCAL)
    if first is None:
        out["reason"] = "no jump: the sweep never left the first wave (raise -MaxGroups)"
        return out
    lo, hi = g[first - 1], g[first]                 # capacity in [lo, hi)
    out.update(t1_ms=t1, cap_lo=lo, cap_hi=hi)

    # The second jump, measured against the new plateau, near twice the capacity.
    second = _next_jump(ms, first, first + 4, SECOND)
    if second is None:
        out["reason"] = "no second wave seen (sweep ended early)"
        return out
    if not (2 * lo - 1 <= g[second] <= 2 * hi + 4):
        out["reason"] = f"second step at G={g[second]}, not near 2x the first ({lo}..{hi})"
        return out

    if units:
        ks = [k for k in range(1, hi // units + 2) if lo <= units * k < hi]
        if len(ks) != 1:
            out["reason"] = (f"capacity interval [{lo}, {hi}) holds {len(ks)} multiples of "
                             f"{units} units: cannot name an integer per-unit residency")
            return out
        k = ks[0]
        out.update(per_unit=k, total=units * k)
        if size:
            out.update(pool_lo=k * size, pool_hi=(k + 1) * size)
    out["clean"] = True
    out["reason"] = "clean"
    return out


def _load_points(doc):
    """Yield (name, T, S, G[], ms[]) from either tool's JSON."""
    if "shapes" in doc and doc["shapes"] and "G" in doc["shapes"][0]:        # igpu-pool.ps1
        for s in doc["shapes"]:
            if s.get("error") or not s.get("G"):
                continue
            yield s["name"], s["T"], s["S"], s["G"], s["median_ms"]
        return
    st = doc.get("staircase", {})                                            # metal-bench.js
    for grp in ("shapes", "sSweep"):
        for s in st.get(grp, []):
            pts = sorted(s["points"], key=lambda p: p["G"])
            if s.get("S"):
                yield s["name"], s["T"], s["S"], [p["G"] for p in pts], [p["timeMs"] for p in pts]


def analyze(docs, units=None, expect=None):
    """Analyse one or more runs. A shape counts only if clean and in agreement in all."""
    if isinstance(docs, dict):
        docs = [docs]
    per_run, units_used = [], units
    for doc in docs:
        u = units or doc.get("units_given_by_operator") \
            or next((d.get("gpuCoreCount") for d in doc.get("deviceFacts", [])), None)
        units_used = units_used or u
        rows = {}
        for name, T, S, g, ms in _load_points(doc):
            r = analyze_shape(g, ms, u, S)
            r.update(name=name, T=T, S=S)
            rows[name] = r
        per_run.append(rows)

    names = []
    for rows in per_run:
        for n in rows:
            if n not in names:
                names.append(n)
    merged, lows, highs = [], [], []
    for name in names:
        runs = [rows.get(name) for rows in per_run]
        base = next(r for r in runs if r is not None)
        row = {"name": name, "T": base["T"], "S": base["S"], "runs": len(per_run),
               "clean_runs": sum(1 for r in runs if r and r["clean"]),
               "cap_lo": base.get("cap_lo"), "cap_hi": base.get("cap_hi")}
        if any(r is None for r in runs):
            row.update(reproducible=False, reason="missing from some runs")
        elif not all(r["clean"] for r in runs):
            why = "; ".join(f"run {i + 1}: {r['reason']}" for i, r in enumerate(runs) if not r["clean"])
            row.update(reproducible=False, reason=f"not clean in every run ({why})")
        else:
            ks = {r.get("per_unit") for r in runs}
            if None in ks:
                row.update(reproducible=False, reason="clean, but no per-unit value (units unknown)")
            elif len(ks) > 1:
                row.update(reproducible=False,
                           reason="IRREPRODUCIBLE: per-unit residency differs between runs "
                                  + str([r["per_unit"] for r in runs]))
            else:
                k = ks.pop()
                row.update(reproducible=True, per_unit=k, total=base["total"],
                           pool_lo=k * base["S"], pool_hi=(k + 1) * base["S"],
                           reason="reproducible" if len(per_run) > 1 else "clean (single run: not checked)")
                lows.append(row["pool_lo"]); highs.append(row["pool_hi"])
        merged.append(row)

    pool = None
    if lows:
        lo, hi = max(lows), min(highs)
        pool = {"lower": lo, "upper": hi, "consistent": lo < hi, "shapes": len(lows),
                "runs": len(per_run), "single_run": len(per_run) == 1}
    calib = []
    for item in expect or []:
        want_name, want = item.split("=")
        want = int(want)
        rows = [r.get(want_name) for r in per_run]
        if any(r is None or r.get("cap_lo") is None for r in rows):
            calib.append({"shape": want_name, "expected": want, "ok": False, "why": "no staircase for that shape"})
            continue
        calib.append({"shape": want_name, "expected": want,
                      "observed_intervals": [[r["cap_lo"], r["cap_hi"]] for r in rows],
                      "ok": all(r["cap_lo"] <= want < r["cap_hi"] for r in rows)})
    return {"units": units_used, "runs": len(per_run), "shapes": merged, "pool": pool,
            "calibration": calib}


def render(res):
    lines = [f"runs: {res['runs']}   units: {res['units'] if res['units'] else 'unknown (per-unit values skipped)'}", ""]
    lines.append(f"{'shape':<28}{'T':>5}{'S':>7}  {'capacity in':<12}{'per unit':>9}  {'pool in (B)':<18}{'clean':>6}  verdict")
    for r in res["shapes"]:
        cap = f"[{r['cap_lo']}, {r['cap_hi']})" if r.get("cap_lo") is not None else "-"
        pu = r.get("per_unit", "-")
        pool = f"[{r['pool_lo']}, {r['pool_hi']})" if "pool_lo" in r else "-"
        lines.append(f"{r['name']:<28}{r['T']:>5}{r['S']:>7}  {cap:<12}{pu!s:>9}  {pool:<18}"
                     f"{r['clean_runs']}/{r['runs']:>3}  {r['reason']}")
    lines.append("")
    p = res["pool"]
    if p is None:
        lines.append("POOL: not determined (needs reproducible clean shapes and a unit count).")
    elif p["consistent"]:
        lines.append(f"POOL per unit: {p['lower']} <= pool < {p['upper']} bytes "
                     f"(intersection of {p['shapes']} reproducible shapes over {p['runs']} run(s))")
    else:
        lines.append(f"POOL: the {p['shapes']} reproducible shapes DISAGREE (max lower {p['lower']} >= min "
                     f"upper {p['upper']}). No pool is reported; the disagreement is the result.")
    if p and p["single_run"]:
        lines.append("WARNING: ONE RUN. Shapes were not checked for reproducibility, and a boundary shape "
                     "reads like a clean one in a single run. Take a second run at a different time and "
                     "pass both files.")
    for c in res["calibration"]:
        obs = c.get("observed_intervals", "?")
        lines.append(f"CALIBRATION {c['shape']}: expected total {c['expected']}, observed {obs}  "
                     f"-> {'PASS' if c['ok'] else 'FAIL'}")
    return "\n".join(lines)


def _synthetic(cap, t1=1.0, slope=0.002, step_tail=True):
    """A latency-bound wave curve with the grid the kit uses."""
    gs = list(range(1, 65)) + list(range(66, 201, 2)) + list(range(204, 1300, 4))
    ms, g_out = [], []
    for g in gs:
        waves = math.ceil(g / cap)
        ms.append(t1 * waves * (1 + slope * min(g, cap)))
        g_out.append(g)
        if waves >= 4:
            break
    return g_out, ms


def selftest():
    ok = True
    for cap, units in ((6, 2), (8, 2), (252, 84), (40, 10), (30, 10), (20, 10)):
        g, ms = _synthetic(cap)
        r = analyze_shape(g, ms, units, 31744)
        good = r["clean"] and r.get("total") == cap
        print(f"synthetic cap {cap:>3} units {units:>2}: ", "ok" if good else f"FAIL {r}")
        ok &= good
    # an ambiguous case must refuse rather than guess: cap interval [200,204) with 84 units has no multiple
    g, ms = _synthetic(201)
    r = analyze_shape(g, ms, 84, 4096)
    good = not r["clean"]
    print("ambiguous interval refuses:     ", "ok" if good else f"FAIL {r}")
    ok &= good
    g, ms = _synthetic(10)
    r = analyze_shape(g[:12], ms[:12], 2, 4096)
    good = not r["clean"]
    print("truncated sweep refuses:        ", "ok" if good else f"FAIL {r}")
    ok &= good
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json", nargs="*", help="one or more result files (use >= 2 runs)")
    ap.add_argument("--units", type=int)
    ap.add_argument("--expect", action="append", metavar="SHAPE=TOTAL",
                    help="calibration: expected total resident groups for a shape")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json-out", metavar="PATH")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.json:
        ap.error("give a result JSON (or --selftest)")
    docs = []
    for path in a.json:
        with open(path, encoding="utf-8-sig") as fh:
            docs.append(json.load(fh))
    res = analyze(docs, a.units, a.expect)
    print(render(res))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
