"""Can this machine decode faster than its disk?

The curve, and the one comparison everything in this package reduces to. It is
lmsluice code -- it compares a decode rate against *this machine's* storage
rate, and both halves of that are machine measurement -- while every constant
in it belongs to the codec and arrives from `cost_model()`.

    load time, plain file      =  N / disk
    load time, coded archive   =  max(f * N / disk, N / decode)   (overlapped)
    speedup                    =  min(1/f, decode / disk)

## The decode term, in cycles

This used to be written in FP32-FLOP equivalents, and that was the wrong
quantity rather than merely an imprecise one. lmz has since established the
mechanism: the decode loop is a dependent shared-memory load feeding a state
update feeding the next load -- a pointer chase, latency-bound, which does not
scale with a device's FP32 rate. `probe/README.md` had been apologising for the
proxy since it was written; this retires it.

    compute   = resident_lanes x clock / k_cycles_per_byte
    bandwidth = peak_dram x achieved_fraction / (1 + 1/expansion)
    decode    = min(compute, bandwidth)

Every term is now published or measurable. `k_cycles_per_byte`,
`achieved_fraction` and `expansion` come from the codec. `resident_lanes`,
`clock` and `peak_dram` are properties of the device a probe can read.

**The traffic factor is 1 + 1/expansion, not 1.** A decoder reads coded bytes
*and* writes plain ones, so at an expansion of 2.88 it moves 1.347 bytes of
DRAM traffic per decoded byte. Using 1.0 under-counts by a third and was an
error lmz caught. Note it belongs here and **not** in `plan.py`, where the
decode rate is *measured* and the traffic is already inside the number --
applying it there would be the same error in the other direction.

## Which term binds, and why the model may now say

This module used to report both readings and refuse to choose. One measured
point on one card fitted a compute-bound and a bandwidth-bound reading equally
well, and they differ 8x on a 2-CU part, so choosing would have been inventing
the answer.

That is no longer the state, and not because a reading turned out wrong. **Both
are true, and which one binds is a property of the device.** lmz 1.3.0 measured
`k` across a grid sweep rather than at one launch -- a sweep across residency is
exactly what separates a latency-bound cost from a bandwidth ceiling, because
the two predict different slopes as blocks are added. It also corrected `k`
downward, from 230-330 to 217-248, and the direction matters more than the size:
the old interval was fitted where **bandwidth was binding**, which is the regime
that cannot distinguish the two; the new one comes from the rows below
saturation, which is the regime a small device actually lives in.

So the model picks, on the codec's published grounds and not on our inference,
and `Constants.can_pick` is read from the codec rather than assumed. On a
high-bandwidth card the memory system runs out first; on a 2-CU part compute
binds by roughly 3x and the bandwidth reading is not live there at all.

**A tighter interval is not a measurement, and this one is still a projection.**
Everything lmz did was on an RTX 5080. Nobody has run a 2-CU device. Two things
stay unverified and are stated on every row that depends on them: whether decode
stays linear in resident lanes down to 2 CUs or meets a cache, scheduler or
driver wall first, and how a unified-memory part behaves with a CPU contending
for the same bus. The founding caveat survives the narrowing unchanged -- at one
block per unit the smallest AMD iGPU still lands below the CPU decoder.

The refusal machinery stays. Being able to decline a constant in a unit this
curve cannot use predates lmz being able to answer the question, and being right
today is not a reason to stop checking.

**Still runs with no dependencies**: `python3 -m lmsluice.gate`, or the file
directly, on a machine with neither lmsluice installed nor lmz present. That is
load-bearing -- the whole point is that it can be copied onto someone else's
box -- so the codec's constants arrive through an optional argument and there
is a documented fallback carrying its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

# lmz on model weights: 34.7% less disk, so the coded stream is 0.653 of the
# original. A default for the standalone table only; a real plan uses the ratio
# of the archive in front of it.
F_CODED = 0.653

# The unit this curve is written in. A codec publishing `k` in any other unit
# cannot be fed to it without a conversion, and the conversion is not this
# module's to invent -- see `cost_from`.
K_UNIT = "lane-cycles per decoded byte"

# Fallback constants for standalone use, from lmz's published cost model as of
# commit bd3fb42. Labelled, not hidden: when a codec is passed in, these are
# not consulted.
FALLBACK = {
    "k_cycles_per_byte": (217.0, 248.0),
    "expansion": 2.88,
    "achieved_fraction": 0.59,
    "lanes": 8,
    "lut_bytes": 16 << 10,
    "per_group_bytes": 640,
    "block_threads": 192,
    "blocks_at_measurement": (3, 4),
    "k_single_point": False,
}
FALLBACK_PROVENANCE = (
    "lmz.gpu.cost_model('shared') as of lmz 1.3.0: k measured by a grid sweep "
    "at fixed byte traffic, 1 to 1191 blocks over identical input, linear in "
    "resident blocks to within 3% below 84 blocks and bending as bandwidth "
    "takes over, on one RTX 5080. NOT a single-point fit, and lmz's own words "
    "for the derivation are 'over the rows below saturation' -- the "
    "compute-bound regime, which is the one a small device runs in. THE REST "
    "OF THIS SENTENCE IS OUR READING AND NOT LMZ'S: the earlier 230-330 came "
    "from an occupancy sweep at 64-384 threads a block, and on this card that "
    "sweep sits in the bandwidth-bound region -- 418 GB/s measured against a "
    "compute ceiling above 1200 -- so a k divided out of it is bounded by the "
    "memory system rather than by the kernel, which is consistent with the "
    "corrected value being lower. lmz does not characterise its own earlier "
    "number this way. One device either way: no low-FLOP-per-byte part has "
    "been measured.")

# The occupancy the k interval was measured under: lmz publishes it as a range,
# 3 to 4 blocks resident per SM. This is not a footnote, it is what the interval
# means -- the two ends of `k` and the two ends of this range are the same fact
# seen twice, so a device at or above the LOW end sits inside the measurement
# rather than below it, and may claim a bracket. Below it, the row is a ceiling.
#
# k is lane-cycles under *partial latency hiding*. The decode loop is a
# dependent shared-memory chain, and what hides a stall is another block
# resident on the same unit with work to do. lmz measured it on a card that
# holds four. A device that holds fewer hides less of the chain, so its
# effective k moves toward the high end **or past it** -- which is why this is
# an interval at all, and why on such a device the interval is a **floor
# rather than a bracket**.
#
# The consequence for this module: a low-occupancy row is a one-sided
# prediction. "No faster than X" is what it may claim, and it must say so in
# the row, because the row is what gets quoted.
K_MEASURED_AT_BLOCKS_PER_UNIT = 3      # the LOW end of lmz's 3-4


@dataclass(frozen=True)
class Constants:
    """The codec's side of the model. All of it published, none of it ours.

    The shared-memory shape is here rather than at module scope because it is
    the codec's kernel being described, not this machine. It used to be three
    literals up top with a comment saying lmz published them only as prose;
    lmz publishes them as fields now, so they arrive with everything else and
    the literals are a fallback for standalone use.
    """

    k_low: float
    k_high: float
    expansion: float
    achieved_fraction: float
    provenance: str

    # The kernel's shared-memory request, which decides residency.
    lut_bytes: int = FALLBACK["lut_bytes"]
    per_group_bytes: int = FALLBACK["per_group_bytes"]
    lanes_per_group: int = FALLBACK["lanes"]
    block_threads: int = FALLBACK["block_threads"]

    # The occupancy k was measured under, and whether k came from a sweep.
    blocks_at_measurement: int = K_MEASURED_AT_BLOCKS_PER_UNIT
    k_single_point: bool | None = FALLBACK["k_single_point"]

    @property
    def traffic(self) -> float:
        """DRAM bytes moved per decoded byte: the plain out, the coded in."""
        return 1.0 + 1.0 / self.expansion

    @property
    def can_pick(self) -> bool:
        """Whether `k` is strong enough to say which term binds on a device.

        A single-point fit cannot: one measured rate divided by resident lanes
        yields whatever closes the arithmetic, so it fits a compute-bound and a
        bandwidth-bound reading equally and the model must report both. A value
        from a sweep across residency can, because the sweep is what separates
        them. This is the property that changed under the model, and it is read
        from the codec rather than assumed.
        """
        return self.k_single_point is False

    def shmem_per_block(self, threads: int | None = None) -> int:
        t = self.block_threads if threads is None else threads
        return self.lut_bytes + (t // self.lanes_per_group) * self.per_group_bytes

    def blocks_per_unit(self, shmem_per_unit: int, max_threads_per_unit: int,
                        threads: int | None = None) -> int:
        """How many blocks a unit holds. Two limits apply; the smaller wins."""
        t = self.block_threads if threads is None else threads
        return min(shmem_per_unit // self.shmem_per_block(t),
                   max_threads_per_unit // t)


def _published(cost, field: str, default, pair: bool = False):
    """One published scalar or interval, or `default` where the codec is silent.

    Every value the curve needs is asked for the same way, so a codec that
    publishes half of them contributes half and the rest fall back visibly
    rather than the whole record being discarded for one gap.
    """
    got = getattr(cost, field, None)
    if got is None or getattr(got, "low", None) is None:
        return default
    return (got.low, got.high) if pair else got.low


def cost_from(codec_cost=None) -> Constants:
    """Constants from a codec's published cost model, or the fallback.

    **Units are checked, not assumed.** A published `k` in a unit this curve
    does not recognise is reported and *not* used -- refusing loudly beats
    converting hopefully, which is how the FP32 proxy survived as long as it
    did. That refusal predates lmz being able to answer the question and
    survives it: being right today is not a reason to stop checking.
    """
    def build(k_low, k_high, provenance, cost=None):
        """One construction path, so a refusal carries the same shape as a
        success and cannot quietly drop the rest of the model."""
        blocks = _published(cost, "blocks_per_unit_at_measurement",
                            FALLBACK["blocks_at_measurement"], pair=True)
        single = getattr(cost, "k_single_point", None) if cost is not None else None
        return Constants(
            k_low, k_high,
            _published(cost, "expansion", FALLBACK["expansion"]),
            _published(cost, "achieved_fraction", FALLBACK["achieved_fraction"]),
            provenance,
            lut_bytes=int(_published(cost, "shmem_lut_bytes",
                                     FALLBACK["lut_bytes"])),
            per_group_bytes=int(_published(cost, "shmem_per_group_bytes",
                                           FALLBACK["per_group_bytes"])),
            lanes_per_group=int(_published(cost, "lanes", FALLBACK["lanes"])),
            block_threads=int(_published(cost, "block_threads",
                                         FALLBACK["block_threads"])),
            # The low end: k's own high end already prices that occupancy, so a
            # device at or above it sits inside the measurement.
            blocks_at_measurement=int(min(blocks)),
            k_single_point=(FALLBACK["k_single_point"] if single is None
                            else single))

    if codec_cost is None:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        return build(lo, hi, f"fallback: {FALLBACK_PROVENANCE}")
    k = getattr(codec_cost, "k", None)
    if k is None or not k.low:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        return build(lo, hi,
                     f"codec published no k; fallback: {FALLBACK_PROVENANCE}",
                     codec_cost)
    if k.unit != K_UNIT:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        # The refusal is about `k` alone. Whatever else the codec published in
        # units this curve does recognise is still used, and the row says which
        # value was declined -- a units error in one field is not a reason to
        # throw away the others.
        return build(lo, hi,
                     f"REFUSED: the codec publishes k as {k.low}-{k.high} "
                     f"{k.unit}, and this curve is written in {K_UNIT}. "
                     f"They do not convert without further per-device "
                     f"numbers, so the published value is not used. "
                     f"Fallback: {FALLBACK_PROVENANCE}", codec_cost)
    return build(k.low, k.high,
                 k.provenance or "published by the codec", codec_cost)


@dataclass(frozen=True)
class Device:
    """A device, as the cycles model needs it.

    It carries **device facts only** -- units, shared memory per unit, threads
    per unit, clock, bus. Residency is derived from those against the codec's
    published kernel shape at predict time, not stored. An earlier version
    baked the lanes in at import, which meant a device row silently described
    whatever kernel shape was current the day it was written.

    A `None` in any of them is not an error -- the row then carries an interval
    and names what is missing, which is the honest state for a device nobody
    has probed.
    """

    name: str
    gb_s: float                     # peak DRAM bandwidth
    source: str                     # measured where, or derived how
    units: int | None = None        # SMs, WGPs, CUs: whatever owns a shmem pool
    shmem_per_unit: int | None = None
    max_threads_per_unit: int | None = None
    clock_ghz: float | None = None
    gflops: float | None = None     # kept only to show what the old model used
    occupancy_verified: bool = False     # the shared-memory budget was read,
                                         # not assumed from an architecture

    def blocks_per_unit(self, c: Constants) -> int | None:
        if not self.shmem_per_unit or not self.max_threads_per_unit:
            return None
        return c.blocks_per_unit(self.shmem_per_unit, self.max_threads_per_unit)

    def resident_lanes(self, c: Constants) -> int | None:
        """Lanes resident across the device, from its own shared-memory budget.

        **This captures occupancy and not latency hiding**, and the difference
        is the whole of the caveat below. How many lanes are resident is
        arithmetic on a memory budget. Whether a resident lane is *stalled* on
        its dependent load depends on there being another block to switch to,
        and that is inside `k` -- measured across 3 to 4 blocks per unit and
        not adjusted here, because adjusting it would be inventing a latency
        model lmz has not published.
        """
        b = self.blocks_per_unit(c)
        if b is None or not self.units:
            return None
        return self.units * b * c.block_threads

    @property
    def hides_latency_as_measured_needs(self) -> bool:
        """Whether the occupancy was read off the device rather than assumed."""
        return self.occupancy_verified

    def hides_latency_as_measured(self, c: Constants) -> bool:
        """Whether this row may claim a bracket rather than a ceiling.

        Requires the occupancy to be **verified**, not merely derived. The
        derivation needs the device's shared-memory-per-unit budget, and for
        every part here except the one lmz measured on, that budget is an
        architectural assumption rather than a number anybody read off the
        device. On the 2-CU display adapter the assumption alone decides
        whether the row is a bracket or a ceiling -- 128 KiB per WGP gives four
        blocks and a bracket, 64 KiB gives two and a ceiling -- and being
        confidently optimistic on precisely that device class is the failure
        this model exists to avoid. So an unverified budget is treated as
        low-occupancy until `probe/` reads it.
        """
        b = self.blocks_per_unit(c)
        return (self.occupancy_verified and b is not None
                and b >= c.blocks_at_measurement)

    def bandwidth_bound(self, c: Constants) -> float:
        """GB/s of plaintext the memory system can sustain."""
        return self.gb_s * c.achieved_fraction / c.traffic

    def compute_bound(self, c: Constants) -> tuple[float, float] | None:
        """(low, high) GB/s from lanes and clock, or None if either is unknown.

        Lanes times clock, divided by k. **Never a FLOP rate**: lmz establishes
        that k is dependent-load latency rather than an issue rate, by decode
        being linear in resident lanes across an 84x range where an
        issue-limited kernel would have flattened. `gflops` is carried on the
        row to show what the retired model used and is not read here.
        """
        lanes = self.resident_lanes(c)
        if not lanes or not self.clock_ghz:
            return None
        hz = self.clock_ghz * 1e9
        return (lanes * hz / c.k_high / 1e9, lanes * hz / c.k_low / 1e9)

    def predict(self, c: Constants) -> tuple[float, float, str]:
        """(low, high, what bounds it).

        `low` is 0.0 wherever the prediction is one-sided -- either because an
        input is missing, or because this device holds fewer blocks than k was
        measured with and so hides less of the dependent-load chain. In that
        case `high` is a ceiling and the row must be read as "no faster than".
        """
        bw = self.bandwidth_bound(c)
        comp = self.compute_bound(c)
        if comp is None:
            return 0.0, bw, "no lanes/clock: bandwidth ceiling only"
        lo, hi = min(comp[0], bw), min(comp[1], bw)
        if not self.hides_latency_as_measured(c):
            b = self.blocks_per_unit(c)
            why = (f"{b} blocks/unit assumed, not read" if b is not None
                   else "occupancy unknown")
            return 0.0, hi, (f"CEILING: {why}; k was measured at "
                             f"{c.blocks_at_measurement}+ blocks/unit, "
                             f"below which it is a floor")
        if hi >= bw - 1e-9:
            return lo, hi, ("bandwidth" if lo >= bw - 1e-9 else "both, k decides")
        return lo, hi, "compute"

    def margin(self, c: Constants) -> float | None:
        """How far the binding term sits below the other one.

        The quantity that decides whether the two readings are separable on
        *this* device. Above 1 the compute term binds and by how much; below 1
        bandwidth does. A single-point `k` cannot support this and the caller
        must check `c.can_pick` before quoting it.
        """
        comp = self.compute_bound(c)
        if comp is None:
            return None
        bw = self.bandwidth_bound(c)
        return bw / comp[1] if comp[1] else None


# --- Devices ------------------------------------------------------------------
# Rows carry device facts; residency is derived against the codec's kernel
# shape at predict time. A device nobody has probed gets a bandwidth ceiling
# and no floor, which is an honest gap rather than a wide guess dressed as an
# interval. `probe/` is what closes them.

DEVICES = [
    Device("RTX 5080 (dGPU)", 960.0,
           "bus and clock from lmz cost_model() provenance; 84 SMs x 228 KiB "
           "shared, 2048 threads/SM",
           units=84, shmem_per_unit=228 << 10, max_threads_per_unit=2048,
           clock_ghz=2.66, gflops=53_691,
           occupancy_verified=True),   # lmz measured k on this device
    Device("M4 Max (unified)", 546.0,
           "spec bus; lanes and clock not taken -- Apple threadgroup memory "
           "is 32 KiB, close to the kernel's own request, so occupancy is the "
           "question and a guess would be the whole answer",
           gflops=14_336),
    Device("Strix Halo (unified)", 256.0,
           "bus from blueprint §1; lanes and clock not taken", gflops=14_848),
    Device("Radeon 780M (iGPU)", 60.0,
           "spec arithmetic, never run; lanes and clock not taken",
           gflops=4_300),
    Device("Phone SoC GPU (class)", 68.0,
           "class figure, blueprint §1; lanes and clock not taken",
           gflops=2_500),
    Device("Radeon 2-CU (iGPU)", 59.4,
           "bus measured, MEASURED.md; clock 2.2 GHz from that file's FMA "
           "arithmetic; shared memory per WGP is an RDNA2 architectural "
           "figure and IS NOT VERIFIED -- Vulkan on this device reports "
           "maxComputeSharedMemorySize=32768, a per-workgroup cap rather than "
           "the per-unit pool occupancy needs, and the kernel's request fits "
           "it with room to spare. See the closing note.",
           units=1, shmem_per_unit=128 << 10, max_threads_per_unit=2048,
           clock_ghz=2.2, gflops=567),
           # occupancy_verified stays False: 128 KiB per WGP is architectural,
           # not read off this adapter, and it is the difference between four
           # blocks and two.
]

# The other axis, and the one the first version of this gate left out: the
# machine's own storage. Every verdict below is a verdict about a pairing.
STORAGE = [
    ("NVMe Gen5", 12.0, "spec"),
    ("NVMe Gen4 (this box)", 6.34, "measured, vram/probe"),
    ("UFS 4.0 (phone)", 4.0, "spec"),
    ("SATA SSD", 0.55, "spec"),
    ("eMMC / SD", 0.30, "spec"),
]

# Measured decode rates, for comparison (no model involved).
MEASURED_DECODE = {
    "lmz CUDA, shared table, RTX 5080": 418.0,
    "lmz CUDA, per-chunk table, RTX 5080": 111.0,
    "lmz CUDA, through lmsluice, BF16 1 MiB chunks": 101.0,
    "lmz CPU, 4-16 threads": 2.0,
}


def speedup(decode_gb_s: float, disk_gb_s: float, f: float = F_CODED) -> float:
    """How much faster a coded archive opens than a plain file."""
    return min(1.0 / f, decode_gb_s / disk_gb_s)


def verdict(dev: Device, disk_gb_s: float, c: Constants,
            f: float = F_CODED) -> str:
    lo, hi, _why = dev.predict(c)
    if lo > disk_gb_s:
        return "pays"
    if hi <= disk_gb_s:
        return "tax"
    return "measure"


def _table(codec_cost=None) -> None:
    c = cost_from(codec_cost)
    print("Can this machine decode faster than its disk?\n")
    print(f"f = {F_CODED}  ->  ideal speedup 1/f = {1 / F_CODED:.2f}x")
    print(f"k = {c.k_low:.0f}-{c.k_high:.0f} {K_UNIT}")
    print(f"traffic = 1 + 1/{c.expansion} = {c.traffic:.3f} DRAM bytes per decoded byte")
    print(f"    provenance: {c.provenance}\n")

    print("PREDICTIONS, not measurements. Every row is computed from the inputs")
    print("shown; a row missing an input carries an interval and says which.\n")
    print(f"{'device':<24}{'lanes':>8}{'GHz':>6}{'GB/s bus':>10}"
          f"{'compute':>16}{'b/width':>9}   decode GB/s   bound")
    for d in DEVICES:
        comp = d.compute_bound(c)
        bw = d.bandwidth_bound(c)
        lo, hi, why = d.predict(c)
        cs = "  unknown" if comp is None else f"{comp[0]:.0f} - {comp[1]:.0f}"
        band = (f"<= {hi:.1f}" if lo <= 0.0
                else (f"{lo:.1f} - {hi:.1f}" if hi - lo > 0.05 else f"{hi:.1f}"))
        print(f"{d.name:<24}{(d.resident_lanes(c) or 0):>8}"
              f"{(d.clock_ghz or 0):>6.2f}{d.gb_s:>10.1f}{cs:>16}{bw:>9.0f}"
              f"   {band:>12}   {why}")
    print()

    # --- Which term binds, and whether we are entitled to say -----------------
    if c.can_pick:
        print("WHICH TERM BINDS IS A PROPERTY OF THE DEVICE, and k is now strong")
        print("enough to say so. It came from a sweep across residency rather than")
        print("from one launch, and a sweep is what separates a compute-bound from")
        print("a bandwidth-bound reading -- one point fits both.\n")
        for d in DEVICES:
            m = d.margin(c)
            if m is None:
                continue
            if m > 1:
                print(f"  {d.name:<24} compute binds, {m:.1f}x below its own "
                      f"bandwidth ceiling")
            else:
                print(f"  {d.name:<24} bandwidth binds, {1 / m:.1f}x below its "
                      f"compute ceiling")
        print()
        print("  So the two readings the old model could not separate are both true,")
        print("  and they bind on different devices. On a high-bandwidth card the")
        print("  memory system runs out first; on a small integrated part compute")
        print("  does, by a wide enough margin that the bandwidth reading is not")
        print("  live there at all.\n")
    else:
        print("k is a single-point fit, so this model reports both readings and")
        print("declines to pick between them. One measured rate divided by resident")
        print("lanes yields whatever closes the arithmetic.\n")

    print("A row reading '<= X' is one-sided and must be quoted that way.")
    print(f"k was measured under partial latency hiding at "
          f"{c.blocks_at_measurement}+ blocks per unit; a device holding")
    print("fewer hides less of the dependent-load chain, so its effective k")
    print("moves toward the high end or past it and the interval becomes a")
    print("floor. Resident lanes captures the occupancy part of that and not")
    print("the hiding part -- which is why k stays an interval and why these")
    print("rows may claim a ceiling and not a bracket.\n")

    print("Does the archive pay, and against which disk?")
    print(f"{'device':<24}", end="")
    for name, _disk, _src in STORAGE:
        print(f"{name.split(' (')[0]:>14}", end="")
    print()
    print(f"{'':<24}", end="")
    for _name, disk, _src in STORAGE:
        print(f"{format(disk, '.2f') + ' GB/s':>14}", end="")
    print()
    for d in DEVICES:
        lo, _hi, _why = d.predict(c)
        print(f"{d.name:<24}", end="")
        for _name, disk, _src in STORAGE:
            v = verdict(d, disk, c)
            sp = speedup(lo, disk)
            cell = v if sp >= 1 / F_CODED - 1e-9 else f"{v} {sp:.2f}x"
            print(f"{cell:>14}", end="")
        print()
    print()
    print("Bare 'pays' is the saturated case: the decoder beats the disk by more")
    print("than 1/f, so the archive's ratio has become load speed and nothing")
    print("faster helps. 'measure' means the interval straddles that disk. One")
    print("device changes verdict with the disk beside it, which is why a gate")
    print("calibrated against one machine's NVMe is not a gate.\n")

    print("Measured decode rates, for comparison (no model involved)")
    for name, v in MEASURED_DECODE.items():
        print(f"  {name:<46}{v:>8.1f} GB/s")
    print()
    ref = DEVICES[0]
    lo, hi, _ = ref.predict(c)
    print(f"Check: the model predicts {ref.name} at {lo:.0f}-{hi:.0f} GB/s against")
    print(f"{MEASURED_DECODE['lmz CUDA, shared table, RTX 5080']:.0f} measured. That is the")
    print("one point the constants were taken on, so agreement is a consistency")
    print("check and not a validation -- the model has not yet been tested against")
    print("a device it was not fitted on.\n")

    small = DEVICES[-1]
    lo, hi, _ = small.predict(c)
    cpu = MEASURED_DECODE["lmz CPU, 4-16 threads"]
    print("And the founding claim, stated as weakly as the evidence allows:")
    print(f"the smallest integrated GPU AMD ships is predicted at no more than")
    print(f"{hi:.1f} GB/s against the CPU decoder's {cpu:.1f} -- so at most "
          f"{hi / cpu:.1f}x, and")
    print("with no floor, because its occupancy is assumed rather than read.\n")

    print("THE NARROWING IS NOT A MEASUREMENT. k went from 230-330 to 217-248")
    print("because lmz swept its own card across grid sizes. NOBODY HAS RUN A")
    print("2-CU DEVICE. Two things stay unverified, and they are worth more to")
    print("a reader than the interval that just closed:")
    print()
    print("  1. Whether decode stays linear in resident lanes all the way down")
    print("     to 2 CUs, or meets a cache, scheduler or driver wall first. The")
    print("     linearity was established over an 84x range ON ONE CARD, none of")
    print("     it near this size.")
    print("  2. How a unified-memory part behaves when the CPU is contending for")
    print("     the same bus. Every bandwidth figure here assumes the decoder has")
    print("     the memory system to itself, which on an iGPU it does not.")
    print()
    print("AND THE CEILING ITSELF IS NOT SAFE. The lanes above assume this")
    print("device holds four blocks per unit. Vulkan reports its per-workgroup")
    print("shared-memory cap as 32768 B against the kernel's request -- so it")
    print("fits, and how many such blocks a unit holds is not something Vulkan")
    small_1 = Device(small.name, small.gb_s, small.source, units=1,
                     shmem_per_unit=c.shmem_per_block(), 
                     max_threads_per_unit=small.max_threads_per_unit,
                     clock_ghz=small.clock_ghz)
    one = small_1.compute_bound(c)
    print("exposes. At one block per unit the row becomes "
          f"{one[0]:.1f}-{one[1]:.1f} GB/s, which is")
    print(f"BELOW the CPU decoder's {cpu:.1f}. The plausible range for this device")
    print("therefore spans the CPU decoder, and nothing here establishes that the")
    print("iGPU wins. That is a question for the measurement, not for the wording.")



def main() -> None:
    """Standalone: no codec, no imports, the published constants."""
    _table(None)


if __name__ == "__main__":
    main()
