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

## What this buys

The old model gave the 2-CU display iGPU an interval of 4 - 36 GB/s, because
one measured point fitted a compute-bound and a bandwidth-bound reading that
differ 8x on a small device. In cycles the same device predicts a far narrower
band, because `k` is now measured on the kernel rather than inferred from one
point. The rows stop being 8x wide -- and must not become false precision
instead, so every row still says it is a prediction and names the inputs it
came from, and any row missing an input carries an interval naming which.

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
    "k_cycles_per_byte": (230.0, 330.0),
    "expansion": 2.88,
    "achieved_fraction": 0.59,
    "lanes": 8,
}
FALLBACK_PROVENANCE = (
    "lmz.gpu.cost_model() as of bd3fb42: k measured by an occupancy sweep at "
    "fixed byte traffic (64..384 threads/block, two runs within 1%) on one "
    "RTX 5080. One device: a low-FLOP-per-byte part would tighten the interval "
    "and has not been measured.")

# The occupancy the k interval was measured under: four blocks resident per
# SM. This is not a footnote, it is what the interval means.
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
K_MEASURED_AT_BLOCKS_PER_UNIT = 4


@dataclass(frozen=True)
class Constants:
    """The codec's side of the model. All of it published, none of it ours."""

    k_low: float
    k_high: float
    expansion: float
    achieved_fraction: float
    provenance: str

    @property
    def traffic(self) -> float:
        """DRAM bytes moved per decoded byte: the plain out, the coded in."""
        return 1.0 + 1.0 / self.expansion


def cost_from(codec_cost=None) -> Constants:
    """Constants from a codec's published cost model, or the fallback.

    **Units are checked, not assumed.** A published `k` in a unit this curve
    does not recognise is reported and *not* used -- refusing loudly beats
    converting hopefully, which is how the FP32 proxy survived as long as it
    did.
    """
    if codec_cost is None:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        return Constants(lo, hi, FALLBACK["expansion"],
                         FALLBACK["achieved_fraction"],
                         f"fallback: {FALLBACK_PROVENANCE}")
    k = getattr(codec_cost, "k", None)
    if k is None or not k.low:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        return Constants(lo, hi, FALLBACK["expansion"],
                         FALLBACK["achieved_fraction"],
                         f"codec published no k; fallback: {FALLBACK_PROVENANCE}")
    if k.unit != K_UNIT:
        lo, hi = FALLBACK["k_cycles_per_byte"]
        return Constants(lo, hi, FALLBACK["expansion"],
                         FALLBACK["achieved_fraction"],
                         f"REFUSED: the codec publishes k as {k.low}-{k.high} "
                         f"{k.unit}, and this curve is written in {K_UNIT}. "
                         f"They do not convert without further per-device "
                         f"numbers, so the published value is not used. "
                         f"Fallback: {FALLBACK_PROVENANCE}")
    return Constants(k.low, k.high, FALLBACK["expansion"],
                     FALLBACK["achieved_fraction"],
                     k.provenance or "published by the codec")


@dataclass(frozen=True)
class Device:
    """A device, as the cycles model needs it.

    `resident_lanes` and `clock_ghz` are what the model added; `gb_s` it always
    needed. A `None` in any of them is not an error -- the row then carries an
    interval and names what is missing, which is the honest state for a device
    nobody has probed.
    """

    name: str
    gb_s: float                     # peak DRAM bandwidth
    source: str                     # measured where, or derived how
    resident_lanes: int | None = None
    clock_ghz: float | None = None
    gflops: float | None = None     # kept only to show what the old model used
    blocks_per_unit: int | None = None   # occupancy, against k's measurement
    occupancy_verified: bool = False     # the shared-memory budget was read,
                                         # not assumed from an architecture

    @property
    def hides_latency_as_measured(self) -> bool:
        """Whether this row may claim a bracket rather than a ceiling.

        Requires the occupancy to be **verified**, not merely derived. The
        derivation needs the device's shared-memory-per-unit budget, and for
        every part here except the one lmz measured on, that budget is an
        architectural assumption rather than a number anybody read off the
        device. On the 2-CU display adapter the assumption alone decides
        whether the row is a bracket or a ceiling -- 128 KiB per WGP gives
        four blocks and a bracket, 64 KiB gives two and a ceiling -- and being
        confidently optimistic on precisely that device class is the failure
        this model exists to avoid. So an unverified budget is treated as
        low-occupancy until `probe/` reads it.
        """
        return (self.occupancy_verified
                and self.blocks_per_unit is not None
                and self.blocks_per_unit >= K_MEASURED_AT_BLOCKS_PER_UNIT)

    def bandwidth_bound(self, c: Constants) -> float:
        """GB/s of plaintext the memory system can sustain."""
        return self.gb_s * c.achieved_fraction / c.traffic

    def compute_bound(self, c: Constants) -> tuple[float, float] | None:
        """(low, high) GB/s from lanes and clock, or None if either is unknown."""
        if not self.resident_lanes or not self.clock_ghz:
            return None
        hz = self.clock_ghz * 1e9
        return (self.resident_lanes * hz / c.k_high / 1e9,
                self.resident_lanes * hz / c.k_low / 1e9)

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
        if not self.hides_latency_as_measured:
            why = (f"{self.blocks_per_unit} blocks/unit assumed, not read"
                   if self.blocks_per_unit is not None
                   else "occupancy unknown")
            return 0.0, hi, (f"CEILING: {why}; k was measured at "
                             f"{K_MEASURED_AT_BLOCKS_PER_UNIT} blocks/unit, "
                             f"below which it is a floor")
        if hi >= bw - 1e-9:
            return lo, hi, ("bandwidth" if lo >= bw - 1e-9 else "both, k decides")
        return lo, hi, "compute"


# --- Resident lanes -------------------------------------------------------
#
# The one derived quantity, and it is derived here rather than baked so it
# re-evaluates on a device nobody has run. lmz publishes the shape of its
# kernel's shared-memory request in the provenance of `cost_model()`: a
# 16 KiB lookup table plus 640 bytes per group of 8 lanes, and the kernel is
# compute-bound below 192 threads a block.
#
# The published shape is prose, not a field -- see the report; reading it as
# structure would mean parsing English, so it is restated here with a pointer
# back and will move into the constants when the codec publishes it properly.

LUT_BYTES = 16 << 10           # per block, whatever the block size
PER_GROUP_BYTES = 640          # per group of `lanes` threads
LANES_PER_GROUP = 8
BLOCK_THREADS = 192            # cost_model()["bound"]["compute_below_threads"]


def shmem_per_block(threads: int = BLOCK_THREADS) -> int:
    return LUT_BYTES + (threads // LANES_PER_GROUP) * PER_GROUP_BYTES


def blocks_per_unit(shmem_per_unit: int, max_threads_per_unit: int,
                    threads: int = BLOCK_THREADS) -> int:
    """How many blocks a unit holds. Two limits apply; the smaller wins."""
    return min(shmem_per_unit // shmem_per_block(threads),
               max_threads_per_unit // threads)


def resident(units: int, shmem_per_unit: int, max_threads_per_unit: int,
             threads: int = BLOCK_THREADS) -> int:
    """Lanes resident across the device, from its own shared-memory budget.

    `units` is SMs, CUs or whatever the device calls the thing that owns a
    shared-memory pool.

    **This captures occupancy and not latency hiding**, and the difference is
    the whole of the caveat above. How many lanes are resident is arithmetic
    on a memory budget, and it is what this returns. Whether a resident lane
    is *stalled* on its dependent load depends on there being another block to
    switch to, and that is inside `k` -- measured at four blocks per unit and
    not adjusted here, because adjusting it would be inventing a latency model
    lmz has not published. So a device below that occupancy is predicted
    optimistically by exactly the amount the chain goes unhidden, and the row
    says "no faster than" rather than pretending to bracket it.
    """
    return units * blocks_per_unit(shmem_per_unit, max_threads_per_unit,
                                   threads) * threads


# --- Devices ------------------------------------------------------------------
# Two rows carry the numbers the cycles model needs, because this box measured
# them. The rest carry `None` and say so: a device nobody has probed gets a
# bandwidth ceiling and no floor, which is an honest gap rather than a wide
# guess dressed as an interval. `probe/` is what closes them.

DEVICES = [
    Device("RTX 5080 (dGPU)", 960.0,
           "bus and clock from lmz cost_model() provenance; lanes derived "
           "from 84 SMs x 228 KiB shared, 2048 threads/SM",
           resident_lanes=resident(84, 228 << 10, 2048), clock_ghz=2.66,
           gflops=53_691, blocks_per_unit=blocks_per_unit(228 << 10, 2048),
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
           "arithmetic; lanes shown assume 4 blocks/unit and ARE NOT VERIFIED "
           "-- Vulkan on this device reports maxComputeSharedMemorySize=32768, "
           "which is a per-workgroup cap rather than the per-unit pool "
           "occupancy needs, and the kernel's 31744 B request fits it with "
           "1024 B to spare. At 1 block/unit the row would be 1.3-1.8 GB/s, "
           "below the CPU decoder. See the closing note.",
           resident_lanes=resident(1, 128 << 10, 2048), clock_ghz=2.2,
           gflops=567, blocks_per_unit=blocks_per_unit(128 << 10, 2048)),
           # occupancy_verified stays False: 128 KiB per WGP is an RDNA2
           # architectural figure, not a number read off this adapter, and it
           # is the difference between four blocks and two.
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
        print(f"{d.name:<24}{(d.resident_lanes or 0):>8}"
              f"{(d.clock_ghz or 0):>6.2f}{d.gb_s:>10.1f}{cs:>16}{bw:>9.0f}"
              f"   {band:>12}   {why}")
    print()
    print("A row reading '<= X' is one-sided and must be quoted that way.")
    print(f"k was measured under partial latency hiding at "
          f"{K_MEASURED_AT_BLOCKS_PER_UNIT} blocks per unit; a device holding")
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
    print("with no floor, because its occupancy is assumed rather than read.")
    print("That is weaker than the old 4-36 GB/s interval looked and stronger")
    print("than it actually was: the old floor was an artefact of reading one")
    print("measured point as compute-bound.\n")
    print("AND THE CEILING ITSELF IS NOT SAFE. The lanes above assume this")
    print("device holds four blocks per unit. Vulkan reports its per-workgroup")
    print("shared-memory cap as 32768 B against a 31744 B kernel request -- so")
    print("the kernel fits by 1024 B, and how many such blocks a unit holds is")
    print("not something Vulkan exposes. At one block per unit the row becomes")
    print("1.3-1.8 GB/s, which is BELOW the CPU decoder's 2.0. The plausible")
    print("range for this device therefore spans the CPU decoder, and nothing")
    print("here establishes that the iGPU wins. That is a question for the")
    print("measurement, not for the wording.")


def main() -> None:
    """Standalone: no codec, no imports, the published constants."""
    _table(None)


if __name__ == "__main__":
    main()
