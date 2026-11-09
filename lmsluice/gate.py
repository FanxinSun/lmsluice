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
cuda's interval was taken on an RTX 5080. lmz's Metal kernel -- a different
kernel, with its own constants in `FALLBACK_METAL` -- has since been measured on
an Apple M4, a unified-memory part: compute-bound at ~28% of DRAM, k 269-313 at
the kernel's clock, linear in resident threadgroups to 4 per core with k rising
about 16% as they share a core. On that part a CPU streaming 71-81 GB/s on the
same bus slowed the decoder 17-29% although DRAM was not what bound it -- the
first measurement of the second caveat below, on one device. Nobody has run a
2-CU device, so both caveats stay open AT THAT SCALE and are stated on every row
that depends on them: whether decode stays linear in resident lanes down to 2
CUs or meets a cache, scheduler or driver wall first, and how much a CPU on the
same bus costs there. The founding caveat survives unchanged -- at one block per
unit the smallest AMD iGPU still lands below the CPU decoder.

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

from dataclasses import dataclass, replace

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
    "takes over, on one RTX 5080. NOT a single-point fit. Why it moved is "
    "lmz's own account in cost_model()['provenance']['supersedes'] as of lmz "
    "e8afb10, and it is neither 'the old value was wrong' nor 'the old sweep "
    "was bandwidth-bound': the old sweep was MIXED, and fitting one constant "
    "across rows governed by different resources is the defect. At 64 threads "
    "a block the kernel is compute-bound (245.7 GB/s against a 264 ceiling); "
    "from 96 threads up it is bandwidth-bound (417.3 against 791 at 384). One "
    "compute row and six memory rows averaged into one interval -- which is "
    "why the old low end was nearly right, carried by the single good row, "
    "and its high end was not. (217, 248) is the same quantity taken only "
    "where compute binds: a refinement of a badly-conditioned fit, not a "
    "contradiction. Nothing about the kernel changed. One device either way: "
    "no low-FLOP-per-byte part has been measured.")

# The same, for lmz's Metal shared-table kernel: a DIFFERENT kernel from CUDA
# k_shared (128 threads, direct refill reads, a 19,456 B compiled footprint),
# so it gets its own constants rather than CUDA's. Measured on an Apple M4,
# where CUDA's constants would predict it 26-44% too fast.
FALLBACK_METAL = {
    "k_cycles_per_byte": (269.4, 313.1),
    "expansion": 2.975,
    # Never reached: on the M4 the kernel stops at its residency limit with
    # DRAM at ~28% of spec, so no saturation fraction exists. CUDA's stands in
    # (visibly, in the provenance); bandwidth does not bind there by ~2x.
    "achieved_fraction": None,
    "lanes": 8,
    "lut_bytes": 17_408,         # LUT 16,384 + starts 1,024: fixed per threadgroup
    "per_group_bytes": 128,      # the output stage; the unused ring compiles out
    "block_threads": 128,
    "blocks_at_measurement": (1, 4),
    "k_single_point": False,
}
FALLBACK_METAL_PROVENANCE = (
    "lmz.gpu.cost_model('shared', backend='metal'): the Metal shared-table "
    "kernel (lmz_decode_plane_direct) measured on an Apple M4, 10 GPU cores, "
    "by a grid sweep at fixed byte traffic over 936.6 MB of real Qwen3 BF16 "
    "exponent planes, byte-identical at every size. k at the KERNEL's clock "
    "(IOReport P15 = 1578 MHz, whose residency matches the summed kernel time "
    "in every row): 269.4-276.5 at <=1 threadgroup per core, 288 at 2, 297-298 "
    "at 3, 312-313 at 4. On this device k RISES with co-residency -- groups "
    "contend rather than only hiding latency -- so a device holding fewer than "
    "4 per core sits toward the low end. Compute-bound at every occupancy; "
    "desktop GPU clients were present in most rows, so it is conditional on "
    "that background. Record: lmz docs/evaluations/m4-metal-decoder-remeasure.md")

# The backend whose kernel a device runs. Constants describe one kernel, and
# lmz's kernels differ in shape by backend, so a row is predicted with its own
# backend's constants. Where no kernel of that backend has been measured
# (vulkan, today), cuda's are BORROWED and the row says so: a projection, not a
# description of a kernel that never ran there.
BACKENDS = ("cuda", "metal", "vulkan")
FALLBACKS = {
    "cuda": (FALLBACK, FALLBACK_PROVENANCE),
    "metal": (FALLBACK_METAL, FALLBACK_METAL_PROVENANCE),
}

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

    # Whose kernel these constants describe, and -- when no kernel of that
    # backend has been measured -- whose constants were borrowed to stand in.
    backend: str = "cuda"
    borrowed_from: str | None = None

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

    def blocks_per_unit(self, pool: int, cap: int, max_threads_per_unit: int,
                        threads: int | None = None) -> int:
        """lmz's published `residency_formula`, applied.

            blocks = floor(pool / shm) if shm <= cap else 0

        **The cap and the pool are different quantities and both are needed.**
        The cap says whether a block of this size is *legal* (CUDA's
        `sharedMemPerBlockOptin`); the pool says how many *fit* (CUDA's
        `sharedMemPerMultiprocessor`). On discrete NVIDIA parts they agree
        within a kilobyte -- this card reports 101376 and 102400 -- which is
        why conflating them is invisible there. They diverge by up to 3x on
        integrated parts, which is exactly the device class this model exists
        to serve, so the conflation fails only where it matters.
        """
        t = self.block_threads if threads is None else threads
        if self.shmem_per_block(t) > cap:
            return 0                       # the block cannot be launched at all
        return min(pool // self.shmem_per_block(t), max_threads_per_unit // t)


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


def cost_from(codec_cost=None, backend: str = "cuda") -> Constants:
    """Constants from a codec's published cost model, or the fallback.

    `backend` names whose kernel the constants describe, and `codec_cost`, if
    given, must be the codec's record for that backend's kernel. The default
    is cuda, which is exactly what this function returned before backends
    existed. A backend with no measured kernel gets cuda's constants marked
    `borrowed_from`, and every row predicted with them says so.

    **Units are checked, not assumed.** A published `k` in a unit this curve
    does not recognise is reported and *not* used -- refusing loudly beats
    converting hopefully, which is how the FP32 proxy survived as long as it
    did. That refusal predates lmz being able to answer the question and
    survives it: being right today is not a reason to stop checking.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
    borrowed = None
    if backend in FALLBACKS:
        fb, fb_prov = FALLBACKS[backend]
    else:
        fb, fb_prov = FALLBACKS["cuda"]
        if codec_cost is None:
            borrowed = "cuda"
            fb_prov = (f"BORROWED: no {backend} kernel has been measured, so "
                       f"these are cuda's constants used as a projection. "
                       f"{fb_prov}")
    af_default = fb["achieved_fraction"]
    if af_default is None:
        af_default = FALLBACK["achieved_fraction"]
        fb_prov = (f"{fb_prov}. achieved_fraction is not published for this "
                   f"kernel (never reached), so cuda's {af_default} stands in.")

    def build(k_low, k_high, provenance, cost=None):
        """One construction path, so a refusal carries the same shape as a
        success and cannot quietly drop the rest of the model."""
        blocks = _published(cost, "blocks_per_unit_at_measurement",
                            fb["blocks_at_measurement"], pair=True)
        single = getattr(cost, "k_single_point", None) if cost is not None else None
        af = _published(cost, "achieved_fraction", af_default)
        if (cost is not None and af == af_default
                and fb["achieved_fraction"] is None):
            provenance = (f"{provenance}; achieved_fraction not published, "
                          f"cuda's {af_default} stands in")
        return Constants(
            k_low, k_high,
            _published(cost, "expansion", fb["expansion"]),
            af,
            provenance,
            lut_bytes=int(_published(cost, "shmem_lut_bytes",
                                     fb["lut_bytes"])),
            per_group_bytes=int(_published(cost, "shmem_per_group_bytes",
                                           fb["per_group_bytes"])),
            lanes_per_group=int(_published(cost, "lanes", fb["lanes"])),
            block_threads=int(_published(cost, "block_threads",
                                         fb["block_threads"])),
            # The low end: k's own high end already prices that occupancy, so a
            # device at or above it sits inside the measurement.
            blocks_at_measurement=int(min(blocks)),
            k_single_point=(fb["k_single_point"] if single is None
                            else single),
            backend=backend, borrowed_from=borrowed)

    if codec_cost is None:
        lo, hi = fb["k_cycles_per_byte"]
        return build(lo, hi, f"fallback: {fb_prov}")
    k = getattr(codec_cost, "k", None)
    if k is None or not k.low:
        lo, hi = fb["k_cycles_per_byte"]
        return build(lo, hi,
                     f"codec published no k; fallback: {fb_prov}",
                     codec_cost)
    if k.unit != K_UNIT:
        lo, hi = fb["k_cycles_per_byte"]
        # The refusal is about `k` alone. Whatever else the codec published in
        # units this curve does recognise is still used, and the row says which
        # value was declined -- a units error in one field is not a reason to
        # throw away the others.
        return build(lo, hi,
                     f"REFUSED: the codec publishes k as {k.low}-{k.high} "
                     f"{k.unit}, and this curve is written in {K_UNIT}. "
                     f"They do not convert without further per-device "
                     f"numbers, so the published value is not used. "
                     f"Fallback: {fb_prov}", codec_cost)
    return build(k.low, k.high,
                 k.provenance or "published by the codec", codec_cost)


def constants_by_backend(cuda_cost=None, metal_cost=None) -> dict:
    """One Constants per backend: each from its codec record, or its fallback."""
    return {"cuda": cost_from(cuda_cost, "cuda"),
            "metal": cost_from(metal_cost, "metal"),
            "vulkan": cost_from(None, "vulkan")}


def constants_for(dev, by_backend: dict) -> Constants:
    """The constants a device row is predicted with: its own backend's.

    A backend missing from `by_backend` gets cuda's, marked borrowed -- never
    silently, because a row predicted with another kernel's cost is a
    projection and is quoted as a measurement otherwise.
    """
    c = by_backend.get(dev.backend)
    if c is not None:
        return c
    base = by_backend["cuda"]
    return replace(base, borrowed_from=base.backend,
                   provenance=(f"BORROWED for a {dev.backend} device: "
                               f"{base.provenance}"))


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
    shmem_pool_per_unit: int | None = None   # how many blocks FIT
    shmem_cap_per_block: int | None = None   # whether one block is LEGAL
    max_threads_per_unit: int | None = None
    clock_ghz: float | None = None
    gflops: float | None = None     # kept only to show what the old model used
    occupancy_verified: bool = False     # the shared-memory budget was read,
                                         # not assumed from an architecture
    # Whose decoder kernel this device runs, which decides whose constants it
    # is predicted with. Keyword-only in practice: it is last, with a default.
    backend: str = "cuda"

    def blocks_per_unit(self, c: Constants) -> int | None:
        """None when the pool has not been obtained -- which is not the same as
        a small pool, and must not collapse to one."""
        if (not self.shmem_pool_per_unit or not self.shmem_cap_per_block
                or not self.max_threads_per_unit):
            return None
        return c.blocks_per_unit(self.shmem_pool_per_unit,
                                 self.shmem_cap_per_block,
                                 self.max_threads_per_unit)

    def residency_binds(self, c: Constants) -> str | None:
        """Which of the two residency limits is the binding one, or None.

        Worth reporting for the same reason `transport.Report.limited_by` is:
        the answer differs between devices, and a model that only prints the
        result cannot be checked against a machine nobody here owns. It also
        makes a live hazard visible -- the shared-memory term is large mainly
        because of a fixed 16 KiB lookup table, so narrowing that table (which
        both this package and lmz have been pointing at) shrinks exactly the
        margin by which shared memory currently wins, and the thread term
        starts to bind on parts where it never has.
        """
        if (not self.shmem_pool_per_unit or not self.max_threads_per_unit):
            return None
        shm = self.shmem_pool_per_unit // c.shmem_per_block()
        thr = self.max_threads_per_unit // c.block_threads
        return "shared memory" if shm <= thr else "threads per unit"

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
        lo, hi, why = self._predict(c)
        return lo, hi, why + self._constants_note(c)

    def _constants_note(self, c: Constants) -> str:
        """Say when the constants are not this device's own kernel's.

        A row predicted with another backend's cost is a projection. It reads
        exactly like a prediction from its own kernel unless it says so, and the
        row is what gets quoted.
        """
        if c.borrowed_from:
            return (f" [k BORROWED from {c.borrowed_from}: no {self.backend} "
                    f"kernel has been measured]")
        if c.backend != self.backend:
            return (f" [{c.backend} kernel's constants on a {self.backend} "
                    f"device: a projection]")
        return ""

    def _predict(self, c: Constants) -> tuple[float, float, str]:
        bw = self.bandwidth_bound(c)
        comp = self.compute_bound(c)
        if comp is None:
            return 0.0, bw, "no lanes/clock: bandwidth ceiling only"
        lo, hi = min(comp[0], bw), min(comp[1], bw)
        if not self.hides_latency_as_measured(c):
            b = self.blocks_per_unit(c)
            # Two different reasons land here, and the row must say which: an
            # occupancy nobody read off the device, or one that WAS read and
            # simply sits below the occupancy k was measured at. Calling the
            # second "assumed" misreports a measurement as a guess.
            if b is None:
                why = "occupancy unknown"
            elif not self.occupancy_verified:
                why = f"{b} blocks/unit assumed, not read"
            else:
                why = f"{b} blocks/unit, read off the device"
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
    # Every figure below was read off the device with cuDeviceGetAttribute,
    # not taken from a spec sheet or from lmz. An earlier version used 228 KiB
    # as the pool and 2048 threads/SM; the device reports 102400 and 1536.
    Device("RTX 5080 (dGPU)", 960.0,
           "queried: 84 SMs, pool 102400 B/SM "
           "(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR), cap 101376 B/block "
           "(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN), 1536 threads/SM, "
           "2.617 GHz clock rate; bus from lmz cost_model() provenance",
           units=84, shmem_pool_per_unit=102_400, shmem_cap_per_block=101_376,
           max_threads_per_unit=1536, clock_ghz=2.66, gflops=53_691,
           occupancy_verified=True, backend="cuda"),   # lmz measured k on this device
    # Every figure below was read or measured on the device by
    # probe/metal-bench.js (results/probe_m4/metal-bench-20261109-125645.json,
    # docs/evaluations/m4-metal-probe.md) unless marked otherwise.
    Device("Apple M4 (unified)", 120.0,
           "MEASURED on a Mac mini (Mac16,10), macOS 15.4.1, 2026-11-09 by "
           "probe/metal-bench.js: 10 GPU cores (ioreg gpu-core-count); cap "
           "32768 B (maxThreadgroupMemoryLength); pool 77,824 <= pool < 83,968 "
           "B from occupancy staircases (4 TGs/core at 19,456 B, 3 at 20,992 "
           "and 23,552, 2 at 31,744-32,768), the 4/core decoder shape "
           "independently matched by lmz's decoder grid sweep; clock 1.578 GHz "
           "= IOReport state P15, the state sustained FMA and decode runs sit "
           "in. Threads per core: probed five times by latency-bound "
           "staircase and NOT reproducible (704-1,280 per core, best single "
           "cap 992-1,280 by run), so 1024 is the API lower bound (one "
           "1024-thread threadgroup is legal per core); it binds nothing "
           "derived here. Bus 120 GB/s is spec; 102 GB/s streaming read was "
           "measured. With CUDA k_shared's constants this row is a CEILING (2 "
           "blocks/unit, below the 3 k was measured at); lmz's own Metal kernel "
           "has a different shape and k -- see MEASURED_DECODE",
           units=10, shmem_pool_per_unit=77_824, shmem_cap_per_block=32_768,
           max_threads_per_unit=1024, clock_ghz=1.578, gflops=3_663,
           occupancy_verified=True, backend="metal"),
    Device("M4 Max (unified)", 546.0,
           "spec bus; lanes and clock not taken -- Apple threadgroup memory "
           "is 32 KiB, close to the kernel's own request, so occupancy is the "
           "question and a guess would be the whole answer. The Apple M4 row "
           "above measures per-core facts of the same GPU generation; carrying "
           "them over to the Max would be a projection, so they are not",
           gflops=14_336, backend="metal"),
    Device("Strix Halo (unified)", 256.0,
           "bus from blueprint §1; lanes and clock not taken", gflops=14_848, backend="vulkan"),
    Device("Radeon 780M (iGPU)", 60.0,
           "spec arithmetic, never run; lanes and clock not taken",
           gflops=4_300, backend="vulkan"),
    Device("Phone SoC GPU (class)", 68.0,
           "class figure, blueprint §1; lanes and clock not taken",
           gflops=2_500, backend="vulkan"),
    Device("Radeon 2-CU (iGPU)", 59.4,
           "QUERIED from the device through Vulkan on the Windows side "
           "(the adapter is not reachable from Vulkan inside WSL2 -- "
           "enumeration there returns llvmpipe alone): 'AMD Radeon(TM) "
           "Graphics', integrated, cap 32768 B "
           "(maxComputeSharedMemorySize), 2 compute units "
           "(VK_AMD_shader_core_properties: 1 engine x 1 array x 2 CUs, and "
           "shader_core_properties2 activeComputeUnitCount=2), 2048 threads "
           "per CU (16 wavefronts x 2 SIMD x 64 wide). Bus measured, "
           "MEASURED.md; clock 2.2 GHz from that file's FMA arithmetic. "
           "THE POOL IS UNOBTAINED -- see the closing note.",
           units=2, shmem_cap_per_block=32_768, max_threads_per_unit=2048,
           clock_ghz=2.2, gflops=567, backend="vulkan"),
           # shmem_pool_per_unit is deliberately absent, and it is not for want
           # of looking: core Vulkan reports no per-unit pool, and NEITHER
           # VK_AMD_shader_core_properties NOR shader_core_properties2 -- both
           # present on this adapter and both queried -- has an LDS field at
           # all. Substituting the cap for it, or an architectural figure for
           # it, is an assumption wearing a number, and this model's whole job
           # is to not do that.
]

# The other axis, and the one the first version of this gate left out: the
# machine's own storage. Every verdict below is a verdict about a pairing.
STORAGE = [
    ("NVMe Gen5", 12.0, "spec"),
    ("NVMe Gen4 (this box)", 6.34, "measured, vram/probe"),
    ("UFS 4.0 (phone)", 4.0, "spec"),
    ("M4 NVMe (Mac mini internal)", 3.12,
     "measured cold, F_NOCACHE, lmsluice profile; 3.10 GB/s parallel read"),
    ("SATA SSD", 0.55, "spec"),
    ("M4 USB SSD (exFAT)", 0.373, "measured cold, F_NOCACHE, lmsluice profile"),
    ("eMMC / SD", 0.30, "spec"),
]

# Measured decode rates, for comparison (no model involved).
MEASURED_DECODE = {
    "lmz CUDA, shared table, RTX 5080": 418.0,
    "lmz CUDA, per-chunk table, RTX 5080": 111.0,
    "lmz CUDA, through lmsluice, BF16 1 MiB chunks": 101.0,
    "lmz CPU, 4-16 threads": 2.0,
    # lmz/docs/evaluations/m4-metal-decoder-remeasure.md, 2026-11-09: real
    # Qwen3 exponent planes, GPU timestamps, byte-identical. Its own k there is
    # 268.9-273.1 lane-cycles/B at <=1 TG/core and ~300 at 4/core, at an
    # IOReport-measured clock -- compute-bound, DRAM at ~28% of spec.
    "lmz Metal, shared table, Apple M4, kernel only": 25.2,
    # The same kernel at exactly its residency limit (40 threadgroups, one
    # wave): what the model's compute term describes, with no wave tail.
    "lmz Metal, shared table, Apple M4, 40 resident threadgroups": 25.9,
    # The same decoder called the shipped way (lmz.gpu.decode_batch, wall
    # clock): host copies, not decoding, set this number.
    "lmz Metal, through lmz.gpu.decode_batch, Apple M4": 2.0,
    # Phase 4 of the same record. The per-chunk-table kernel is what archives
    # written today use; its shipped launch geometry (4 streams per 32-thread
    # threadgroup) is ~2.3x below the same kernel at ~100 threadgroups.
    "lmz Metal, per-chunk table, Apple M4, shipped geometry": 4.58,
    "lmz Metal, per-chunk table, Apple M4, best grid (100 TGs)": 10.6,
    # Fused BF16, counted in BF16 output bytes (19.6 in exponent-plane bytes).
    # lmz's host API exposes no fused function; this is the kernel dispatched
    # directly.
    "lmz Metal, fused BF16, Apple M4, kernel only (BF16 bytes)": 39.2,
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


def _table(codec_cost=None, metal_cost=None) -> None:
    by = constants_by_backend(codec_cost, metal_cost)
    c = by["cuda"]
    print("Can this machine decode faster than its disk?\n")
    print(f"f = {F_CODED}  ->  ideal speedup 1/f = {1 / F_CODED:.2f}x")
    print("Constants are per kernel, and a device is predicted with its own")
    print("backend's kernel:")
    for name in BACKENDS:
        cb = by[name]
        tag = (f"  BORROWED from {cb.borrowed_from}: no {name} kernel measured"
               if cb.borrowed_from else "")
        print(f"  k[{name:<6}] = {cb.k_low:.1f}-{cb.k_high:.1f} {K_UNIT}; "
              f"{cb.block_threads} threads, {cb.shmem_per_block()} B a block, "
              f"traffic {cb.traffic:.3f}{tag}")
    for name in ("cuda", "metal"):
        print(f"    {name} provenance: {by[name].provenance}")
    print()

    print("PREDICTIONS, not measurements. Every row is computed from the inputs")
    print("shown; a row missing an input carries an interval and says which.\n")
    print(f"{'device':<24}{'lanes':>8}{'GHz':>6}{'GB/s bus':>10}"
          f"{'compute':>16}{'b/width':>9}   decode GB/s   bound")
    for d in DEVICES:
        cd = constants_for(d, by)
        comp = d.compute_bound(cd)
        bw = d.bandwidth_bound(cd)
        lo, hi, why = d.predict(cd)
        binds = d.residency_binds(cd)
        if binds:
            why = f"{why} [residency: {binds}]"
        cs = "  unknown" if comp is None else f"{comp[0]:.0f} - {comp[1]:.0f}"
        band = (f"<= {hi:.1f}" if lo <= 0.0
                else (f"{lo:.1f} - {hi:.1f}" if hi - lo > 0.05 else f"{hi:.1f}"))
        print(f"{d.name:<24}{(d.resident_lanes(cd) or 0):>8}"
              f"{(d.clock_ghz or 0):>6.2f}{d.gb_s:>10.1f}{cs:>16}{bw:>9.0f}"
              f"   {band:>12}   [{d.backend}] {why}")
    print()

    # --- Which term binds, and whether we are entitled to say -----------------
    if c.can_pick:
        print("WHICH TERM BINDS IS A PROPERTY OF THE DEVICE, and k is now strong")
        print("enough to say so. It came from a sweep across residency rather than")
        print("from one launch, and a sweep is what separates a compute-bound from")
        print("a bandwidth-bound reading -- one point fits both.\n")
        for d in DEVICES:
            m = d.margin(constants_for(d, by))
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
    print(f"cuda's k was measured under partial latency hiding at "
          f"{c.blocks_at_measurement}+ blocks per unit; a device holding")
    print("fewer hides less of the dependent-load chain, so its effective k")
    print("moves toward the high end or past it and the interval becomes a")
    print("floor. Resident lanes captures the occupancy part of that and not")
    print("the hiding part -- which is why k stays an interval and why these")
    print("rows may claim a ceiling and not a bracket.")
    print(f"metal's k was measured at {by['metal'].blocks_at_measurement}-4 "
          f"threadgroups per core, and there it RISES with co-residency:")
    print("co-resident groups contend for the core rather than only hiding")
    print("each other's latency. Same model, opposite trend; one device each.\n")

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
        cd = constants_for(d, by)
        lo, _hi, _why = d.predict(cd)
        print(f"{d.name:<24}", end="")
        for _name, disk, _src in STORAGE:
            v = verdict(d, disk, cd)
            sp = speedup(lo, disk)
            cell = v if sp >= 1 / F_CODED - 1e-9 else f"{v} {sp:.2f}x"
            print(f"{cell:>14}", end="")
        print()
    print()
    print("Bare 'pays' is the saturated case: the decoder beats the disk by more")
    print("than 1/f, so the archive's ratio has become load speed and nothing")
    print("faster helps. 'measure' means the interval straddles that disk. One")
    print("device changes verdict with the disk beside it, which is why a gate")
    print("calibrated against one machine's NVMe is not a gate.")
    print("A verdict assumes the route delivers the decoder kernel's rate. On the")
    print("Apple M4 the kernel does, but lmz's shipped Python path does not: it")
    print("measured ~2 GB/s there (MEASURED_DECODE), so a Mac route needs a")
    print("zero-copy decoder interface before its 'pays' holds end to end.\n")

    print("Measured decode rates, for comparison (no model involved)")
    for name, v in MEASURED_DECODE.items():
        print(f"  {name:<46}{v:>8.1f} GB/s")
    print()
    ref = DEVICES[0]
    lo, hi, _ = ref.predict(c)
    print(f"Check: the model predicts {ref.name} at {lo:.0f}-{hi:.0f} GB/s against")
    print(f"{MEASURED_DECODE['lmz CUDA, shared table, RTX 5080']:.0f} measured. That is the")
    print("one point cuda's constants were taken on, so agreement is a")
    print("consistency check and not a validation.")
    m4 = next(d for d in DEVICES if d.name.startswith("Apple M4"))
    mlo, mhi, _ = m4.predict(constants_for(m4, by))
    at40 = MEASURED_DECODE["lmz Metal, shared table, Apple M4, 40 resident threadgroups"]
    print(f"Likewise {m4.name}: {mlo:.1f}-{mhi:.1f} GB/s predicted with metal's")
    print(f"constants against {at40:.1f} measured at its residency limit -- again the")
    print("device those constants came from.")
    xb = m4.blocks_per_unit(c)
    print("The one check on a device the constants were NOT fitted on is")
    print(f"residency: cuda's kernel shape ({c.shmem_per_block()} B, "
          f"{c.block_threads} threads) on the M4's probed")
    print(f"pool gives {xb} blocks per core, and probe/metal-bench.js measured 2 per")
    print("core for that shape by occupancy staircase. The formula transferred;")
    print("whether cuda's k transfers is not testable there, because that kernel")
    print("does not run on Metal.\n")

    small = DEVICES[-1]
    cpu = MEASURED_DECODE["lmz CPU, 4-16 threads"]
    print("And the founding claim, which this model CANNOT currently decide:")
    print()
    print(f"{small.name} has a measured bus and clock and a queried CAP of")
    print(f"{small.shmem_cap_per_block} B, so the kernel's {c.shmem_per_block()} B "
          f"block is legal. But residency")
    print("divides by the POOL, which is a different quantity. It was looked for")
    print("and is not obtainable here: core Vulkan reports no per-unit pool, and")
    print("VK_AMD_shader_core_properties and shader_core_properties2 -- both")
    print("present on this adapter, both queried -- carry no LDS field at all.")
    print("What the candidates would give:")
    print()
    print(f"  {'candidate pool':<34}{'blocks':>7}{'lanes':>8}   decode GB/s   vs CPU")
    for pool, why in ((small.shmem_cap_per_block, "the cap, if pool == cap"),
                      (64 << 10, "64 KiB/CU, RDNA2 documented LDS")):
        probe = Device(small.name, small.gb_s, "", units=small.units,
                       shmem_pool_per_unit=pool,
                       shmem_cap_per_block=small.shmem_cap_per_block,
                       max_threads_per_unit=small.max_threads_per_unit,
                       clock_ghz=small.clock_ghz)
        b = probe.blocks_per_unit(c)
        comp = probe.compute_bound(c)
        band = min(comp[1], probe.bandwidth_bound(c))
        print(f"  {why:<34}{b:>7}{probe.resident_lanes(c):>8}"
              f"   {comp[0]:5.1f} - {band:4.1f}   {band/cpu:.1f}x")
    print()
    print("NEITHER IS A MEASUREMENT and the row above still carries no compute")
    print("term, because an unobtained pool is not a known one. But the two")
    print("candidates are not symmetric, and that is the useful part:")
    print()
    print("**pool >= cap is forced, not assumed.** A workgroup may legally")
    print("declare `cap` bytes, so a unit that could not hold one such workgroup")
    print("could never schedule it. So residency is AT LEAST floor(cap/shm) = 1")
    print(f"block per CU, and with {small.units} CUs queried off the device that is a")
    print("FLOOR on lanes, not a guess: the arithmetic cannot go below it.")
    print()
    floor_dev = Device(small.name, small.gb_s, "", units=small.units,
                       shmem_pool_per_unit=small.shmem_cap_per_block,
                       shmem_cap_per_block=small.shmem_cap_per_block,
                       max_threads_per_unit=small.max_threads_per_unit,
                       clock_ghz=small.clock_ghz)
    fl = floor_dev.compute_bound(c)
    fl_lo = min(fl[0], floor_dev.bandwidth_bound(c))
    # State the floor per GHz rather than at one clock. This device's 2.2 GHz
    # is derived from an FMA measurement, not queried, and the compute term is
    # linear in clock across the whole range that matters -- bandwidth would
    # only bind above 16.8 GHz -- so a number quoted at one clock hides the
    # weakest input in the row. A slope and a threshold do not.
    lanes = floor_dev.resident_lanes(c)
    per_ghz = lanes / c.k_high
    threshold = cpu * c.k_high / lanes
    print(f"So the floor is {per_ghz:.2f} GB/s PER GHz of engine clock, and it")
    print(f"exceeds the CPU decoder's {cpu:.1f} GB/s for any clock above "
          f"{threshold:.2f} GHz.")
    print("Every AMD integrated part of the last several generations boosts well")
    print("clear of that, so the claim is robust rather than contingent -- and a")
    print("reader can check it against their own adapter without trusting the")
    print(f"{small.clock_ghz} GHz used here, which is derived from an FMA")
    print(f"measurement rather than queried. At that clock the floor is "
          f"{fl_lo:.1f} GB/s, {fl_lo/cpu:.1f}x the CPU decoder.")
    print()
    print("THE FLOOR IS CONSERVATIVE TWICE OVER, and both point the same way:")
    print("  1. the pool is taken as EQUAL to the cap, its smallest legal value;")
    print("  2. the cap itself is under-reported. On the reference card Vulkan")
    print("     says 49152 where CUDA's opt-in cap is 101376, so Vulkan reports")
    print("     the non-opt-in limit. If this adapter's real opt-in cap is 64 KiB")
    print("     rather than the 32768 Vulkan gives, residency doubles again.")
    print("So the true value is above this floor, not scattered around it.")
    print()
    print("THAT FLOOR IS ARITHMETIC, NOT EVIDENCE. It inherits every assumption")
    print("above it, and two of them are unverified precisely at this scale:")
    print("whether decode stays linear in resident lanes down to 2 CUs, and how")
    print("a unified-memory part behaves with the CPU on the same bus. The clock")
    print("is derived from an FMA measurement rather than queried. So this is a")
    print("floor on the MODEL, and the model is what is untested here. Its k is")
    print("also borrowed from cuda: no Vulkan kernel has been measured.")
    print("The one unified-memory part measured, an Apple M4, lost 17-29% of its")
    print("decode rate to a CPU streaming 71-81 GB/s on the same bus, although")
    print("DRAM was not what bound it -- so on a small part the floor above may")
    print("not hold under load, and nothing here says by how much.")
    print()
    print("This row has now been stated three different ways in one day -- 4-36,")
    print("then <= 7.8, then <= 1.9 -- each time because a shared-memory number")
    print("was substituted for one it is not. The lesson is in the file rather")
    print("than the number: cap and pool are different quantities, and every")
    print("device fact above is now queried or marked unobtained.")
    print()
    print("What would settle it is this adapter's LDS per compute unit, or one")
    print("run of the kernel on it. The first is not available: core Vulkan does")
    print("not report it and neither AMD shader-core extension carries it. So it")
    print("is a MEASUREMENT question, not a query question, and that is what a")
    print("Vulkan backend would buy -- not a better estimate, an end to them.")


def main() -> None:
    """Standalone: no codec, no imports, the published constants."""
    _table(None)


if __name__ == "__main__":
    main()
