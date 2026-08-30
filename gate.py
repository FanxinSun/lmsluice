"""Can this machine decode faster than its disk?

The whole of `load/` rests on one comparison, and the point of putting it in
a file is that it has to be evaluated on *someone else's* machine, not on the
one it was calibrated on. Nothing here is specific to the box in MEASURED.md
except the rows marked as measured there.

    load time, plain file      =  N / disk
    load time, coded archive   =  max(f * N / disk, N / decode)   (overlapped)

    speedup                    =  min(1/f, decode / disk)

So an archive pays when the decoder beats the disk, and its ratio becomes
load speed only once the decoder beats the disk by 1/f. Both terms are
properties of the machine in front of you.

The decoder term is the one that needs a model, because it cannot be looked
up. An entropy decoder spends compute to avoid reading bytes, so it is bounded
from two sides:

    decode  <=  BW / (1 + f)     it reads f coded bytes and writes 1 decoded
                                 byte for every byte it produces
    decode  <=  C / k            k = compute per decoded byte

    decode  =   min(BW / (1 + f), C / k)

and which of the two binds is decided by the device's compute-per-bandwidth
ratio, with the crossover at C/BW = k / (1 + f).

**k is not measured, and that is the honest state of this project.** One
number exists -- lmz's CUDA decoder at 418 GB/s on an RTX 5080 -- and it fits
both readings at once: it is 84% of that card's bandwidth bound, and it is
exactly C/k for k = 128. The two readings agree on the card that produced
them and disagree by 8x on a 2-CU integrated GPU, which is the class this
project exists for. Scaling that one point by FP32 throughput -- what the
first version of README.md's table did -- silently picks the compute reading,
i.e. assumes what was never measured.

Until k is pinned on one low-FLOP/byte device, every device gets an interval:
the compute reading at k = 128 as the floor, the bandwidth bound as the
ceiling. Verdicts that hold across the whole interval are safe to design on;
the rest are reasons to run `probe/`.

No dependencies. `python3 gate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

# lmz on model weights: 34.7% less disk (README.md), so the coded stream is
# 0.653 of the original. The end-to-end 1.46x measured against plain
# safetensors implies an effective f nearer 0.685 once container and
# per-chunk overheads are counted; both are quoted rather than averaged.
F_CODED = 0.653

# Upper bound on the decoder's compute cost per decoded byte, in FP32-FLOP
# equivalents, derived from the single measured point by assuming that point
# was compute-bound. rANS decode is integer work with dependent shared-memory
# loads, so "FP32-FLOP equivalent" is a proxy unit and not a physical one; it
# is used because it is the unit `probe/igpu-bench.ps1` can measure on any
# device without a toolkit.
K_FLOP_PER_BYTE = 53_691.0 / 418.0  # = 128.4


@dataclass(frozen=True)
class Device:
    name: str
    gflops: float  # FP32 FMA throughput
    gb_s: float  # streaming read bandwidth
    source: str  # measured where, or derived how

    @property
    def flop_per_byte(self) -> float:
        return self.gflops / self.gb_s

    @property
    def bandwidth_bound(self) -> float:
        return self.gb_s / (1.0 + F_CODED)

    @property
    def compute_bound(self) -> float:
        return self.gflops / K_FLOP_PER_BYTE

    @property
    def interval(self) -> tuple[float, float]:
        """(floor, ceiling) GB/s of decode: the compute reading, then the
        bandwidth bound. They coincide once k is pinned."""
        lo = min(self.compute_bound, self.bandwidth_bound)
        return lo, self.bandwidth_bound

def speedup(decode_gb_s: float, disk_gb_s: float, f: float = F_CODED) -> float:
    """How much faster a coded archive opens than a plain file."""
    return min(1.0 / f, decode_gb_s / disk_gb_s)


def verdict(dev: Device, disk_gb_s: float) -> str:
    lo, hi = dev.interval
    if lo > disk_gb_s:
        return "pays"
    if hi <= disk_gb_s:
        return "tax"
    return "measure"


# --- Devices ------------------------------------------------------------------
# Two rows are measured on the box in MEASURED.md. The rest are public
# specifications turned into arithmetic, and are NOT results: they say where a
# class of machine sits on the curve, so that a conclusion can be checked
# against the class instead of against one desktop.

DEVICES = [
    Device("RTX 5080 (dGPU)", 53_691, 822, "measured, MEASURED.md"),
    Device("M4 Max (unified)", 14_336, 459, "spec: 40 cores x 128 ALU x 2 x 1.4 GHz; 546 x 0.84"),
    Device("Strix Halo (unified)", 14_848, 215, "spec: 40 CU x 64 x 2 x 2.9 GHz; 215 measured, blueprint §1"),
    Device("Radeon 780M (iGPU)", 4_300, 60, "spec arithmetic, never run"),
    Device("Phone SoC GPU (class)", 2_500, 57, "spec class figure; 68 x 0.84, blueprint §1"),
    Device("Radeon 2-CU (iGPU)", 567, 59.4, "measured, MEASURED.md"),
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

# Measured decode rates, for the rows where no model is needed.
MEASURED_DECODE = {
    "lmz CUDA, shared table, RTX 5080": 418.0,
    "lmz CUDA, per-chunk table, RTX 5080": 111.0,
    "lmz CPU, 4-16 threads": 2.0,
}


def _table() -> None:
    print("Can this machine decode faster than its disk?\n")
    print(f"f = {F_CODED}  ->  ideal speedup 1/f = {1 / F_CODED:.2f}x")
    print(f"k <= {K_FLOP_PER_BYTE:.0f} FLOP-equivalents per decoded byte -- one point, one card")
    print(f"compute binds below C/BW = {K_FLOP_PER_BYTE / (1 + F_CODED):.0f} FLOP/byte\n")

    print("Where each class of machine sits on the curve")
    print(f"{'device':<24}{'FLOP/B':>8}{'compute':>10}{'b/width':>10}   decode GB/s")
    for d in DEVICES:
        lo, hi = d.interval
        band = f"{lo:.0f}" if abs(hi - lo) < 0.5 else f"{lo:.0f} - {hi:.0f}"
        print(
            f"{d.name:<24}{d.flop_per_byte:>8.0f}{d.compute_bound:>10.0f}"
            f"{d.bandwidth_bound:>10.0f}   {band}"
        )
    print()
    print("**No shipping device is bandwidth-bound at k = 128.** Every row sits")
    print("below the crossover, so if the decoder really costs what the compute")
    print("reading says, decode is a compute problem everywhere and FLOP-per-byte")
    print("is not the discriminator -- absolute compute is. That inverts how the")
    print("first version of README.md's table read, and it is a consequence of")
    print("one unmeasured constant, which is the point.")
    print()
    print("The interval is that constant. Its floor takes the RTX 5080 point as")
    print("compute-bound; its ceiling takes it as bandwidth-bound. The two")
    print("readings differ by 16% on the card that produced the number and by 8x")
    print("on a 2-CU display adapter -- the device this project's case rests on.\n")

    print("Does the archive pay, and against which disk?")
    print(f"{'device':<24}", end="")
    for name, disk, _src in STORAGE:
        print(f"{name.split(' (')[0]:>14}", end="")
    print()
    print(f"{'':<24}", end="")
    for _name, disk, _src in STORAGE:
        print(f"{format(disk, '.2f') + ' GB/s':>14}", end="")
    print()
    for d in DEVICES:
        lo, _ = d.interval
        print(f"{d.name:<24}", end="")
        for _name, disk, _src in STORAGE:
            v = verdict(d, disk)
            sp = speedup(lo, disk)
            cell = v if sp >= 1 / F_CODED - 1e-9 else f"{v} {sp:.2f}x"
            print(f"{cell:>14}", end="")
        print()
    print()
    print("Bare 'pays' is the saturated case: the decoder beats the disk by more")
    print("than 1/f, so the archive's ratio has become load speed and nothing")
    print("faster helps. 'measure' means the interval straddles that disk -- the")
    print("pairing is undetermined until k is pinned, which is not a negative")
    print("result. One device changes verdict with the disk beside it, which is")
    print("why a gate calibrated against one machine's NVMe is not a gate.\n")

    print("Measured decode rates, for comparison (no model involved)")
    for k, v in MEASURED_DECODE.items():
        print(f"  {k:<40}{v:>8.1f} GB/s")
    print()
    print(f"The CPU row is the problem this project exists for: 2.0 GB/s is a tax")
    print(f"on any machine whose disk beats it and a {speedup(2.0, 0.55):.2f}x win on one with a")
    print("SATA SSD. Same code, same archive, opposite verdicts, decided entirely")
    print("by the machine in front of it.")
    print()
    small = DEVICES[-1]
    print(f"And the claim this project was founded on survives the worst reading:")
    print(f"the smallest integrated GPU AMD ships decodes at {small.interval[0]:.1f} GB/s even if")
    print(f"the pessimistic k is right, against the CPU decoder's 2.0 -- {small.interval[0] / 2.0:.1f}x, on a")
    print("2-CU display adapter. What does not survive is reading its verdict")
    print("against one desktop's NVMe as the verdict.")


if __name__ == "__main__":
    _table()
