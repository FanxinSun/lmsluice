"""What a machine can do, as numbers rather than assumptions.

Every decision this package makes reduces to comparing two rates: how fast
bytes cross a storage path, and how fast a codec turns coded bytes back into
plain ones. Neither is a property of the software. Both are properties of the
machine it woke up on, and on most machines nobody has measured either.

So they are measured, cached, and carried in the small records below. Nothing
here is a default tuned to any particular box: a rate that has not been
measured is `None`, and a plan built on `None` says so rather than guessing.

The units are bytes per second throughout, and every rate names which side of
the codec it counts. A storage rate counts the bytes actually on the device --
coded bytes for an archive, plain ones for a plain file. A codec rate counts
plain bytes, because plain bytes are what the caller asked for. Mixing the two
is the easiest mistake available here and the reason it is written down twice.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field

SCHEMA = 1


def cache_dir() -> str:
    """Where a measured profile is kept between runs.

    XDG on Linux and the BSDs, the platform's own cache directory elsewhere.
    A probe costs seconds and its answer changes only when the hardware does,
    so paying it once per machine rather than once per load is the difference
    between a routing decision being free and being an excuse not to make one.
    """
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "yfce-lmsluice")


@dataclass(frozen=True)
class Storage:
    """How fast bytes cross one storage path.

    `cold` is the number that matters for a model load, because a checkpoint
    is read once and is usually larger than the cache that would hold it.
    `warm` is kept because the ratio between them is the size of the mistake
    a warm measurement would have caused.

    `layered` records that something below this process caches too -- a
    hypervisor's host page cache, a network filesystem's client cache, a
    controller's DRAM -- so `cold` was only true on first touch and cannot be
    re-measured on this file without evicting a cache we cannot reach. It is
    detected, not assumed: the same range is read cold twice, and a second
    read that is much faster than the first is what a lower cache looks like.

    It is True or None and never False, which is not fussiness. The test can
    only fire when the lower cache was cold to begin with, so a file the host
    already holds produces two equally fast reads and no detection -- while
    still reporting that cache's speed as the disk's. Not detected therefore
    means not ruled out, and `cold` is an upper bound on every machine where
    it stays None. Measured on a WSL2 box, the same file gave 2.38 GB/s on
    first touch and 6.38 GB/s an hour later with the flag clear, which is the
    size of the error this distinction exists to stop being invisible.
    """

    key: str                       # device or mount this describes
    cold: float | None = None
    warm: float | None = None
    write: float | None = None
    layered: bool | None = None
    depth: int = 1                 # queue depth `cold` was reached at
    method: str = ""               # how the cache was dropped, or why it was not
    sample_bytes: int = 0

    @property
    def source(self) -> float | None:
        """The rate a first read of a large file will actually see."""
        return self.cold if self.cold is not None else self.warm


@dataclass(frozen=True)
class Codec:
    """How fast a codec runs on this machine, in plain bytes per second.

    Measured against a particular archive rather than in the abstract, because
    it is not one number: chunk size, which codecs the encoder chose, and how
    many of them the interpreter has to step between all move it, and they are
    properties of the file rather than of the build. `archive` names what it
    was measured on so a rate is never quoted about a file it did not come
    from.
    """

    name: str                      # "lmz-cpu", "lmz-cuda", ...
    decode: float | None = None
    encode: float | None = None
    threads: int = 0
    archive: str = ""
    ratio: float | None = None     # coded / plain, of the archive it saw
    sample_bytes: int = 0


@dataclass
class Profile:
    """One machine's measurements, and enough about it to know they are stale."""

    host: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)
    codecs: dict = field(default_factory=dict)
    measured_at: float = 0.0
    schema: int = SCHEMA

    # -- serialisation ----------------------------------------------------
    def to_json(self) -> dict:
        return {"schema": self.schema, "measured_at": self.measured_at,
                "host": self.host,
                "storage": {k: asdict(v) for k, v in self.storage.items()},
                "codecs": {k: asdict(v) for k, v in self.codecs.items()}}

    @staticmethod
    def from_json(d: dict) -> "Profile":
        if d.get("schema") != SCHEMA:
            raise ValueError("profile was written by a different version")
        return Profile(
            host=d.get("host", {}),
            storage={k: Storage(**v) for k, v in d.get("storage", {}).items()},
            codecs={k: Codec(**v) for k, v in d.get("codecs", {}).items()},
            measured_at=d.get("measured_at", 0.0), schema=SCHEMA)

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(cache_dir(), "profile.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(self.to_json(), fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str | None = None) -> "Profile | None":
        path = path or os.path.join(cache_dir(), "profile.json")
        try:
            with open(path) as fh:
                p = Profile.from_json(json.load(fh))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        # A profile from different hardware is not this machine's, and the
        # cheapest way to be wrong here is to trust a cache file that moved
        # with a home directory onto another box.
        return p if p.host.get("fingerprint") == fingerprint() else None


def fingerprint() -> str:
    """Enough of the machine to notice it is a different one.

    Deliberately coarse. It exists to invalidate a cached profile that
    travelled -- in a home directory, a container image, a synced folder --
    not to describe the hardware, which `host_info` does.
    """
    return "|".join((platform.system(), platform.machine(),
                     platform.node(), str(os.cpu_count() or 0)))


def host_info() -> dict:
    """What was true about the machine when it was measured."""
    info = {
        "fingerprint": fingerprint(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpus": os.cpu_count() or 0,
        "page_size": _page_size(),
    }
    try:
        from ._lmz import lmz as _find

        m = _find()
        info["lmz"] = m.__version__
        info["lmz_backends"] = m.backends()
    except Exception as exc:              # noqa: BLE001 -- lmz is optional here
        info["lmz"] = None
        info["lmz_error"] = str(exc)
    return info


def _page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 4096


def storage_key(path: str) -> str:
    """Which storage path a file sits on.

    Two files on one machine can be on devices that differ by two orders of
    magnitude -- an NVMe, a network mount, a USB stick -- so a rate measured
    on one says nothing about the other, and a profile keyed by machine alone
    would quietly apply the fast one everywhere. Keyed by device id where the
    platform exposes one, and by mount point where it does not.
    """
    path = os.path.abspath(path)
    probe = path if os.path.exists(path) else os.path.dirname(path) or "."
    try:
        st = os.stat(probe)
    except OSError:
        return probe
    if os.name == "nt" or st.st_dev == 0:
        drive = os.path.splitdrive(path)[0]
        return drive or _mount_of(probe)
    return f"dev:{st.st_dev}"


def _mount_of(path: str) -> str:
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def age(profile: Profile) -> float:
    """Seconds since the profile was taken."""
    return max(0.0, time.time() - profile.measured_at)
