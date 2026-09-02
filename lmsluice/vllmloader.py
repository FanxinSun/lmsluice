"""A vLLM `--load-format lmsluice`, through the mechanism vLLM actually has.

vLLM takes third-party loaders by registration, not by configuration: a
`BaseModelLoader` subclass decorated with `register_model_loader("name")`,
reached from a `vllm.general_plugins` entry point so the registration happens
before a loader is looked up. (The `--load-format-plugin` RFC that would have
made this a flag was closed as not planned; building on it would be building on
something that does not exist.)

Installing this package therefore adds `--load-format lmsluice` to a vLLM that
was never told about it, and removing the package removes the option.

**This module is imported by vLLM and by nothing else**, which is why it is the
one place allowed to import vLLM at module scope. `torchadapter` is the only
module that imports torch; this one does not import it at all, and reaches
tensors through that adapter.

## What it is honest about

**It streams, because the interface is a generator for a reason.** vLLM calls
`model.load_weights(iterable_of_(name, tensor))` so that a 70 B checkpoint never
exists twice, once in the loader and once in the parameters. Handing it a dict
would fit the signature and defeat the design, so this drives
`torchadapter.stream_tensors`, which holds one window at a time.

**On a fast local disk it will decline to help, and that is correct.** The gate
compares a decode rate against the machine's own storage rate, and on a warm
local NVMe the storage side wins outright -- `MEASURED.md` measures the decoder
at ~3.4 GB/s against a mapped read at 22. A plain checkpoint is then the right
thing to read and this loader reads it plainly. It earns its place on the
sources where the arithmetic goes the other way: a slow disk, a network mount, a
download, a phone. Choosing this loader on a Gen5 NVMe and expecting a win is
choosing it for the wrong machine, and it will say so rather than pretend.

**It does not speak the Hub.** `model_config.model` must name something on this
filesystem. lmsluice has no S3, no GCS, no Azure and no authentication -- a gap
`docs/competition.md` §7 records -- so a repository id is refused with that
sentence rather than half-downloaded.
"""

from __future__ import annotations

import logging
import os

from vllm.model_executor.model_loader import (  # noqa: F401 -- vLLM-only import
    BaseModelLoader,
    register_model_loader,
)

log = logging.getLogger("lmsluice")

# Containers this loader will read, in the order it prefers them. A coded
# archive first, because selecting this loader is a request to use one; the
# plain file is the fallback and, on a fast disk, the faster answer anyway.
SUFFIXES = (".lmz", ".lmsl", ".safetensors")


def register() -> None:
    """The `vllm.general_plugins` entry point.

    Importing this module is what registers the loader -- the decorator below
    runs at import -- so there is nothing for this to do but exist and be
    importable. It is a function because an entry point must name a callable.
    """
    log.debug("lmsluice model loader registered as --load-format lmsluice")


def model_files(path: str) -> list[str]:
    """The checkpoint files under `path`, coded ones preferred.

    A directory holding both `model.safetensors` and an archive built from it
    is the normal case during evaluation, and reading both would load every
    weight twice. So the first suffix that matches anything wins.
    """
    if os.path.isfile(path):
        return [path]
    for suffix in SUFFIXES:
        found = sorted(
            os.path.join(path, n) for n in os.listdir(path) if n.endswith(suffix))
        if found:
            return found
    return []


@register_model_loader("lmsluice")
class LmsluiceModelLoader(BaseModelLoader):
    """Fills vLLM's parameters over the route lmsluice measures for."""

    def download_model(self, model_config) -> None:
        """Nothing to fetch, or a refusal that names the reason.

        vLLM calls this before loading so a loader can populate a cache. This
        one has no cache to populate: it reads what is already on the
        filesystem. A model id that is not a path is refused here rather than
        at load time, because failing early is the difference between a clear
        message and a half-initialised engine.
        """
        target = model_config.model
        if os.path.exists(target):
            return
        raise ValueError(
            f"lmsluice cannot fetch {target!r}: it reads local files and has "
            f"no Hub, S3, GCS or Azure client and no authentication. Download "
            f"the model first and pass the directory, or use a loader that "
            f"speaks object storage.")

    def load_weights(self, model, model_config) -> None:
        """Hand vLLM a generator, one window at a time."""
        model.load_weights(self._weights(model_config))

    def _weights(self, model_config):
        from .torchadapter import stream_tensors

        files = model_files(model_config.model)
        if not files:
            raise ValueError(
                f"no checkpoint under {model_config.model!r}: looked for "
                f"{', '.join(SUFFIXES)}")
        log.info("lmsluice: %d file(s), %s", len(files),
                 ", ".join(os.path.basename(f) for f in files))
        for path in files:
            yield from stream_tensors(path)
