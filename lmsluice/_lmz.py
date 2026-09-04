"""Find lmz, whether it is installed or sitting next door.

lmz is a normal package and `pip install lmzip` is the usual way to have it.
It is also, in this workspace, a sibling checkout that is run in place -- its
own `lmz-cli` exists to do exactly that -- so a loader that only worked
against an installed copy could not be developed against the tree it is meant
to load from.

Both cases are handled here and nowhere else, so there is one import to fix if
either changes, and the failure says what to do rather than raising
`ModuleNotFoundError` at whatever line touched it first.
"""

from __future__ import annotations

import os
import sys

HINT = ("lmz is not importable, so the better ratios and the CUDA decoder are "
        "unavailable; the stdlib zstd codec still works. Install it with "
        "`pip install lmzip`, or put an lmz checkout in the directory "
        "containing this package.")


def lmz():
    """The lmz package, imported."""
    try:
        import lmz as _lmz
    except ImportError:
        root = _sibling()
        if root is None:
            raise ImportError(HINT) from None
        sys.path.insert(0, root)
        try:
            import lmz as _lmz
        except ImportError:
            raise ImportError(HINT) from None
    return _lmz


def _sibling() -> str | None:
    """A checkout of lmz beside this package, if there is one.

    The layout looked for is the workspace's: this file is
    `<root>/lmsluice/lmsluice/_lmz.py` and the checkout is `<root>/lmz`, whose
    package
    is `<root>/lmz/lmz`. Checked by finding the package's `__init__.py` rather
    than by the directory existing, because `<root>/lmz` is also the name of
    the repository directory and adding that to the path imports nothing.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (2, 3):
        root = os.path.abspath(os.path.join(here, *([os.pardir] * up)))
        cand = os.path.join(root, "lmz")
        if os.path.isfile(os.path.join(cand, "lmz", "__init__.py")):
            return cand
    return None


def require(*names):
    """Import submodules of lmz, raising the helpful error if it is absent."""
    lmz()
    import importlib

    return tuple(importlib.import_module(f"lmz.{n}") for n in names)
