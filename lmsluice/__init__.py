"""lmsluice -- move model bytes at the rate the machine actually allows.

A compressor decides how many bytes there are. This decides how they travel:
which route is faster on this machine and this link, and then runs that route
with its stages overlapped so the move costs the slowest stage rather than the
sum of all of them.

    import lmsluice

    m = lmsluice.open_model("model.safetensors")  # or model.lmz, or an https URL
    print(m.plan.explain())                    # why this route, on what numbers
    weights = m.load()                         # one buffer, filled in parallel

    for name, view in m.stream(budget=1 << 30):
        upload(name, view)                     # bounded memory, overlapped I/O

The decision it makes is one line of arithmetic, and the whole package exists
to put real numbers into it:

    compression pays on a path exactly when the codec is faster
    than the link it feeds from -- and then by min(1/ratio, codec/link)

which is why nothing here ships a rate. It measures them, on the machine it
finds itself on, and says so when it could not.
"""

__version__ = "0.1.0"

from .archive import Archive, Tensor
from .model import Model, NotMappable, open_model
from .plan import Decision, Route, Stage, gate, read_plan, write_plan
from .probe import decode_rate, encode_rate, read_rate, write_rate
from .rates import Codec, Profile, Storage
from .source import (CachedSource, FileSource, HttpSource, Source,
                     open_source)
from .transport import Report, transport

__all__ = [
    "open_model", "Model", "NotMappable", "Tensor", "Archive",
    "read_plan", "write_plan", "gate", "Decision", "Route", "Stage",
    "Profile", "Storage", "Codec",
    "read_rate", "write_rate", "decode_rate", "encode_rate",
    "open_source", "Source", "FileSource", "HttpSource", "CachedSource",
    "transport", "Report", "__version__",
]
