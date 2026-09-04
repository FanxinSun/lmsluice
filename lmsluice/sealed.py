"""Encryption at rest, as an envelope around a container rather than inside one.

A checkpoint on a shared filesystem, in an object store, or on a laptop that
leaves the building is readable by anyone who can read the file. tensorizer
encrypts per tensor for that reason, and it is the one row in
`docs/competition.md` §7 where a competitor has something we do not.

## Why an envelope, and not an option on the codec

The first version of this put a `key_file` option on `ZstdEncoder` and sealed
each chunk as it was compressed. One pass, no extra disk -- and wrong, for a
reason `docs/boundary.md` states directly: **lmz is the compressor, lmsluice is
the transport.** Encryption does not change when the format changes; it changes
when the threat model changes. Its three-question test therefore puts it in
this tree, and putting it inside a codec makes every future codec implement it
again.

The practical half of the same argument settles it. `LmzEncoder.encode` is
`lmz().compress(src, dst, **opts)` -- lmz writes its own container, and we do
not edit `lmz/`. An option on our stdlib codec would have encrypted only the
fallback codec and left the one that gives the good ratios on BF16 weights
unencryptable. An envelope wraps a container it knows nothing about, so lmz
archives, zstd archives and whatever comes next are all covered by this file
and none of them mention encryption.

## The shape on disk

    [0, 8)          LMSLSEAL
    body            the inner container's bytes, extent by extent, in inner
                    order: payload extents sealed, everything else verbatim
    index           zlib(JSON): the extent table, the encryption header, the MAC
    last 24 bytes   FOOTER(index_off, index_len, LMSLSTAI)

**Payload sealed, structure clear.** The codec's units are encrypted; its index,
tensor names, shapes and offsets are not. That is tensorizer's choice too and it
is the right trade: the weights are the secret, the shape of the file is not.
It is also what lets `info` and `map` read an archive's structure, and a loader
plan a partial read, without the key.

**Clear does not mean unprotected.** The extent table and every clear byte are
covered by an HMAC-SHA256 under a separate subkey. Confidentiality for the
payload, integrity for all of it -- otherwise an attacker who cannot read a
weight could still edit the index that says where the weights are, and
authenticated payloads would be authenticating bytes fetched from the wrong
place. Without a key the structure still reads, and `verified` is False.

**Compress, then encrypt. Never the reverse** -- ciphertext is indistinguishable
from random and does not compress, so encrypting first would cost the entire
ratio. Wrapping a finished container gets that order by construction.

**One subkey per archive, so counters are safe.** `crypt` draws a random salt
per archive and derives the subkeys from it; the nonce is then the extent's
ordinal. See `crypt.py` on why a counter nonce is sound only under a fresh key.

## What it costs, and the one thing it takes away

Sealing is a second pass: the inner container is read once and the envelope
written once, so peak disk is inner + outer rather than outer alone. Both are
compressed sizes, and the envelope is opt-in.

The thing it takes away is `mmap`. A sealed archive cannot be mapped, because
there is no arrangement of page tables that decrypts. `Model.map()` already
refuses the coded route and that refusal now carries this case; a machine whose
gate says "read the plain file" is a machine where encryption costs a real
decode, and `plan.py` prices it rather than hiding it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import zlib

from . import crypt
from .source import Source

MAGIC = b"LMSLSEAL"
TAIL = b"LMSLSTAI"
VERSION = 1
FOOTER = struct.Struct("<QQ8s")
_MAC_INFO = b"lmsluice envelope mac v1"

# Extent kinds, stored as the fifth field of a row so the table reads as data.
CLEAR = 0
SEALED = 1


class NotSealed(ValueError):
    """The file does not carry the envelope magic."""


def is_sealed(source) -> bool:
    """By magic, not by extension: a name is a hint, the first eight bytes are
    a fact. Never raises -- this is asked of every file that is opened."""
    try:
        return source.pread(0, len(MAGIC)) == MAGIC
    except Exception:                     # noqa: BLE001
        return False


def _mac_key(master: bytes, salt: bytes) -> bytes:
    return crypt._hkdf(master, salt, _MAC_INFO, crypt.KEY_BYTES)


def _row(e) -> bytes:
    return struct.pack("<QQQQB", e[0], e[1], e[2], e[3], e[4])


# -- writing ---------------------------------------------------------------

def extents_of(units, inner_size: int) -> list[list[int]]:
    """The inner file cut into payload and non-payload runs, in inner order.

    The codec says where its payload is; everything it does not claim is
    structure and stays readable. Adjacent clear runs are merged so a container
    with a header and a trailing index costs two rows rather than thousands.
    """
    spans = sorted((int(u.off), int(u.clen)) for u in units)
    rows: list[list[int]] = []
    at = 0
    for off, clen in spans:
        if off < at:
            raise ValueError(
                f"codec units overlap at {off}: this envelope seals each unit "
                f"once and cannot express a byte that is in two of them")
        if off > at:
            rows.append([at, off - at, 0, 0, CLEAR])
        rows.append([off, clen, 0, 0, SEALED])
        at = off + clen
    if at < inner_size:
        rows.append([at, inner_size - at, 0, 0, CLEAR])
    if at > inner_size:
        raise ValueError(f"units run to {at}, past the end of a {inner_size}-byte file")
    return rows


def seal_file(inner_path: str, out_path: str, key_file: str | None = None,
              *, units=None, inner_name: str | None = None,
              block: int = 8 << 20) -> dict:
    """Wrap a finished container in an encrypted envelope.

    `units` is the codec's unit list; when it is not given the container is
    opened to ask for one, which is why this works for any codec and mentions
    none. Returns the encryption header -- never the key.
    """
    import os

    if not crypt.available():
        raise crypt.CryptoUnavailable(
            f"cannot encrypt: {crypt.backend()[1]}. Refusing to write a file "
            f"that would look protected and not be.")

    master = crypt.load_key(key_file)
    salt = crypt.new_salt()
    key = crypt.subkey(master, salt)
    mac = hmac.new(_mac_key(master, salt), digestmod=hashlib.sha256)
    header = {"algorithm": "AES-256-GCM", "kdf": "HKDF-SHA256",
              "key_id": crypt.key_id(master), "salt": salt.hex(),
              "backend": crypt.backend()[0]}
    del master

    if units is None:
        from .archive import Archive

        with Archive(inner_path) as arc:
            units = list(arc.codec.units())

    inner_size = os.path.getsize(inner_path)
    rows = extents_of(units, inner_size)

    ordinal = 0
    with open(inner_path, "rb") as fi, open(out_path, "wb") as fo:
        fo.write(MAGIC)
        for row in rows:
            off, length, _, _, kind = row
            row[2] = fo.tell()
            fi.seek(off)
            if kind == SEALED:
                # Sealed whole, so one unit is one nonce and one tag. A unit is
                # what the codec reads in one go, so this never splits a read.
                blob = crypt.seal(key, ordinal, fi.read(length))
                fo.write(blob)
                row[3] = len(blob)
                ordinal += 1
            else:
                # Verbatim, but not unguarded: every clear byte goes into the
                # MAC as it is written, streamed so a large header costs no
                # more memory than a block.
                left = length
                while left:
                    chunk = fi.read(min(block, left))
                    if not chunk:
                        raise ValueError(f"{inner_path} ended early at {off}")
                    mac.update(chunk)
                    fo.write(chunk)
                    left -= len(chunk)
                row[3] = length
        for row in rows:
            mac.update(_row(row))
        index = {"version": VERSION, "inner": inner_size,
                 "name": inner_name or os.path.basename(inner_path),
                 "encryption": header, "extents": rows,
                 "mac": mac.hexdigest()}
        blob = zlib.compress(json.dumps(index, separators=(",", ":")).encode(), 6)
        index_off = fo.tell()
        fo.write(blob)
        fo.write(FOOTER.pack(index_off, len(blob), TAIL))
    return header


# -- reading ---------------------------------------------------------------

class SealedSource(Source):
    """The plaintext container inside an envelope, addressed as if it were the
    file itself.

    A `Source` and not a codec on purpose: every codec, `Archive`, `Model` and
    `probe` already read through this interface, so unwrapping here is the whole
    of the read side and no other module learns the word "encrypted". It is the
    same move `CachedSource` makes for a server that ignores `Range`.
    """

    def __init__(self, outer: Source, *, key_file: str | None = None):
        self._outer = outer
        self.name = outer.name
        if outer.pread(0, len(MAGIC)) != MAGIC:
            raise NotSealed(f"{outer.name} is not an lmsluice envelope")
        foot = outer.pread(outer.size - FOOTER.size, FOOTER.size)
        index_off, index_len, tail = FOOTER.unpack(foot)
        if tail != TAIL:
            raise ValueError(f"{outer.name}: envelope footer is damaged")
        self.index = json.loads(zlib.decompress(outer.pread(index_off, index_len)))
        self.size = int(self.index["inner"])
        self.random_access = True
        self.encryption = self.index["encryption"]
        self._rows = [tuple(r) for r in self.index["extents"]]
        self._starts = [r[0] for r in self._rows]
        self._key = None
        self.verified = False
        self._why = ""
        self._open_key(key_file)

    def _open_key(self, key_file) -> None:
        """Check the key when the file is opened, not on the first chunk.

        An archive that is going to fail should fail with a message about keys.
        Discovering it on chunk one gives an authentication failure halfway
        through a load, which reads like corruption and sends people looking in
        the wrong place.

        Opening without a key is allowed and deliberate: structure is in the
        clear, so `info` and a plan work for someone who cannot read the
        weights. Only a sealed extent insists.
        """
        want = self.encryption.get("key_id")
        try:
            master = crypt.load_key(key_file)
        except crypt.CryptoUnavailable as exc:
            self._why = str(exc)
            return
        got = crypt.key_id(master)
        if got != want:
            raise ValueError(
                f"{self.name}: written with a different key (it wants key id "
                f"{want}, the one supplied is {got}). Nothing was decrypted.")
        salt = bytes.fromhex(self.encryption["salt"])
        self._key = crypt.subkey(master, salt)
        self._verify(_mac_key(master, salt))
        del master

    def _verify(self, mac_key: bytes) -> None:
        """The clear bytes and the extent table, checked before anything is
        read through them. A payload tag proves a unit is authentic; it cannot
        prove the table did not move where that unit is read from."""
        mac = hmac.new(mac_key, digestmod=hashlib.sha256)
        for row in self._rows:
            if row[4] == CLEAR:
                mac.update(self._outer.pread(row[2], row[3]))
        for row in self._rows:
            mac.update(_row(list(row)))
        if not hmac.compare_digest(mac.hexdigest(), self.index["mac"]):
            raise ValueError(
                f"{self.name}: the archive's structure failed authentication. "
                f"Its index or one of its clear regions was altered after it "
                f"was written. Nothing was decrypted.")
        self.verified = True

    def pread(self, offset: int, length: int) -> bytes:
        """Inner bytes, unsealing whichever extents the range crosses.

        No cache and no lock, and both on purpose: a unit is sealed whole, the
        codec reads a unit whole, so a read unseals each extent it touches
        exactly once. Unsealing is pure given the key, which makes this safe
        under the fetch pool without any shared mutable state.
        """
        import bisect

        if length <= 0:
            return b""
        end = min(offset + length, self.size)
        i = bisect.bisect_right(self._starts, offset) - 1
        if i < 0:
            raise ValueError(f"{self.name}: offset {offset} is before the file")
        out = bytearray()
        ordinal = sum(1 for r in self._rows[:i] if r[4] == SEALED)
        while i < len(self._rows) and self._rows[i][0] < end:
            off, length_i, outer_off, outer_len, kind = self._rows[i]
            if kind == CLEAR:
                lo = max(offset, off)
                hi = min(end, off + length_i)
                out += self._outer.pread(outer_off + (lo - off), hi - lo)
            else:
                if self._key is None:
                    raise crypt.CryptoUnavailable(
                        f"{self.name} is encrypted and its weights cannot be "
                        f"read without the key. Its structure can, which is "
                        f"why opening it worked. {self._why}")
                try:
                    plain = crypt.open_chunk(
                        self._key, ordinal, self._outer.pread(outer_off, outer_len))
                except crypt.DecryptionFailed as exc:
                    raise ValueError(
                        f"{self.name}: unit {ordinal} (inner offset {off}) "
                        f"failed authentication -- {exc}") from None
                if len(plain) != length_i:
                    raise ValueError(
                        f"{self.name}: unit {ordinal} decrypted to "
                        f"{len(plain)} bytes, expected {length_i}")
                lo = max(offset, off) - off
                hi = min(end, off + length_i) - off
                out += plain[lo:hi]
                ordinal += 1
            i += 1
        return bytes(out)

    def close(self) -> None:
        self._outer.close()


def open_source(target, source=None, *, key_file: str | None = None):
    """A `SealedSource` over `target`, opening the outer source if need be."""
    from .source import Source as _S, open_source as _open

    if source is None:
        source = target if isinstance(target, _S) else _open(target)
    return SealedSource(source, key_file=key_file)
