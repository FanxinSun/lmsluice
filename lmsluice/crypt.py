"""Authenticated encryption for payload bytes, with nothing installed.

A checkpoint on a shared filesystem, in an object store, or on a laptop that
leaves the building is readable by anyone who can read the file. tensorizer
encrypts per tensor for that reason and it is the one row in
`docs/competition.md` §7 where a competitor has something we do not.

## Why OpenSSL through ctypes, and why no cipher is written here

The standard library gives hashing, HMAC, scrypt and a CSPRNG -- and no cipher.
The three ways out are to depend on `cryptography` (which retires "needs nothing
installed", the row only this project ticks), to write a cipher (which is how
amateurs produce broken cryptography), or to reach the one already on the
machine.

**It is already there.** `import ssl` links libcrypto, so on any Python that can
open an HTTPS connection the library is loaded and on disk: Linux, macOS, and
Windows with its bundled DLL. `cuda.py` reaches the CUDA driver by ctypes for
exactly this reason and this follows it. Measured on the development box:
OpenSSL 3.5.5, `libcrypto.so.3`, every EVP symbol present, nothing installed.

**Nothing here invents cryptography.** No keystream from `hashlib`, no XOR, no
home-made construction. AES-256-GCM through OpenSSL's stable EVP interface,
authenticated always -- a mode without a tag would let an attacker flip bits in
a weight tensor undetected, which for a model file is the interesting attack
rather than reading it.

## The shape

**Payload bytes only.** Tensor names, dtypes, shapes and the index stay in the
clear, so `map` and `info` can read a model's structure without the key and a
loader can plan a partial read before it can decrypt one. That is tensorizer's
choice too and it is the right trade: the weights are the secret, the shape of
the file is not.

**Compress, then encrypt. Never the reverse** -- ciphertext is
indistinguishable from random and does not compress, so encrypting first would
cost the entire ratio.

**One key per archive, so counters are safe.** A random 32-byte salt is drawn
per archive and a subkey derived from it by HKDF; the nonce is then just the
chunk index. Deriving the key per archive is what makes a counter nonce sound:
reusing a nonce under one key breaks GCM completely, and a random 96-bit nonce
per chunk would start colliding within one large archive. With a fresh subkey
the counter cannot repeat, and two archives written with the same master key get
different subkeys because the salt differs.

**A key identifier in the header**, derived by HMAC so it reveals nothing, so
the wrong key fails when the archive is opened rather than on the first chunk.

**Keys come from a file, or an environment variable naming a file.** Never from
argv, where every process on the machine can read it out of `/proc`; never
logged; never written into the archive.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import secrets
import threading

KEY_BYTES = 32                 # AES-256
SALT_BYTES = 32                # per archive, random, stored in the clear
NONCE_BYTES = 12               # GCM's native size
TAG_BYTES = 16
_INFO = b"lmsluice chunk key v1"
_KEYID_INFO = b"lmsluice key id v1"

_lock = threading.Lock()
_backend = "not probed"
_why = ""
_lib = None

# EVP_CIPHER_CTX_ctrl selectors, from OpenSSL's evp.h. Stable across 1.1 and 3.
_SET_IVLEN = 0x9
_GET_TAG = 0x10
_SET_TAG = 0x11


class CryptoUnavailable(RuntimeError):
    """No backend could be reached, so nothing will be encrypted."""


class DecryptionFailed(ValueError):
    """A tag did not verify. The plaintext is never returned in this case."""


def _candidates() -> list:
    """Where to look for libcrypto, and on macOS where never to look.

    **macOS aborts the process** if a program dlopens `/usr/lib/libcrypto.dylib`
    — Apple ships that path as a compatibility shim over LibreSSL, not as a
    library to link, and the dyld guard prints "loading libcrypto in an unsafe
    way" and raises SIGABRT. Not an exception: `SIGABRT`, which no `try` can
    catch and which takes the interpreter with it.

    `ctypes.util.find_library("crypto")` returns exactly that path on macOS, so
    the obvious implementation is the fatal one. It was, until CI ran on
    macos-latest and the suite died with `Abort trap: 6` — on a machine this
    project has, and on a path no Linux run can reach.

    So on Darwin the system paths are excluded by name and only a real OpenSSL
    installation is tried. Finding none is fine: encryption reports `none` and
    everything else works.
    """
    import ctypes.util
    import sys

    if sys.platform == "darwin":
        # Homebrew on Apple silicon, Homebrew on Intel, MacPorts, then a
        # generic prefix. Never anything under /usr/lib.
        return ["/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib",
                "/opt/homebrew/opt/openssl@1.1/lib/libcrypto.dylib",
                "/usr/local/opt/openssl@3/lib/libcrypto.dylib",
                "/usr/local/opt/openssl@1.1/lib/libcrypto.dylib",
                "/opt/local/lib/libcrypto.dylib"]
    found = ctypes.util.find_library("crypto")
    return [found, "libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so",
            "libcrypto-3-x64.dll", "libcrypto-1_1-x64.dll", "libeay32.dll"]


# NIST GCM test case 14: a 256-bit key of zeros, a 96-bit IV of zeros, and a
# 16-byte plaintext of zeros. Small enough to run at probe time, and it fails
# for every way a library can be the wrong library.
_KAT_CT = bytes.fromhex("cea7403d4d606b6e074ec5d3baf39d18")
_KAT_TAG = bytes.fromhex("d0d1c8a799996bf0265b98b5d48ab919")


def _selftest(lib) -> bool:
    """Does this library actually compute AES-256-GCM correctly?

    Binding proves a symbol exists. It does not prove the library behind it is
    the one meant, and the difference is not academic: `ctypes.CDLL(None)` on
    macOS resolves `EVP_*` against Apple's LibreSSL, every symbol binds, and
    the first real call **segfaults** -- which CI found after the SIGABRT this
    replaced, on the same platform, one commit later.

    So a candidate is accepted only once it has produced a published answer.
    That shortcut is gone too: a symbol table is not an ABI contract, and the
    only way to know a cipher is the right cipher is to make it compute
    something whose value is already known.
    """
    global _lib
    was, _lib = _lib, lib
    try:
        out = _openssl_seal(bytes(32), bytes(12), bytes(16))
        return out == _KAT_CT + _KAT_TAG
    except Exception:                     # noqa: BLE001
        return False
    finally:
        _lib = was


def _load_openssl():
    for name in _candidates():
        if not name:
            continue
        if not os.path.isabs(name) or os.path.exists(name):
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            try:
                _bind_openssl(lib)
            except AttributeError:
                continue
            if not _selftest(lib):
                continue                  # binds, but does not compute AES-GCM
            return lib, name
    return None, ""


def _bind_openssl(lib) -> None:
    p = ctypes.c_void_p
    lib.EVP_CIPHER_CTX_new.restype = p
    lib.EVP_CIPHER_CTX_new.argtypes = []
    lib.EVP_CIPHER_CTX_free.restype = None
    lib.EVP_CIPHER_CTX_free.argtypes = [p]
    lib.EVP_aes_256_gcm.restype = p
    lib.EVP_aes_256_gcm.argtypes = []
    for name in ("EVP_EncryptInit_ex", "EVP_DecryptInit_ex"):
        f = getattr(lib, name)
        f.restype = ctypes.c_int
        f.argtypes = [p, p, p, p, p]
    for name in ("EVP_EncryptUpdate", "EVP_DecryptUpdate"):
        f = getattr(lib, name)
        f.restype = ctypes.c_int
        f.argtypes = [p, p, ctypes.POINTER(ctypes.c_int), p, ctypes.c_int]
    for name in ("EVP_EncryptFinal_ex", "EVP_DecryptFinal_ex"):
        f = getattr(lib, name)
        f.restype = ctypes.c_int
        f.argtypes = [p, p, ctypes.POINTER(ctypes.c_int)]
    lib.EVP_CIPHER_CTX_ctrl.restype = ctypes.c_int
    lib.EVP_CIPHER_CTX_ctrl.argtypes = [p, ctypes.c_int, ctypes.c_int, p]


def backend() -> tuple[str, str]:
    """(name, why). One of `openssl`, `cryptography`, `none`. Never raises."""
    global _backend, _why, _lib
    # Read the memoized answer without taking the lock. Not premature: `seal`
    # and `open_chunk` ask on every chunk, so the lock was being acquired once
    # per chunk by every thread in the fetch pool -- measured here as decrypt
    # throughput *falling* from 7.2 GB/s on one thread to 0.18 on sixteen,
    # which is a mutex, not a cipher. Probing twice would be harmless (it is
    # idempotent) and cannot happen anyway: the writes below are under the
    # lock, and a Python global read is atomic.
    if _backend != "not probed":
        return _backend, _why
    with _lock:
        if _backend != "not probed":
            return _backend, _why
        if os.environ.get("LMSLUICE_NO_CRYPTO"):
            _backend, _why = "none", "disabled by LMSLUICE_NO_CRYPTO"
            return _backend, _why
        lib, name = _load_openssl()
        if lib is not None:
            _lib = lib
            _backend = "openssl"
            try:
                import ssl

                _why = f"{name} ({ssl.OPENSSL_VERSION})"
            except Exception:             # noqa: BLE001
                _why = name
            return _backend, _why
        try:
            from cryptography.hazmat.primitives.ciphers.aead import (  # noqa: F401
                AESGCM,
            )

            _backend = "cryptography"
            _why = "the optional extra; libcrypto was not reachable by ctypes"
        except Exception as exc:          # noqa: BLE001
            _backend = "none"
            _why = (f"no libcrypto by ctypes and no `cryptography` extra "
                    f"({exc}); encryption is unavailable and nothing else is "
                    f"affected")
        return _backend, _why


def available() -> bool:
    return backend()[0] != "none"


# -- keys ------------------------------------------------------------------

def load_key(path: str | None = None, *, env: str = "LMSLUICE_KEY_FILE") -> bytes:
    """A 32-byte key from a file, or from the file an environment variable names.

    Never from argv: on Linux every process can read another's command line out
    of `/proc`, so a key passed as a flag is a key handed to the machine. The
    variable holds a *path*, not the key, for the same reason -- an environment
    is inherited by children and visible in `/proc/<pid>/environ`.

    Accepts 32 raw bytes or 64 hex characters, and refuses anything else rather
    than hashing it into shape: silently accepting a short passphrase would give
    a file that looks encrypted and is not.
    """
    p = path or os.environ.get(env)
    if not p:
        raise CryptoUnavailable(
            f"no key: pass a key file, or set {env} to a path holding one. "
            f"Make one with:  lmsluice seal --make-key key.bin")
    with open(p, "rb") as fh:
        raw = fh.read(129)
    # Raw bytes are compared BEFORE any stripping, and that is not fussiness:
    # a 32-byte random key ends in a byte `bytes.strip()` treats as whitespace
    # about one time in twenty, and stripping it turns a valid key into a
    # 31-byte error. Found exactly that way, on the first key generated.
    if len(raw) == KEY_BYTES:
        return bytes(raw)
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        text = ""
    if len(text) == KEY_BYTES * 2:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    raise CryptoUnavailable(
        f"{p}: a key must be {KEY_BYTES} raw bytes or {KEY_BYTES * 2} hex "
        f"characters, and this file holds {len(raw)} bytes. It is not hashed into "
        f"shape on purpose -- a passphrase stretched here would look like a "
        f"key and protect far less.")


def new_key() -> bytes:
    """A fresh 32-byte key from the OS CSPRNG.

    `secrets`, not `random`: the latter is a Mersenne Twister whose state is
    recoverable from its output, which is fine for a shuffle and disqualifying
    for a key.
    """
    return secrets.token_bytes(KEY_BYTES)


def new_salt() -> bytes:
    return secrets.token_bytes(SALT_BYTES)


def _hkdf(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF-SHA256. Written out because `hashlib` has no HKDF and this is the
    one construction the standard library leaves to the caller; it is extract
    then expand, exactly as RFC 5869 states it."""
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]),
                         hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def subkey(key: bytes, salt: bytes) -> bytes:
    """This archive's own key. See the module docstring on counter nonces."""
    return _hkdf(key, salt, _INFO, KEY_BYTES)


def key_id(key: bytes) -> str:
    """A public identifier for a key, revealing nothing about it.

    Goes in the archive header so the wrong key is refused when the file is
    opened rather than on the first chunk, which is the difference between a
    clear error and a confusing one.
    """
    return hmac.new(key, _KEYID_INFO, hashlib.sha256).hexdigest()[:16]


def nonce_for(index: int) -> bytes:
    """The nonce for chunk `index` under an archive's subkey.

    A plain counter, which is sound *because* the key is per archive: GCM
    forbids reusing a nonce under one key, and a counter under a fresh key
    cannot repeat. Under a shared key this would be the classic catastrophic
    mistake, which is why `subkey` is not optional.
    """
    if index < 0 or index >= (1 << (NONCE_BYTES * 8)):
        raise ValueError(f"chunk index {index} out of range for a nonce")
    return index.to_bytes(NONCE_BYTES, "big")


# -- the cipher ------------------------------------------------------------

def _openssl_seal(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    ctx = _lib.EVP_CIPHER_CTX_new()
    if not ctx:
        raise CryptoUnavailable("EVP_CIPHER_CTX_new failed")
    try:
        if _lib.EVP_EncryptInit_ex(ctx, _lib.EVP_aes_256_gcm(), None,
                                   None, None) != 1:
            raise CryptoUnavailable("EVP_EncryptInit_ex failed")
        if _lib.EVP_CIPHER_CTX_ctrl(ctx, _SET_IVLEN, len(nonce), None) != 1:
            raise CryptoUnavailable("could not set the GCM nonce length")
        if _lib.EVP_EncryptInit_ex(ctx, None, None, key, nonce) != 1:
            raise CryptoUnavailable("EVP_EncryptInit_ex(key) failed")
        # Written into a bytearray rather than a ctypes string buffer, and
        # trimmed in place. `create_string_buffer` zeroes what it allocates and
        # `.raw` materialises the whole thing before the slice copies it again,
        # so the old form made three passes over memory for one pass of AES.
        # Invisible at 64 KiB, where it all sits in L2; at 8 MiB it was most of
        # the cost, and it is why the first sweep showed large chunks running
        # seven times slower than small ones.
        buf = bytearray(len(plaintext) + 16)
        out = (ctypes.c_char * len(buf)).from_buffer(buf)
        n = ctypes.c_int(0)
        if _lib.EVP_EncryptUpdate(ctx, out, ctypes.byref(n), plaintext,
                                  len(plaintext)) != 1:
            raise CryptoUnavailable("EVP_EncryptUpdate failed")
        total = n.value
        if _lib.EVP_EncryptFinal_ex(ctx, ctypes.byref(out, total),
                                    ctypes.byref(n)) != 1:
            raise CryptoUnavailable("EVP_EncryptFinal_ex failed")
        total += n.value
        tag = ctypes.create_string_buffer(TAG_BYTES)
        if _lib.EVP_CIPHER_CTX_ctrl(ctx, _GET_TAG, TAG_BYTES, tag) != 1:
            raise CryptoUnavailable("could not read the GCM tag")
        del out                       # the export has to go before the resize
        del buf[total:]
        buf += tag.raw[:TAG_BYTES]
        return bytes(buf)
    finally:
        _lib.EVP_CIPHER_CTX_free(ctx)


def _openssl_open(key: bytes, nonce: bytes, blob: bytes) -> bytes:
    # The ciphertext and its tag are addressed in place, not sliced out.
    # `blob[:-TAG_BYTES]` looks free and is a full copy of the chunk made in
    # Python, holding the GIL for its whole length -- with sixteen threads in
    # the fetch pool that copy is the serial section, and it showed up as
    # throughput falling as threads were added. ctypes passes a `bytes` object
    # as a pointer without copying, so the only thing needed is the address of
    # the tag, which is that pointer plus the body length.
    n_body = len(blob) - TAG_BYTES
    tag = ctypes.c_void_p(
        ctypes.cast(ctypes.c_char_p(blob), ctypes.c_void_p).value + n_body)
    ctx = _lib.EVP_CIPHER_CTX_new()
    if not ctx:
        raise CryptoUnavailable("EVP_CIPHER_CTX_new failed")
    try:
        if _lib.EVP_DecryptInit_ex(ctx, _lib.EVP_aes_256_gcm(), None,
                                   None, None) != 1:
            raise CryptoUnavailable("EVP_DecryptInit_ex failed")
        if _lib.EVP_CIPHER_CTX_ctrl(ctx, _SET_IVLEN, len(nonce), None) != 1:
            raise CryptoUnavailable("could not set the GCM nonce length")
        if _lib.EVP_DecryptInit_ex(ctx, None, None, key, nonce) != 1:
            raise CryptoUnavailable("EVP_DecryptInit_ex(key) failed")
        buf = bytearray(n_body + 16)
        out = (ctypes.c_char * len(buf)).from_buffer(buf)
        n = ctypes.c_int(0)
        if _lib.EVP_DecryptUpdate(ctx, out, ctypes.byref(n), blob,
                                  n_body) != 1:
            raise DecryptionFailed("EVP_DecryptUpdate failed")
        total = n.value
        if _lib.EVP_CIPHER_CTX_ctrl(ctx, _SET_TAG, TAG_BYTES, tag) != 1:
            raise CryptoUnavailable("could not set the GCM tag")
        # The tag is checked here, and a failure means the bytes are not
        # authentic. Nothing decrypted is returned in that case -- returning
        # "probably right" plaintext is how authenticated encryption gets
        # turned back into unauthenticated encryption by its caller.
        if _lib.EVP_DecryptFinal_ex(ctx, ctypes.byref(out, total),
                                    ctypes.byref(n)) != 1:
            raise DecryptionFailed(
                "authentication failed: wrong key, or these bytes were "
                "altered after they were written")
        # Only now, with the tag checked. A `bytearray` and not `bytes`: the
        # callers slice it, and the final copy is the one pass that can still
        # be skipped on a route that is already moving a checkpoint.
        del out
        del buf[total + n.value:]
        return buf
    finally:
        _lib.EVP_CIPHER_CTX_free(ctx)


def seal(key: bytes, index: int, plaintext) -> bytes:
    """Encrypt one chunk. Returns ciphertext followed by its 16-byte tag."""
    name, why = backend()
    if name == "none":
        raise CryptoUnavailable(why)
    data = bytes(plaintext)
    nonce = nonce_for(index)
    if name == "openssl":
        return _openssl_seal(key, nonce, data)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key).encrypt(nonce, data, None)


def open_chunk(key: bytes, index: int, blob) -> bytes:
    """Decrypt and authenticate one chunk, or raise. Never returns on failure."""
    name, why = backend()
    if name == "none":
        raise CryptoUnavailable(why)
    data = bytes(blob)
    if len(data) < TAG_BYTES:
        raise DecryptionFailed(f"chunk is {len(data)} bytes, shorter than a tag")
    nonce = nonce_for(index)
    if name == "openssl":
        return _openssl_open(key, nonce, data)
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        return AESGCM(key).decrypt(nonce, data, None)
    except InvalidTag as exc:
        raise DecryptionFailed(
            "authentication failed: wrong key, or these bytes were altered "
            "after they were written") from exc


def overhead(n_chunks: int) -> int:
    """Bytes an archive grows by: one tag a chunk. The salt is in the header."""
    return n_chunks * TAG_BYTES


def describe() -> str:
    name, why = backend()
    if name == "none":
        return f"encryption unavailable: {why}"
    return f"AES-256-GCM via {name} ({why})"
