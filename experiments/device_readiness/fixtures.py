"""Small deterministic artifacts for the portable-readiness contract.

The generated safetensors file is a transport fixture. It has named language,
speech and vision roles so route/consumer accounting can be exercised without
claiming model quality. Its bytes are stable across runs and are described by
``docs/evaluations/workload.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct


FIXTURE_VERSION = "portable-device-readiness-fixture-v1"


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def deterministic_bytes(label: str, count: int, seed: int = 7) -> bytes:
    """Expand a stable hash stream without using platform RNG state."""
    out = bytearray()
    counter = 0
    while len(out) < count:
        out.extend(hashlib.sha256(
            f"{FIXTURE_VERSION}:{seed}:{label}:{counter}".encode()).digest())
        counter += 1
    return bytes(out[:count])


def bf16_like(label: str, count: int, seed: int = 7) -> bytes:
    """Produce deterministic BF16 words with a weight-like exponent band."""
    raw = deterministic_bytes(label, count * 2, seed)
    out = bytearray(raw)
    for i in range(count):
        # Keep several nearby exponents while retaining deterministic mantissa
        # noise. This is representative for compression mechanics only.
        exponent = 0x3D + ((i * 7) // max(1, count // 6)) % 4
        out[2 * i + 1] = ((exponent << 1) | (raw[2 * i + 1] & 1)) & 0xFF
    return bytes(out)


def write_safetensors(path: str, entries: list[dict], *, seed: int = 7) -> dict:
    """Write a canonical, minimal safetensors-like artifact and return metadata."""
    header = {
        "__metadata__": {
            "fixture": FIXTURE_VERSION,
            "seed": str(seed),
            "roles": "language,asr,tts,vision,specialist",
        }
    }
    blobs = []
    offset = 0
    for entry in entries:
        blob = entry["bytes"]
        header[entry["name"]] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)
    return {
        "path": os.path.basename(path),
        "sha256": hash_file(path),
        "bytes": os.path.getsize(path),
        "header_bytes": len(encoded) + 8,
        "tensor_count": len(entries),
        "tensors": [
            {"name": e["name"], "role": e["role"], "dtype": e["dtype"],
             "shape": list(e["shape"]), "bytes": len(e["bytes"]),
             "input_sha256": hashlib.sha256(e["bytes"]).hexdigest(),
             "consumer_payload_sha256": hashlib.sha256(e["bytes"]).hexdigest()}
            for e in entries
        ],
    }


def generate_bundle(directory: str, *, seed: int = 7, chunk_size: int = 32 << 10) -> dict:
    """Generate the frozen synthetic bundle and optional stdlib-coded copy."""
    os.makedirs(directory, exist_ok=True)
    entries = [
        {"name": "language.embed.weight", "role": "language",
         "dtype": "BF16", "shape": (48, 128),
         "bytes": bf16_like("language", 48 * 128, seed)},
        {"name": "asr.encoder.weight", "role": "speech-in",
         "dtype": "F32", "shape": (16, 64),
         "bytes": deterministic_bytes("asr", 16 * 64 * 4, seed)},
        {"name": "tts.decoder.weight", "role": "speech-out",
         "dtype": "F16", "shape": (24, 64),
         "bytes": deterministic_bytes("tts", 24 * 64 * 2, seed)},
        {"name": "vision.patch.weight", "role": "vision",
         "dtype": "U8", "shape": (64, 96),
         "bytes": deterministic_bytes("vision", 64 * 96, seed)},
        {"name": "specialist.task.weight", "role": "specialist",
         "dtype": "BF16", "shape": (80, 128),
         "bytes": bf16_like("specialist", 80 * 128, seed)},
        {"name": "oversized.tensor", "role": "memory-edge",
         "dtype": "U8", "shape": (96 << 10,),
         "bytes": deterministic_bytes("oversized", 96 << 10, seed)},
    ]
    plain = os.path.join(directory, "portable-workload.safetensors")
    plain_meta = write_safetensors(plain, entries, seed=seed)

    coded = os.path.join(directory, "portable-workload.lmsluice")
    coded_meta = None
    try:
        from lmsluice.zstdcodec import encoder

        measured = encoder().encode(plain, coded, chunk_size=chunk_size, level=1)
        coded_meta = {
            "path": os.path.basename(coded),
            "sha256": hash_file(coded),
            "bytes": os.path.getsize(coded),
            "format": "lmsluice-stdlib-coded",
            "chunk_size": chunk_size,
            "measured": measured,
        }
    except Exception as exc:  # pragma: no cover - interpreter capability varies
        coded_meta = {
            "path": os.path.basename(coded),
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    low_compress = os.path.join(directory, "low-compressibility.bin")
    with open(low_compress, "wb") as fh:
        fh.write(deterministic_bytes("low-compressibility", 48 << 10, seed))
    return {
        "version": FIXTURE_VERSION,
        "seed": seed,
        "plain": plain_meta,
        "coded": coded_meta,
        "low_compressibility": {
            "path": os.path.basename(low_compress),
            "sha256": hash_file(low_compress),
            "bytes": os.path.getsize(low_compress),
            "format": "opaque-binary",
        },
    }


__all__ = ["FIXTURE_VERSION", "deterministic_bytes", "generate_bundle",
           "hash_file", "write_safetensors"]
