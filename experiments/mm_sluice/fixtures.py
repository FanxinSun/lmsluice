"""Tiny deterministic complete-bundle fixtures.

The ONNX bytes are encoded with the ONNX protobuf wire format directly so the
probe has no dependency on the optional ``onnx`` package.  The graph is a
small ``Add`` model with one external FLOAT initializer.  It is a packaging
and consumer-boundary fixture; its output is not a speech, vision or language
quality result.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

from lmsluice.bundle import canonical_manifest_sha256, make_manifest, write_manifest


FIXTURE_VERSION = "mm-sluice-01-complete-bundle-v1"
SEED = 17


def deterministic_bytes(label: str, count: int, seed: int = SEED) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < count:
        out.extend(hashlib.sha256(
            f"{FIXTURE_VERSION}:{seed}:{label}:{counter}".encode("utf-8")
        ).digest())
        counter += 1
    return bytes(out[:count])


def _varint(value: int) -> bytes:
    value = int(value)
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _vfield(number: int, value: int) -> bytes:
    return _varint((number << 3) | 0) + _varint(value)


def _bfield(number: int, value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _onnx_dim(value: int) -> bytes:
    return _vfield(1, value)


def _onnx_tensor_shape(dims: tuple[int, ...]) -> bytes:
    return b"".join(_bfield(1, _onnx_dim(dim)) for dim in dims)


def _onnx_tensor_type(elem_type: int, dims: tuple[int, ...]) -> bytes:
    return _vfield(1, elem_type) + _bfield(2, _onnx_tensor_shape(dims))


def _onnx_value_info(name: str, elem_type: int = 1, dims=(1,)) -> bytes:
    type_proto = _bfield(1, _onnx_tensor_type(elem_type, tuple(dims)))
    return _bfield(1, name) + _bfield(2, type_proto)


def _onnx_external_entry(key: str, value: str) -> bytes:
    return _bfield(1, key) + _bfield(2, value)


def _onnx_external_tensor(name: str) -> bytes:
    # TensorProto: dims=1, data_type=FLOAT(1), name=8,
    # external_data=13, data_location=14(EXTERNAL=1).
    external = _bfield(13, _onnx_external_entry("location", "weights.bin"))
    external += _bfield(13, _onnx_external_entry("offset", "3"))
    external += _bfield(13, _onnx_external_entry("length", "4"))
    return _vfield(1, 1) + _vfield(2, 1) + _bfield(8, name) + external + _vfield(14, 1)


def onnx_add_external_graph() -> bytes:
    """Return a valid small ONNX ModelProto with external initializer data."""
    node = (_bfield(1, "input") + _bfield(1, "weight") +
            _bfield(2, "output") + _bfield(3, "add") + _bfield(4, "Add"))
    graph = (_bfield(1, node) + _bfield(2, "mm_sluice_add") +
             _bfield(5, _onnx_external_tensor("weight")) +
             _bfield(11, _onnx_value_info("input")) +
             _bfield(12, _onnx_value_info("output")))
    opset = _vfield(2, 13)
    # ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8.
    return (_vfield(1, 8) + _bfield(2, "lmsluice-mm-sluice-01") +
            _bfield(7, graph) + _bfield(8, opset))


def _wire_varint(data: bytes, offset: int):
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise ValueError("truncated protobuf varint")
        item = data[offset]
        offset += 1
        value |= (item & 0x7F) << shift
        if item < 0x80:
            return value, offset
        shift += 7


def _wire_fields(data: bytes):
    fields = []
    offset = 0
    while offset < len(data):
        tag, offset = _wire_varint(data, offset)
        number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            value, offset = _wire_varint(data, offset)
        elif wire_type == 2:
            length, offset = _wire_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes field")
            value, offset = data[offset:end], end
        else:
            raise ValueError(f"unsupported fixture protobuf wire type {wire_type}")
        fields.append((number, wire_type, value))
    return fields


def audit_onnx_graph(graph_path: str, weights_path: str) -> dict:
    """Check the generated graph's small external-data contract without onnx."""
    with open(graph_path, "rb") as fh:
        model = _wire_fields(fh.read())
    graph = next(value for number, wire_type, value in model
                 if number == 7 and wire_type == 2)
    node = next(value for number, wire_type, value in _wire_fields(graph)
                if number == 1 and wire_type == 2)
    tensor = next(value for number, wire_type, value in _wire_fields(graph)
                  if number == 5 and wire_type == 2)
    node_fields = _wire_fields(node)
    operator = next(value.decode("utf-8") for number, wire_type, value in node_fields
                    if number == 4 and wire_type == 2)
    external = {}
    for item in _wire_fields(tensor):
        if item[0] != 13 or item[1] != 2:
            continue
        entry = _wire_fields(item[2])
        key = next(value.decode("utf-8") for number, wire_type, value in entry
                   if number == 1 and wire_type == 2)
        value = next(value.decode("utf-8") for number, wire_type, value in entry
                     if number == 2 and wire_type == 2)
        external[key] = value
    location = external.get("location")
    offset = int(external.get("offset", "-1"))
    length = int(external.get("length", "-1"))
    data_location = next(value for number, wire_type, value in _wire_fields(tensor)
                         if number == 14 and wire_type == 0)
    if operator != "Add" or location != "weights.bin" or offset != 3 or length != 4:
        raise ValueError("generated ONNX external-data contract is inconsistent")
    if data_location != 1:
        raise ValueError("generated ONNX tensor is not marked EXTERNAL")
    with open(weights_path, "rb") as fh:
        payload = fh.read()
    if len(payload[offset:offset + length]) != length:
        raise ValueError("generated ONNX external-data range is truncated")
    value = struct.unpack("<f", payload[offset:offset + length])[0]
    if value != 1.0:
        raise ValueError("generated ONNX external-data value changed")
    return {
        "operator": operator,
        "external_location": location,
        "external_offset": offset,
        "external_length": length,
        "external_float": value,
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def generate_bundle(directory: str, *, seed: int = SEED) -> dict:
    """Generate a complete graph/sidecar/config/preprocess/state fixture."""
    root = os.path.abspath(directory)
    os.makedirs(os.path.join(root, "backend"), exist_ok=True)
    graph = os.path.join(root, "model.onnx")
    with open(graph, "wb") as fh:
        fh.write(onnx_add_external_graph())
    # The graph consumes bytes [3, 7) of this file: one FLOAT equal to 1.0.
    weights = os.path.join(root, "weights.bin")
    with open(weights, "wb") as fh:
        fh.write(b"HDR")
        fh.write(struct.pack("<f", 1.0))
        fh.write(b"TRAILER")
    files = {
        "config.json": {
            "model_family": "synthetic-add",
            "dtype": "float32",
            "generated_seed": seed,
        },
        "preprocess.json": {
            "audio": {"sample_rate_hz": 16000, "channels": 1, "units": "PCM16"},
            "image": {"layout": "NCHW", "color": "RGB", "normalization": "[0,1]"},
        },
        "vocabulary.txt": "<blank>\nsynthetic\n",
        "calibration.json": {"coordinate_frame": "fixture", "version": "v1"},
    }
    for relative, value in files.items():
        with open(os.path.join(root, relative), "w", encoding="utf-8") as fh:
            if isinstance(value, str):
                fh.write(value)
            else:
                json.dump(value, fh, sort_keys=True, separators=(",", ":"))
                fh.write("\n")
    opaque = os.path.join(root, "backend", "incompressible.bin")
    with open(opaque, "wb") as fh:
        fh.write(deterministic_bytes("opaque", 257, seed))
    entries = [
        {"path": "model.onnx", "role": "graph",
         "dependencies": [{"path": "weights.bin", "offset": 3, "length": 4}]},
        {"path": "weights.bin", "role": "weights"},
        {"path": "config.json", "role": "config"},
        {"path": "preprocess.json", "role": "preprocess"},
        {"path": "vocabulary.txt", "role": "vocabulary"},
        {"path": "calibration.json", "role": "calibration"},
        {"path": "backend/incompressible.bin", "role": "opaque",
         "consumer": {"engine": "onnxruntime", "backend": "CPUExecutionProvider"}},
    ]
    manifest = make_manifest(
        root, entries, schema="1.0", entry_point="model.onnx",
        consumer={
            "engine": "onnxruntime",
            "backend": "CPUExecutionProvider",
            "version_range": ">=1.16,<2",
            "operators": ["Add"],
            "extensions": [],
            "code_loading": "disabled",
        },
        resources={
            "decode_workspace_bytes": {"value": 4096, "method": "declared", "status": "ESTIMATED"},
            "materialized_bytes": {"value": 0, "method": "measured_after_validation", "status": "MEASURED"},
        },
        provenance={
            "kind": "synthetic_generated",
            "fixture_version": FIXTURE_VERSION,
            "seed": seed,
            "source_revision": "local-generated",
            "license": "repository-generated; no external model or dataset",
        },
        io={
            "inputs": [{"name": "input", "dtype": "float32", "shape": [1]}],
            "outputs": [{"name": "output", "dtype": "float32", "shape": [1]}],
        },
        preprocessing={
            "audio": {"status": "schema_example", "sample_rate_hz": 16000},
            "visual": {"status": "schema_example", "layout": "NCHW", "color": "RGB"},
        },
        state={
            "inputs": [], "outputs": [], "initialization": "zero",
            "reset_on": ["cancellation", "model_change", "discontinuity"],
            "cadence": "caller_defined", "max_bytes": 0,
        },
        evaluation={
            "input": {"kind": "synthetic", "x": 2.0, "sha256": hashlib.sha256(struct.pack("<f", 2.0)).hexdigest()},
            "expected_output": {"kind": "synthetic", "value": 3.0,
                                "tolerance": 0.0, "quality": "UNAVAILABLE"},
        },
    )
    manifest_path = write_manifest(os.path.join(root, "bundle.json"), manifest)
    payload = manifest["bundle"]
    entry_meta = []
    for entry in payload["entries"]:
        entry_meta.append({**entry, "actual_path": os.path.join(root, entry["path"])})
    total_bytes = sum(entry["length"] for entry in payload["entries"])
    return {
        "fixture_version": FIXTURE_VERSION,
        "seed": seed,
        "source_root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "entries": entry_meta,
        "graph_path": graph,
        "weights_path": weights,
        "total_bytes": total_bytes,
        "input": {"input": [2.0]},
        "expected_output": [3.0],
        "file_hashes": {entry["path"]: file_sha256(os.path.join(root, entry["path"]))
                        for entry in payload["entries"]},
    }


__all__ = ["FIXTURE_VERSION", "SEED", "audit_onnx_graph",
           "deterministic_bytes", "file_sha256", "generate_bundle",
           "onnx_add_external_graph"]
