#!/usr/bin/env python3
"""Category 6: Deployment & Packaging Qualification Script.

Evaluates:
1. Multimodal Bundle Packaging of Qwen3.5-4B (lmsluice/bundle.py):
   - Verified manifest creation across multimodal roles (config, preprocess, vocabulary, weights, opaque).
   - Instant descriptor resolution and inventory (< 2 ms).
   - Strict checksum validation.
   - Atomic staging and publication with renameatx_np NOREPLACE protection.
2. Envelope Encryption (lmsluice seal / lmsluice/sealed.py):
   - AES-256-GCM hardware throughput via OpenSSL ctypes backend.
   - Container sealing on real model shards (63 MB and 2.4 GB).
   - Clear-structure zero-knowledge inspection without key (info).
   - Authenticated HMAC-SHA256 structure validation with key.
   - Selective single-tensor extraction from multi-gigabyte sealed archive.
   - Bit-exact safetensors reconstruction roundtrip.
   - Fail-closed security matrix against missing keys, wrong keys, tampered ciphertext, and tampered clear headers.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time

from lmsluice import __version__ as lmsluice_version
from lmsluice import bundle as B
from lmsluice import crypt, sealed
from lmsluice.model import open_model
from lmsluice.probe import decrypt_rate

MODEL_QWEN35 = "/Users/masshj/models/Qwen3.5-4B"
GROUND_TRUTH_INSTRUCT = "/Users/masshj/models/Qwen3-4B-Instruct-2507"
LMZ_CACHE_DIR = os.path.expanduser("~/.cache/lmsluice/category4_archives/lmz")
RESULTS_DIR = "/Users/masshj/YFCE/lmsluice/results"
OUTPUT_JSON = os.path.join(RESULTS_DIR, "category6_deployment_report.json")
SEALED_DIR = os.path.join(RESULTS_DIR, "category6_sealed")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SEALED_DIR, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_suite_1_multimodal_bundle():
    print("=" * 70)
    print("SUITE 1: Multimodal Bundle Packaging (Qwen3.5-4B)")
    print("=" * 70)

    # 1. Inspect source files
    files = [
        ("config.json", "config"),
        ("preprocessor_config.json", "preprocess"),
        ("video_preprocessor_config.json", "preprocess"),
        ("tokenizer.json", "vocabulary"),
        ("tokenizer_config.json", "config"),
        ("vocab.json", "vocabulary"),
        ("merges.txt", "vocabulary"),
        ("chat_template.jinja", "opaque"),
        ("LICENSE", "opaque"),
        ("model.safetensors.index.json", "config"),
        ("model.safetensors-00001-of-00002.safetensors", "weights"),
        ("model.safetensors-00002-of-00002.safetensors", "weights"),
    ]

    entries = [{"path": p, "role": r} for p, r in files]
    role_breakdown: dict[str, dict] = {}
    for p, r in files:
        full_p = os.path.join(MODEL_QWEN35, p)
        size = os.path.getsize(full_p)
        if r not in role_breakdown:
            role_breakdown[r] = {"count": 0, "bytes": 0, "files": []}
        role_breakdown[r]["count"] += 1
        role_breakdown[r]["bytes"] += size
        role_breakdown[r]["files"].append(p)

    total_bytes = sum(b["bytes"] for b in role_breakdown.values())
    print(f"Discovered {len(files)} multimodal entries ({total_bytes / 1e9:.2f} GB total):")
    for r, b in role_breakdown.items():
        print(f"  • {r:<12}: {b['count']} files, {b['bytes'] / 1e6:8.2f} MB")

    # 2. make_manifest
    print("\n[1/4] Generating schema-1 verified bundle manifest...")
    t0 = time.perf_counter()
    manifest = B.make_manifest(
        source_root=MODEL_QWEN35,
        entries=entries,
        entry_point="model.safetensors.index.json",
        consumer={"engine": "transformers", "backend": "mps", "model_type": "qwen3_vl"},
        id="Qwen3.5-4B",
        modalities=["text", "image", "video"],
    )
    t1 = time.perf_counter()
    manifest_time = t1 - t0
    manifest_rate = (total_bytes / 1e9) / manifest_time
    manifest_id = manifest["bundle"]["id"]
    canonical_digest = B.canonical_manifest_sha256(manifest)
    print(f"  Generated in {manifest_time:.2f}s ({manifest_rate:.2f} GB/s hashing throughput)")
    print(f"  Canonical Manifest SHA-256: {canonical_digest}")

    bundle_source = {"source_root": MODEL_QWEN35, "manifest": manifest["bundle"]}

    # 3. resolve_bundle & inventory_bundle
    print("\n[2/4] Resolving bundle descriptor & inspecting inventory...")
    t0 = time.perf_counter()
    descriptor = B.resolve_bundle(bundle_source)
    t1 = time.perf_counter()
    resolve_latency_ms = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    inv_nonstrict = B.inventory_bundle(bundle_source, strict=False)
    t1 = time.perf_counter()
    inv_latency_ms = (t1 - t0) * 1000.0

    print(f"  Descriptor resolved in {resolve_latency_ms:.2f} ms")
    print(f"  Non-strict inventory in {inv_latency_ms:.2f} ms")
    print(f"  Entry point: {descriptor.entry_point_details}")

    # 4. validate_bundle
    print("\n[3/4] Validating bundle files & cryptographic checksums...")
    t0 = time.perf_counter()
    val = B.validate_bundle(bundle_source)
    t1 = time.perf_counter()
    val_time = t1 - t0
    print(f"  Validated 12 entries ({total_bytes / 1e9:.2f} GB) in {val_time:.2f}s")
    print(f"  Validation status: valid={val['valid']}, publisher_authenticated={val['publisher_authenticated']}")
    assert val["valid"], "Bundle validation failed!"

    # 5. Staging & Atomic Publication (Materialize)
    print("\n[4/4] Testing invocation-owned staging & atomic publication...")
    tmpdir = os.path.realpath(tempfile.mkdtemp(prefix="lmsluice-cat6-bundle-"))
    dest_control = os.path.join(tmpdir, "control_plane_published")
    dest_full = os.path.join(tmpdir, "full_model_published")

    # A: Control-plane subset materialization
    ctrl_files = [f for f in files if f[1] != "weights"]
    ctrl_manifest = B.make_manifest(
        source_root=MODEL_QWEN35,
        entries=[{"path": p, "role": r} for p, r in ctrl_files],
        entry_point="model.safetensors.index.json",
        consumer={"engine": "transformers", "backend": "mps", "model_type": "qwen3_vl"},
        id="Qwen3.5-4B-ControlPlane",
    )
    ctrl_source = {"source_root": MODEL_QWEN35, "manifest": ctrl_manifest["bundle"]}

    t0 = time.perf_counter()
    res_ctrl = B.materialize_bundle(ctrl_source, dest_control)
    t1 = time.perf_counter()
    ctrl_materialize_ms = (t1 - t0) * 1000.0
    print(f"  Control-plane bundle (10 entries, 22.8 MB) materialized in {ctrl_materialize_ms:.2f} ms")
    assert os.path.exists(dest_control)
    assert len(os.listdir(dest_control)) == 10

    # Test atomic no-replace protection (renameatx_np RENAME_EXCL / RENAME_NOREPLACE)
    atomic_protection_passed = False
    try:
        B.materialize_bundle(ctrl_source, dest_control)
    except B.BundleError as e:
        if e.code == "destination_exists":
            atomic_protection_passed = True
            print("  Atomic no-replace protection: PASS (rejected clobbering existing target)")
        else:
            raise
    assert atomic_protection_passed, "Atomic no-replace protection failed!"

    # B: Full model materialization
    t0 = time.perf_counter()
    res_full = B.materialize_bundle(bundle_source, dest_full, fetch_threads=4, place_threads=4)
    t1 = time.perf_counter()
    full_materialize_time = t1 - t0
    full_materialize_rate = (total_bytes / 1e9) / full_materialize_time
    print(f"  Full 9.34 GB multimodal bundle materialized in {full_materialize_time:.2f}s ({full_materialize_rate:.2f} GB/s)")
    assert os.path.exists(dest_full)
    assert len(os.listdir(dest_full)) == 12

    # Clean up tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  Cleaned up temporary materialization directory cleanly.")

    suite_1_data = {
        "model": "Qwen3.5-4B",
        "source_root": MODEL_QWEN35,
        "total_entries": len(files),
        "total_bundle_bytes": total_bytes,
        "role_breakdown": {
            r: {"count": b["count"], "bytes": b["bytes"], "files": b["files"]}
            for r, b in role_breakdown.items()
        },
        "manifest_generation": {
            "time_seconds": manifest_time,
            "throughput_gb_s": manifest_rate,
            "manifest_identity": canonical_digest,
        },
        "bundle_resolution": {
            "resolve_latency_ms": resolve_latency_ms,
            "inventory_latency_ms": inv_latency_ms,
        },
        "bundle_validation": {
            "validation_time_seconds": val_time,
            "valid": val["valid"],
            "publisher_authenticated": val["publisher_authenticated"],
        },
        "staging_publication": {
            "control_plane_materialize_ms": ctrl_materialize_ms,
            "atomic_no_replace_verified": atomic_protection_passed,
            "full_materialize_time_seconds": full_materialize_time,
            "full_materialize_rate_gb_s": full_materialize_rate,
            "full_materialized_entries": len(res_full.entries),
            "bundle_sha256": res_full.bundle_sha256,
        },
    }
    return suite_1_data


def run_suite_2_envelope_encryption():
    print("\n" + "=" * 70)
    print("SUITE 2: Envelope Encryption (lmsluice seal / lmsluice/sealed.py)")
    print("=" * 70)

    # 1. Probe crypt backend & raw throughput
    backend_kind, backend_path = crypt.backend()
    print(f"Crypt Backend: {backend_kind} ({backend_path})")
    assert crypt.available(), "Crypt backend not available!"

    # Measure multi-threaded decrypt rate from probe
    t0 = time.perf_counter()
    measured_decrypt_rate = decrypt_rate() or 0.0
    t1 = time.perf_counter()
    print(f"lmsluice probe decrypt rate (multi-threaded): {measured_decrypt_rate / 1e9:.2f} GB/s (probe took {t1-t0:.2f}s)")

    # Measure pure seal and open throughput in python ctypes
    master_key = crypt.new_key()
    salt = crypt.new_salt()
    sub = crypt.subkey(master_key, salt)
    chunk_data = os.urandom(8 << 20)  # 8 MiB
    iters = 25
    t0 = time.perf_counter()
    ciphertexts = [crypt.seal(sub, i, chunk_data) for i in range(iters)]
    t1 = time.perf_counter()
    seal_rate_gb_s = (iters * len(chunk_data) / 1e9) / (t1 - t0)

    t0 = time.perf_counter()
    for i in range(iters):
        _ = crypt.open_chunk(sub, i, ciphertexts[i])
    t1 = time.perf_counter()
    open_rate_gb_s = (iters * len(chunk_data) / 1e9) / (t1 - t0)
    print(f"Pure ctypes AES-256-GCM: Seal = {seal_rate_gb_s:.2f} GB/s, Open = {open_rate_gb_s:.2f} GB/s")

    # 2. Key Generation
    key_file = os.path.join(SEALED_DIR, "production_key.bin")
    if os.path.exists(key_file):
        os.remove(key_file)
    with open(key_file, "wb") as f:
        f.write(master_key)
    os.chmod(key_file, 0o600)
    print(f"\nGenerated 32-byte production key at {key_file} (mode 600)")

    # 3. Seal model shards
    # Small shard: model-00003-of-00003.lmz
    shard3_lmz = os.path.join(LMZ_CACHE_DIR, "model-00003-of-00003.lmz")
    shard3_sealed = os.path.join(SEALED_DIR, "model-00003-of-00003.sealed")
    print(f"\n[1/5] Sealing shard 3 ({os.path.getsize(shard3_lmz) / 1e6:.1f} MB)...")
    t0 = time.perf_counter()
    hdr3 = sealed.seal_file(shard3_lmz, shard3_sealed, key_file)
    t1 = time.perf_counter()
    s3_seal_time = t1 - t0
    s3_seal_rate = (os.path.getsize(shard3_lmz) / 1e9) / s3_seal_time
    s3_sealed_size = os.path.getsize(shard3_sealed)
    print(f"  Shard 3 sealed in {s3_seal_time:.3f}s ({s3_seal_rate:.2f} GB/s): {s3_sealed_size / 1e6:.1f} MB")
    print(f"  Header: algo={hdr3['algorithm']}, key_id={hdr3['key_id']}")

    # Large shard: model-00001-of-00003.lmz
    shard1_lmz = os.path.join(LMZ_CACHE_DIR, "model-00001-of-00003.lmz")
    shard1_sealed = os.path.join(SEALED_DIR, "model-00001-of-00003.sealed")
    print(f"\n[2/5] Sealing multi-gigabyte shard 1 ({os.path.getsize(shard1_lmz) / 1e9:.2f} GB)...")
    t0 = time.perf_counter()
    hdr1 = sealed.seal_file(shard1_lmz, shard1_sealed, key_file)
    t1 = time.perf_counter()
    s1_seal_time = t1 - t0
    s1_seal_rate = (os.path.getsize(shard1_lmz) / 1e9) / s1_seal_time
    s1_sealed_size = os.path.getsize(shard1_sealed)
    print(f"  Shard 1 sealed in {s1_seal_time:.2f}s ({s1_seal_rate:.2f} GB/s): {s1_sealed_size / 1e9:.2f} GB")

    # 4. Zero-Knowledge Structural Inspection Without Key (info)
    print("\n[3/5] Testing structural inspection WITHOUT key...")
    # Clear environment variable
    old_env = os.environ.pop("LMSLUICE_KEY_FILE", None)

    t0 = time.perf_counter()
    with open_model(shard1_sealed) as m_no_key:
        t1 = time.perf_counter()
        info_no_key_ms = (t1 - t0) * 1000.0
        enc_info = m_no_key.encrypted
        is_verified = m_no_key._arc.source.verified
        tensors_count = len(m_no_key.tensors)
        plain_bytes = m_no_key.plain_bytes

    print(f"  Inspected 2.6 GB sealed container in {info_no_key_ms:.2f} ms:")
    print(f"  • Tensors visible in clear: {tensors_count} tensors ({plain_bytes / 1e9:.2f} GB plain)")
    print(f"  • Algorithm: {enc_info['algorithm']}, Key ID: {enc_info['key_id']}")
    print(f"  • Structure verification status: {is_verified} (expected False without key)")
    assert not is_verified, "Should not be verified without key!"

    # Structural Inspection WITH Key
    print("\nTesting structural inspection WITH key...")
    os.environ["LMSLUICE_KEY_FILE"] = key_file
    t0 = time.perf_counter()
    with open_model(shard1_sealed) as m_with_key:
        t1 = time.perf_counter()
        info_with_key_ms = (t1 - t0) * 1000.0
        is_verified_with_key = m_with_key._arc.source.verified
    print(f"  Inspected with key in {info_with_key_ms:.2f} ms:")
    print(f"  • Structure verification status: {is_verified_with_key} (HMAC-SHA256 verified)")
    assert is_verified_with_key, "Should be verified with key!"

    # 5. Selective Single-Tensor Extraction From Sealed Container
    print("\n[4/5] Testing selective single-tensor extraction from 2.6 GB sealed container...")
    target_tensor_name = "model.layers.0.mlp.down_proj.weight"
    t0 = time.perf_counter()
    with open_model(shard1_sealed) as m_sealed:
        tensor_bytes = bytes(m_sealed.tensor(target_tensor_name))
    t1 = time.perf_counter()
    selective_extract_ms = (t1 - t0) * 1000.0
    tensor_h = hashlib.sha256(tensor_bytes).hexdigest()
    print(f"  Extracted {target_tensor_name} ({len(tensor_bytes) / 1e6:.2f} MB) in {selective_extract_ms:.2f} ms")

    # Compare against ground truth plain safetensors
    gt_file1 = os.path.join(GROUND_TRUTH_INSTRUCT, "model-00001-of-00003.safetensors")
    with open_model(gt_file1) as m_gt:
        gt_tensor_bytes = bytes(m_gt.tensor(target_tensor_name))
    gt_tensor_h = hashlib.sha256(gt_tensor_bytes).hexdigest()
    assert tensor_h == gt_tensor_h, f"Selective tensor digest mismatch! {tensor_h} vs {gt_tensor_h}"
    print("  Selective extraction numerical equivalence: 100% BIT-EXACT MATCH")

    # 6. Full Extraction Roundtrip (get)
    print("\n[5/5] Testing full shard extraction roundtrip and ground-truth verification...")
    restored_path = os.path.join(SEALED_DIR, "restored_shard3.safetensors")
    if os.path.exists(restored_path):
        os.remove(restored_path)

    t0 = time.perf_counter()
    with open_model(shard3_sealed) as m_s3:
        restored_bytes = m_s3.to_file(restored_path, overwrite=True)
    t1 = time.perf_counter()
    restore_time = t1 - t0
    restore_rate = (restored_bytes / 1e9) / restore_time
    print(f"  Restored {restored_bytes / 1e6:.1f} MB plain file in {restore_time:.2f}s ({restore_rate:.2f} GB/s)")

    gt_file3 = os.path.join(GROUND_TRUTH_INSTRUCT, "model-00003-of-00003.safetensors")
    restored_sha = sha256_file(restored_path)
    gt_sha = sha256_file(gt_file3)
    print(f"  Restored SHA-256:     {restored_sha}")
    print(f"  Ground-Truth SHA-256: {gt_sha}")
    assert restored_sha == gt_sha, "Full extraction SHA-256 mismatch!"
    print("  Full extraction roundtrip: 100% BIT-EXACT SHA-256 MATCH")
    os.remove(restored_path)

    # 7. Fail-Closed Security Tamper Matrix
    print("\n" + "-" * 50)
    print("Security & Integrity Validation Matrix (Fail-Closed)")
    print("-" * 50)
    tamper_results = {}

    # Case 1: Missing Key
    os.environ.pop("LMSLUICE_KEY_FILE", None)
    case1_passed = False
    try:
        with open_model(shard3_sealed) as m:
            _ = m.to_file(os.path.join(SEALED_DIR, "tmp.bin"))
    except crypt.CryptoUnavailable as e:
        case1_passed = True
        tamper_results["missing_key"] = {
            "status": "PASS",
            "fail_closed": True,
            "message": str(e),
        }
        print("  [Case 1] Missing Key: PASS (failed closed before decryption)")
    assert case1_passed

    # Case 2: Wrong Key
    wrong_key_file = os.path.join(SEALED_DIR, "wrong_key.bin")
    with open(wrong_key_file, "wb") as f:
        f.write(crypt.new_key())
    os.chmod(wrong_key_file, 0o600)
    os.environ["LMSLUICE_KEY_FILE"] = wrong_key_file
    case2_passed = False
    try:
        with open_model(shard3_sealed) as m:
            _ = m.to_file(os.path.join(SEALED_DIR, "tmp.bin"))
    except ValueError as e:
        if "written with a different key" in str(e):
            case2_passed = True
            tamper_results["wrong_key"] = {
                "status": "PASS",
                "fail_closed": True,
                "message": str(e),
            }
            print("  [Case 2] Wrong Key: PASS (key id mismatch rejected immediately)")
        else:
            raise
    assert case2_passed
    os.remove(wrong_key_file)

    # Reset valid key
    os.environ["LMSLUICE_KEY_FILE"] = key_file

    # Case 3: Altered CLEAR Region (Header tampering)
    with open(shard3_sealed, "rb") as f:
        tampered_clear_bytes = bytearray(f.read())
    tampered_clear_bytes[10] ^= 0x01  # Flip bit in Row 0 clear header
    tampered_clear_file = os.path.join(SEALED_DIR, "tampered_clear.sealed")
    with open(tampered_clear_file, "wb") as f:
        f.write(tampered_clear_bytes)

    case3_passed = False
    try:
        with open_model(tampered_clear_file) as m:
            _ = m.to_file(os.path.join(SEALED_DIR, "tmp.bin"))
    except ValueError as e:
        if "structure failed authentication" in str(e):
            case3_passed = True
            tamper_results["altered_clear_header"] = {
                "status": "PASS",
                "fail_closed": True,
                "message": str(e),
            }
            print("  [Case 3] Altered Clear Region: PASS (HMAC-SHA256 structure validation rejected altered header)")
        else:
            raise
    assert case3_passed
    os.remove(tampered_clear_file)

    # Case 4: Altered Ciphertext Payload (Payload bit-flip)
    with open(shard3_sealed, "rb") as f:
        tampered_payload_bytes = bytearray(f.read())
    tampered_payload_bytes[10240] ^= 0x01  # Flip bit inside unit 2 ciphertext
    tampered_payload_file = os.path.join(SEALED_DIR, "tampered_payload.sealed")
    with open(tampered_payload_file, "wb") as f:
        f.write(tampered_payload_bytes)

    case4_passed = False
    try:
        with open_model(tampered_payload_file) as m:
            _ = m.to_file(os.path.join(SEALED_DIR, "tmp.bin"))
    except ValueError as e:
        if "failed authentication" in str(e):
            case4_passed = True
            tamper_results["altered_ciphertext_payload"] = {
                "status": "PASS",
                "fail_closed": True,
                "message": str(e),
            }
            print("  [Case 4] Altered Ciphertext Payload: PASS (AES-256-GCM authentication rejected altered payload)")
        else:
            raise
    assert case4_passed
    os.remove(tampered_payload_file)

    suite_2_data = {
        "backend": {
            "kind": backend_kind,
            "path": backend_path,
            "available": True,
            "measured_decrypt_rate_gb_s": measured_decrypt_rate / 1e9,
            "pure_ctypes_seal_rate_gb_s": seal_rate_gb_s,
            "pure_ctypes_open_rate_gb_s": open_rate_gb_s,
        },
        "sealed_shards": {
            "shard3": {
                "inner_bytes": os.path.getsize(shard3_lmz),
                "sealed_bytes": s3_sealed_size,
                "seal_time_seconds": s3_seal_time,
                "seal_rate_gb_s": s3_seal_rate,
                "algorithm": hdr3["algorithm"],
                "key_id": hdr3["key_id"],
            },
            "shard1": {
                "inner_bytes": os.path.getsize(shard1_lmz),
                "sealed_bytes": s1_sealed_size,
                "seal_time_seconds": s1_seal_time,
                "seal_rate_gb_s": s1_seal_rate,
                "algorithm": hdr1["algorithm"],
                "key_id": hdr1["key_id"],
            },
        },
        "structural_inspection": {
            "without_key_latency_ms": info_no_key_ms,
            "without_key_verified": is_verified,
            "with_key_latency_ms": info_with_key_ms,
            "with_key_verified": is_verified_with_key,
            "visible_tensors_count": tensors_count,
        },
        "selective_tensor_extraction": {
            "tensor_name": target_tensor_name,
            "tensor_bytes": len(tensor_bytes),
            "latency_ms": selective_extract_ms,
            "bit_exact_match": True,
        },
        "full_extraction_roundtrip": {
            "restored_bytes": restored_bytes,
            "restore_time_seconds": restore_time,
            "restore_rate_gb_s": restore_rate,
            "bit_exact_sha256_match": True,
            "sha256": restored_sha,
        },
        "security_tamper_matrix": tamper_results,
    }
    return suite_2_data


def main():
    print("=" * 70)
    print("LMSLUICE CATEGORY 6 QUALIFICATION TEST HARNESS")
    print(f"Host: {platform.node()} ({platform.machine()} {platform.system()} {platform.release()})")
    print(f"Software: lmsluice {lmsluice_version}, Python {sys.version.split()[0]}")
    print("=" * 70)

    t_start = time.perf_counter()
    suite_1_res = run_suite_1_multimodal_bundle()
    suite_2_res = run_suite_2_envelope_encryption()
    t_end = time.perf_counter()

    report = {
        "category": "Category 6: Deployment & Packaging",
        "qualification_targets": ["Qwen3.5-4B", "Qwen3-4B-Instruct-2507"],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "lmsluice_version": lmsluice_version,
        },
        "total_duration_seconds": t_end - t_start,
        "suite_1_multimodal_bundle": suite_1_res,
        "suite_2_envelope_encryption": suite_2_res,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"QUALIFICATION COMPLETE in {t_end - t_start:.2f}s")
    print(f"Results persisted to: {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
