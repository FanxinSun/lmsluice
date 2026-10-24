# Category 6: Deployment & Packaging — Qualification Report

> **Qualification Targets:** `Qwen3.5-4B` (Multimodal Vision-Language) and `Qwen3-4B-Instruct-2507`  
> **Host Environment:** Apple Silicon M4 (10 cores, 24 GB Unified Memory, macOS 15.4 / Darwin 24.4)  
> **Software Stack:** PyTorch `2.12.0`, `lmsluice 0.2.0`, `lmz 1.3.0`, OpenSSL `3.5.9` (`libcrypto.dylib`)  
> **Test Date:** October 2026  
> **Status:** **CERTIFIED & QUALIFIED**

---

## Executive Summary

Category 6 qualifies `lmsluice` for production delivery, deployment packaging, and at-rest envelope encryption:
1. **Multimodal Bundle Packaging ([`lmsluice/bundle.py`](../../lmsluice/bundle.py)):**
   * Packaged full multimodal `Qwen3.5-4B` (9.34 GB) into a schema-1 content-addressed bundle manifest covering all multimodal roles: `config`, `preprocess` (vision/video), `vocabulary` (tokenizers, merges, vocab), `opaque` (chat templates, licenses), and `weights`.
   * **Sub-Millisecond Resolution:** Manifest parsing and descriptor resolution completed in **0.94 ms**; structural inventory completed in **1.03 ms**.
   * **Preflight Staging & Atomic Publication:** Verified atomic delivery via invocation-owned staging trees and POSIX/Darwin atomic rename (`renameatx_np` with `RENAME_EXCL`). Verified fail-closed non-destructive collision protection when destination targets pre-exist.
2. **At-Rest Envelope Encryption ([`lmsluice/sealed.py`](../../lmsluice/sealed.py) / [`lmsluice/crypt.py`](../../lmsluice/crypt.py)):**
   * Reached system OpenSSL 3.5.9 `libcrypto` directly via zero-dependency `ctypes` (measured at **6.11 GB/s** multi-threaded decrypt rate on M4 Apple Silicon).
   * Wrapped real compressed model shards (`model-00003` 65.6 MB and multi-gigabyte `model-00001` 2.63 GB) into AES-256-GCM authenticated envelopes at **0.96–1.32 GB/s**.
   * **Zero-Knowledge Structure Inspection:** `lmsluice info` inspected all 174 tensors, shapes, datatypes, and byte extents in **13.36 ms** without the encryption key, keeping weights confidential while exposing file structure for deployment scheduling.
   * **Selective & Full Decryption Fidelity:** Extracted a 49.81 MB layer tensor from the 2.6 GB container in **86.74 ms**; full roundtrip extraction restored uncompressed safetensors shards with **100% SHA-256 bit-exact identity**.
   * **Fail-Closed Tampering Resistance:** Proved fail-closed enforcement across missing keys, wrong keys, altered clear headers (HMAC-SHA256 rejection), and altered ciphertext payloads (AES-256-GCM authentication failure).

---

## 1. Multimodal Bundle Packaging (`Qwen3.5-4B`)

Evaluated against the complete `Qwen3.5-4B` multimodal repository (12 files, 9.34 GB uncompressed):

| Role | File Count | Aggregate Size | Key Multimodal Artifacts |
|---|---:|---:|---|
| **`config`** | 3 | 96.1 KB | `config.json`, `tokenizer_config.json`, `model.safetensors.index.json` |
| **`preprocess`** | 2 | 0.8 KB | `preprocessor_config.json`, `video_preprocessor_config.json` |
| **`vocabulary`** | 3 | 22.88 MB | `tokenizer.json`, `vocab.json`, `merges.txt` |
| **`opaque`** | 2 | 19.3 KB | `chat_template.jinja`, `LICENSE` |
| **`weights`** | 2 | 9,319.83 MB | `model.safetensors-00001-of-00002.safetensors` (5.0 GB)<br>`model.safetensors-00002-of-00002.safetensors` (3.7 GB) |
| **Total Bundle** | **12** | **9,342.83 MB** | **Canonical Manifest SHA-256: `f160309dc3cf0f8da...`** |

### Lifecycle Operation Latencies

* **Manifest Construction (`make_manifest`):** **5.55 s** (hashed and authenticated all 9.34 GB at **1.68 GB/s**).
* **Descriptor Resolution (`resolve_bundle`):** **0.94 ms** (instant parsing of entry points, dependency graphs, and roles).
* **Non-Strict Inventory (`inventory_bundle`):** **1.03 ms** (zero-body metadata verification).
* **Cryptographic Verification (`validate_bundle`):** **5.67 s** (full integrity re-verification across all 12 files).
* **Control-Plane Staging & Publication:** **60.92 ms** (materialized 10 non-weight control artifacts into an atomic staging directory).
* **Full Multimodal Model Materialization:** **20.36 s** (staged, copied, fsynced, verified digests, and atomically published 9.34 GB at **0.46 GB/s**).

### Atomic Publication & Security Invariants

1. **Invocation-Owned Isolation:** Staging occurs in a temporary sibling directory (`stage.XXXXXXXX`) created under the destination parent anchor with `O_EXCL` flags.
2. **Darwin Atomic Rename (`renameatx_np`):** Uses kernel-level `RENAME_EXCL` on Darwin / macOS (and `renameat2(RENAME_NOREPLACE)` on Linux) to ensure instantaneous atomic publication without file clobbering.
3. **Collision Resistance:** Re-materializing into an already published destination fails closed immediately (`BundleError: destination_exists`), guaranteeing that established model deployments cannot be overwritten or corrupted by concurrent runs.

---

## 2. Envelope Encryption (`lmsluice seal` / `lmsluice/sealed.py`)

### Cryptographic Engine Throughput (Apple Silicon M4)

* **Active Backend:** OpenSSL `3.5.9` via `/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib`
* **Dependency Footprint:** Pure standard library `ctypes` (zero third-party dependencies required).
* **Measured Throughput:**
  * **`lmsluice probe` Multi-Threaded Decrypt Rate:** **6.11 GB/s**
  * **Pure Ctypes Single-Stream Seal (AES-256-GCM):** **2.61 GB/s**
  * **Pure Ctypes Single-Stream Open (AES-256-GCM):** **4.79 GB/s**

### Real Model Shard Sealing Benchmark

Evaluated by wrapping finished `lmz` containers in encrypted envelopes:

| Shard Target | Plain Bytes | Coded Bytes | Sealed Envelope Size | Sealing Latency | Sealing Throughput | Overhead |
|---|---:|---:|---:|---:|---:|---:|
| **`model-00003`** | 99.6 MB | 65.6 MB | 65.65 MB | **0.068 s** | **0.96 GB/s** | +781 bytes (tags + HMAC index) |
| **`model-00001`** | 3.96 GB | 2.63 GB | 2.63 GB | **1.99 s** | **1.32 GB/s** | +120.8 KB (tags + HMAC index) |

* **Envelope Structure:**
  * Magic: `LMSLSEAL` (8 bytes)
  * Codec Payload: Extent-by-extent AES-256-GCM sealed units (16-byte tag per unit, nonce = unit ordinal)
  * Clear Structure: Container header, footer, and extent table preserved in clear
  * Integrity: Entire clear structure and extent table protected by HMAC-SHA256 under a dedicated HKDF subkey
  * Footer: `LMSLSTAI` (24 bytes) containing index offset and length

---

## 3. Structural Inspection, Selective Reading, and Recovery

### A. Zero-Knowledge Structural Inspection (`lmsluice info`)

Inspecting `model-00001-of-00003.sealed` (2.63 GB container) without providing the encryption key:
* **Inspection Latency:** **13.36 ms**
* **Visibility:** All **174 tensors** (names, dtypes, shapes, byte sizes, and offsets) read accurately in the clear.
* **Authentication State:** Correctly flagged as `structure NOT verified (no key supplied)`.
* **Deployment Advantage:** Model routers, registries, and Kubernetes orchestrators can plan GPU memory allocations and inspect model architecture without accessing model decryption keys.

### B. Authenticated Structural Inspection (With Key)

Inspecting the sealed container with the 32-byte production key:
* **Inspection Latency:** **14.18 ms**
* **Authentication State:** Upgrades to `structure verified` after validating HMAC-SHA256 across all clear extents and the extent table.

### C. Selective Single-Tensor Extraction

Extracted `model.layers.0.mlp.down_proj.weight` (49.81 MB BF16 tensor, shape `[2560, 9728]`) directly from `model-00001-of-00003.sealed` (2.63 GB):
* **Extraction Latency:** **86.74 ms**
* **I/O Efficiency:** Fetched and decrypted only the specific extent chunks covering the tensor span, avoiding 98.1% of container bytes.
* **Numerical Fidelity:** **100% BIT-EXACT SHA-256 MATCH** against ground-truth uncompressed safetensors.

### D. Full Plain Model Recovery Roundtrip (`lmsluice get`)

Restored `model-00003-of-00003.sealed` to a plain `.safetensors` file:
* **Restoration Throughput:** **99.6 MB in 0.17 s (0.57 GB/s)**
* **Ground-Truth SHA-256:** `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`
* **Restored SHA-256:** `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`
* **Verification:** **100% BIT-EXACT MATCH**.

---

## 4. Fail-Closed Security & Tamper Matrix

Every failure mode was tested live to confirm strict fail-closed enforcement:

| Test Case | Attack / Fault Condition | System Response | Decrypted Bytes | Result |
|---|---|---|:---:|:---:|
| **Case 1: Missing Key** | `LMSLUICE_KEY_FILE` unset and no key passed | Raises `crypt.CryptoUnavailable` with clear instructions on generating keys. | **0 bytes** | **PASS (Fail-Closed) ✓** |
| **Case 2: Wrong Key** | Key supplied with mismatched Key ID | Rejected immediately at header validation: `"written with a different key"`. | **0 bytes** | **PASS (Fail-Closed) ✓** |
| **Case 3: Altered Clear Header** | 1 bit flipped in inner container header | HMAC-SHA256 rejects: `"archive's structure failed authentication... Nothing was decrypted."` | **0 bytes** | **PASS (Fail-Closed) ✓** |
| **Case 4: Altered Ciphertext** | 1 bit flipped in extent payload unit | AES-256-GCM tag verification fails: `"unit 2 failed authentication -- authentication failed"`. | **0 bytes** | **PASS (Fail-Closed) ✓** |

---

## 5. Category 6 Qualification Conclusion

1. **Production-Ready Bundling:** `lmsluice/bundle.py` manages complex multimodal models (`Qwen3.5-4B`) with sub-millisecond resolution and strict atomic publication guarantees.
2. **High-Speed Zero-Dependency Encryption:** Standard library `ctypes` reaching OpenSSL 3 delivers 6.11 GB/s decryption on Apple Silicon M4 without third-party python packages.
3. **Sound Threat Model:** Payload sealed, structure clear, clear bytes HMAC-guarded. Zero leakage of model weights combined with instant metadata observability.
4. **Resilient Fail-Closed Enforcement:** All cryptographic integrity checks prevent unauthorized execution or tampered byte injection.
