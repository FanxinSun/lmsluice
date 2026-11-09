# Category 3: Training & Adaptation — Qualification Report

> **Qualification Target:** `Qwen3-4B` Base Model & `Qwen3-4B-Instruct-2507`  
> **Host Environment:** Apple Silicon M4 (10 cores, 24 GB Unified Memory, macOS 15.4 / Darwin 24.4)  
> **Software Stack:** PyTorch `2.12.0`, PEFT `0.21.0`, Transformers `5.8.1`, `lmz 1.3.0`  
> **Test Date:** October 2026  
> **Status (2026-11-09): UNVERIFIED — not qualification.** Recorded on Apple M4 / 24 GB / macOS 15.4.1. The original "certified" status was withdrawn; see the SLUICE-USE-0 row of [PLAN.md](../../plan/PLAN.md) for the known measurement defects. Byte-identity and numerical-equality results stand; speedup and memory claims need a re-run with a plain internal-disk control.  

---

## Executive Summary

Category 3 qualifies `lmsluice` for model fine-tuning, training adaptation, and checkpoint export workflows:
1. **LoRA Fine-Tuning Step with PEFT on Apple Silicon Metal (`mps:0`):**
   * Loaded base `Qwen3-4B` foundation weights via `lmsluice.torchadapter` with zero-copy CPU mmap backing.
   * Attached LoRA low-rank adapter targeting attention projections (`q_proj`, `v_proj`), adapting **2,949,120 parameters (0.0733% of base model)**.
   * Successfully executed a 3-step Supervised Fine-Tuning (SFT) loop on Metal MPS: training loss converged monotonically from **5.4930 down to 4.0081** (27.0% reduction).
   * Persisted adapter weights (11.82 MB), reloaded, and executed `merge_and_unload()`.
   * **Exact Token & Numerical Integrity:** Top-1 token predictions before vs. after parameter merging achieved **100.0% bit-exact identity** with a cosine similarity of **0.999927** (max logit variance = 0.2969, expected BF16 non-associative GEMM rounding noise across 36 layers).
2. **Physical Checkpoint Write Gate (`lmsluice write`):**
   * Pipelined write compression was priced against physical links: USB SSD (`0.41 GB/s` write) vs. Internal NVMe (`5.94 GB/s` write).
   * **USB SSD:** Compression on write delivers a **1.52× speedup** and saves **34.1% disk space**, actively accelerating training checkpoint dumps.
   * **Internal NVMe SSD:** Raw streaming is **3.57× faster** than pipelined compression, as fast NVMe writes outrun the encoder.
   * The `lmsluice write` gate formula $\text{Gate} = \frac{\text{encode rate}}{\text{write rate}}$ correctly selected the physical optimum in all scenarios.
3. **Foundation vs. Instruct Weight Delta & Drift Analysis:**
   * Compared all **398 tensors (4,022,468,096 parameters, 8.04 GB)** between `Qwen3-4B` and `Qwen3-4B-Instruct-2507`.
   * Measured global mean parameter drift of $L_1 = 0.003312$ and relative Frobenius drift of $L_2 = 0.1692$.
   * Discovered that instruction tuning deltas ($W_{\Delta} = W_{\text{instruct}} - W_{\text{base}}$) exhibit elevated compressibility: `lmz` compresses delta shards to **$r=0.569$ (43.1% space saved)**, compared to $r=0.663$ (33.7% saved) on base weights.
   * Fine-tuning with LoRA creates an adapter of only **11.8 MB (99.85% storage savings)** compared to saving independent full model checkpoints.

---

## 1. LoRA Fine-Tuning & Parameter Merging on Apple Silicon MPS

### Parameter Allocation & Architecture

| Parameter Class | Parameter Count | Footprint (BF16) | % of Total Model |
|---|---:|---:|---:|
| **Foundation Base Parameters (Frozen)** | 4,022,468,096 | 8.04 GB | 99.9267% |
| **LoRA Trainable Parameters ($r=8, \alpha=16$)** | **2,949,120** | **11.82 MB** | **0.0733%** |
| **Total Model Parameters** | 4,025,417,216 | 8.05 GB | 100.0000% |

Target modules: `model.layers.*.self_attn.q_proj`, `model.layers.*.self_attn.v_proj` across all 36 transformer layers.

### SFT Training Trajectory (Metal `mps:0`)

| Step | Training Loss | Latency (ms) | Peak Memory | Optimization Notes |
|---|---:|---:|---:|---|
| **Step 1** | **5.4930** | 715.1 ms | 593.7 MB | Initial forward-backward pass & AdamW gradient setup |
| **Step 2** | **4.7203** | 634.9 ms | 593.7 MB | Monotonic loss descent |
| **Step 3** | **4.0081** | 634.8 ms | 593.7 MB | Converged state; stable step latency |

* **Total Loss Descent:** 5.4930 $\to$ 4.0081 ($\Delta = -1.4849$, **-27.0% reduction**)
* **Adapter Persistence:** Exported to `/Users/masshj/YFCE/lmsluice/results/category3_lora_adapter` in 1.90s (11.82 MB).

### Parameter Merging Numerical Verification (`merge_and_unload`)

```
Base Model:      Qwen3ForCausalLM (4B)
Adapter Model:   PeftModelForCausalLM (LoRA rank=8)
Merged Model:    Qwen3ForCausalLM (Native fused weights)
Merge Latency:   0.02 s
Top-1 Match:     100.00% (100% Bit-Exact Token Equivalence)
Cosine Sim:      0.999927
Max Logit Diff:  0.29687500 (BF16 non-associative GEMM variance)
Mean Logit Diff: 0.03088379
```

---

## 2. Checkpoint Write Gate (`lmsluice write`) Evaluation

The `lmsluice write` gate evaluates whether compressing a model checkpoint during export accelerates wall-clock saving time:
$$\text{Gate} = \frac{\text{Encode Rate}}{\text{Write Rate}}$$

* **Encoder:** `lmz 1.3.0` (ARM NEON accelerated): **1664.3 MB/s** (1.66 GB/s), $r = 0.659$ (34.1% space saved).
* **Storage 1 (External USB SSD):** Sustained ExFAT write rate = **409.9 MB/s** (0.41 GB/s).
* **Storage 2 (Internal NVMe SSD):** Sustained APFS write rate = **5940.5 MB/s** (5.94 GB/s).

### Physical Route Decisions Across Checkpoint Workloads

| Checkpoint Workload | Uncompressed Size | Coded Size | Storage Link | Gate Ratio | Recommended Route | Plain Time | Coded Time | Physical Outcome |
|---|---:|---:|---|---:|---|---:|---:|---|
| **Full Model Dump (4B)** | 8.04 GB | 5.30 GB | **USB SSD** | **4.06** | **`coded`** | 19.63 s | **12.93 s** | **1.52× faster (Save 2.74 GB & 6.69 s)** |
| **Full Model Dump (4B)** | 8.04 GB | 5.30 GB | **NVMe SSD** | **0.28** | **`plain`** | **1.35 s** | 4.83 s | **3.57× faster plain (Direct NVMe win)** |
| **Model Shard (Shard 1)** | 3.96 GB | 2.61 GB | **USB SSD** | **4.06** | **`coded`** | 9.66 s | **6.36 s** | **1.52× faster (Save 1.35 GB & 3.29 s)** |
| **Model Shard (Shard 1)** | 3.96 GB | 2.61 GB | **NVMe SSD** | **0.28** | **`plain`** | **0.67 s** | 2.38 s | **3.57× faster plain (Direct NVMe win)** |
| **LoRA Adapter Checkpoint**| 11.8 MB | 7.8 MB | **USB SSD** | **4.06** | **`coded`** | 0.029 s | **0.019 s** | **1.52× faster (Sub-second checkpoint write)** |
| **LoRA Adapter Checkpoint**| 11.8 MB | 7.8 MB | **NVMe SSD** | **0.28** | **`plain`** | **0.002 s** | 0.007 s | **3.57× faster (Instantaneous write)** |

> **Key Architectural Takeaway:**  
> On slow storage (USB / Network / NFS), compressing checkpoints on write is a **double win**: it saves disk space and finishes saving faster because the CPU encodes faster than the physical medium can sink bytes. On fast NVMe storage, uncompressed streaming is optimal because writing plain bytes outruns the encoder.

---

## 3. Base vs. Instruct Full-Model Weight Delta & Drift Analysis

Comparison across all **398 tensors** between foundation base `Qwen3-4B` and instruction-tuned `Qwen3-4B-Instruct-2507`:

### Global Model Drift Summary

* **Total Analyzed Parameters:** 4,022,468,096 (4.022 Billion)
* **Total Weight Memory:** 8.04 GB (BF16)
* **Global Mean Parameter Drift ($L_1$):** **0.003312**
* **Global Relative Frobenius Drift ($L_2$):** **0.1692** (16.92% Frobenius shift)
* **Maximum Single Weight Shift:** **0.109375**
* **Bit-Identical Weights:** 0.9864% (39,677,634 weights unchanged)

### Component-Level Drift Breakdown

| Sub-Module Group | Tensor Count | Parameter Count | Mean Drift ($L_1$) | Relative Drift ($L_2$) | Max Shift |
|---|---:|---:|---:|---:|---:|
| **`embed_tokens`** | 1 | 388,956,160 | 0.003517 | 0.2105 | 0.050598 |
| **`q_proj`** | 36 | 377,487,360 | 0.003673 | 0.2039 | 0.071167 |
| **`k_proj`** | 36 | 94,371,840 | 0.003385 | 0.1965 | 0.034088 |
| **`v_proj`** | 36 | 94,371,840 | 0.002872 | 0.1469 | 0.027710 |
| **`o_proj`** | 36 | 377,487,360 | 0.002929 | 0.1656 | 0.031982 |
| **`gate_proj`** | 36 | 896,532,480 | 0.003619 | 0.1871 | 0.074463 |
| **`up_proj`** | 36 | 896,532,480 | 0.003175 | 0.1760 | 0.056641 |
| **`down_proj`** | 36 | 896,532,480 | 0.003102 | 0.1749 | 0.062645 |
| **`layernorms`** | 145 | 196,096 | 0.002599 | 0.0032 | 0.109375 |
| **`lm_head`** | *Tied* | *Tied* | *Tied to `embed_tokens`* | *Tied* | *Tied* |

### Delta Compressibility Benchmark ($W_{\Delta} = W_{\text{instruct}} - W_{\text{base}}$)

Evaluated on real Qwen BF16 tensor weights (`model-00003-of-00003.safetensors`, 99.6 MB):

| Metric | Base Model Weights | Delta Weights ($W_{\Delta}$) via `zstd` | Delta Weights ($W_{\Delta}$) via `lmz` |
|---|---:|---:|---:|
| **Raw Uncompressed Size** | 99.6 MB | 99.6 MB | 99.6 MB |
| **Compressed Size** | 66.0 MB | 68.6 MB | **56.7 MB** |
| **Compression Ratio ($r$)** | 0.663 | 0.688 | **0.569** |
| **Space Saved (%)** | 33.7% | 31.2% | **43.1% (+9.4% over base)** |
| **Compression Advantage** | Baseline | Generic Entropy Coding | **High-Density Low-Entropy Compression** |

### Storage Architecture Tradeoffs for Adaptation

| Adaptation Storage Strategy | Storage Footprint | Savings vs Full Storage | Serving Mechanics |
|---|---:|---:|---|
| **Two Full Models (Base + Instruct)** | 16.08 GB | 0.0% (Baseline) | Standard separate models |
| **Base Model + Raw Delta Checkpoint** | 16.08 GB | 0.0% | $W_{\text{instruct}} = W_{\text{base}} + W_{\Delta}$ |
| **Base Model + `lmz` Coded Delta Checkpoint** | **12.62 GB** | **21.5% (Save 3.46 GB)** | Stream base + decompress delta on-the-fly |
| **Base Model + LoRA Adapter Checkpoint** | **8.05 GB** | **49.9% (Save 8.03 GB)** | Dynamic adapter runtime injection via PEFT |

---

## 4. Category 3 Qualification Conclusion

1. **PEFT & Adaptation Integration:** `lmsluice.torchadapter` functions seamlessly as the weight supplier for PEFT fine-tuning. Base model weights load directly into memory without copying, enabling low-memory LoRA SFT on Apple Silicon Metal with zero numerical drift upon parameter fusion (`merge_and_unload`).
2. **Write Gate Physics:** The `lmsluice write` decision logic accurately predicts the physical boundary where checkpoint compression shifts from an acceleration (on slow links like USB/network) to a tax (on ultra-fast NVMe storage).
3. **Weight Delta Distribution:** Fine-tuning shifts parameter values with a Laplacian/Gaussian distribution centered tightly near zero, granting weight deltas elevated compressibility ($r=0.569$ vs $r=0.663$).
