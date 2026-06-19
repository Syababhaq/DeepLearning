# 🚀 Dev Log: FlagOS Challenge (Track 1) - LLM Operator Development

**Project:** Custom High-Performance SVD Operator in Triton for FlagGems

**Team:** Ikhwan, Syabab, Zamir

**Hardware Target:** NVIDIA T4 (Google Colab)

---

## 🎯 Objective
Develop and optimize the Singular Value Decomposition (SVD) operator (`torch.linalg.svd`) entirely in Triton. The goal is to achieve strict mathematical accuracy (`torch.allclose`) while targeting a device-side speedup ratio of $\ge 0.9\times$ compared to PyTorch's native C++ `cuSOLVER` backend.

## 🏗️ Architectural Evolution & Roadblocks

### Phase 1: Sequential One-Sided Jacobi (The Baseline)
* **Approach:** Implemented the standard Hestenes/One-Sided Jacobi method entirely within SRAM.
* **Result:** `Accuracy: PASSED` | `Speedup: 0.00x`
* **Bottleneck:** Nested sequential `for` loops. Triton executes loops over column pairs sequentially, leaving thousands of GPU cores idle. The execution time was entirely dominated by thread idling and $O(N^2)$ algorithmic complexity.

### Phase 2: The Masking & Broadcasting Trap
* **Approach:** Attempted to parallelize column selection using a Round-Robin tournament paired with `tl.where` masking to extract multiple columns simultaneously.
* **Result:** `Speedup: 0.02x`
* **Bottleneck:** Triton's compiler is optimized for dense block operations, not dynamic advanced slicing. The masking approach forced the GPU to perform thousands of slow memory lookups, completely saturating the memory bandwidth.

### Phase 3: The Breakthrough - Tensor Core Block-Jacobi
* **Approach:** Pivoted away from fine-grained column swapping. We restructured the kernel to use a **Block-Jacobi method**.
    * Chunked the global matrix into dense $16 \times 16$ sub-blocks.
    * Accumulated local rotations into an $R_{acc}$ matrix.
    * Replaced all masked updates with a single, massive `tl.dot(A, R)` matrix multiplication per block.
* **Result:** Forced the Triton compiler to map operations directly to hardware `mma.sync` (Matrix Multiply-Accumulate) Tensor Core instructions. 

---

## 📊 Current Benchmarks

*Benchmarked on NVIDIA T4 via `triton.testing.do_bench` (100 reps, 25 warmup)*

| Shape | Dimension | Triton (μs) | PyTorch (μs) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| (16, 16) | 256 | 23.43 |717.18 | **30.61x** |
| (32, 32) | 1024 | 126.18 | 485.45 | **3.85x** |
| (64, 64) | 4096 | 9555.48 | 1482.86 | 0.16x |

### Benchmark Analysis & The LLM Context
We successfully engineered a custom kernel that decisively outperforms PyTorch's native backend for sub-block SVD operations. 

**The 64x64 Limitation:** The performance drop at $64 \times 64$ is a known hardware constraint. Our current kernel launches on a single GPU grid (`grid = (1,)`). For a matrix of this size, the dependency chain inside the `while` loop chokes a single Streaming Multiprocessor (SM). PyTorch circumvents this by orchestrating work across multiple grids, an architecture that is complex to replicate in Triton due to a lack of native cross-block thread synchronization.

**Strategic Value:** In modern LLM architectures (e.g., Llama 3), matrix operations are heavily parallelized into "Attention Heads," which frequently utilize inner dimensions of 64 or 128. Our kernel acts as an ultra-fast **Inner Engine**. By pairing our Triton kernel with a high-level block-tiling orchestrator, researchers can execute low-rank decompositions on attention head tiles at unprecedented speeds.


