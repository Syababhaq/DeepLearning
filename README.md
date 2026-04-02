# 🚩 FlagOS: High-Performance Triton Operator Development
**[FlagGems Operator Development Competition — Track 1]**

This repository contains the development, optimization, and assessment of 20 high-performance LLM operators implemented in **Triton**. The goal is to reach parity with at least 4 PyTorch (2 easy, 1 medium, 1 difficult) with specialized kernels that achieve ≥0.9x speedup and 100% functional correctness for the FlagGems library.

---

### 🚀 Tech Stack
*   **Language:** Python 3.10+
*   **Frameworks:** [Triton](https://github.com/triton-lang/triton), [PyTorch](https://pytorch.org/)
*   **Base Library:** [FlagGems](https://github.com/FlagOpen/FlagGems)
*   **Infrastructure:** NVIDIA T4/A100/H100 (Google Colab & FlagTree Compiler)

---

### 📊 Competition Progress Tracker
We are targeting **4 Operators**. Each operator is validated for functional accuracy (vs. PyTorch) and performance speedup.
Current progress:

| # | Operator | Category | Difficulty | Status | Speedup (vs Torch) |
|---|---|---|---|---|---|
| 02 | `logaddexp` | Binary/Pointwise | Low | ✅ Done | 1.05x |
| 03 | `cosh` | Unary/Pointwise | Low | ✅ Done | 1.02x |
| 11 | `median` | Reduction/Sort | **Medium** | Currently working | ~0.3x |


---

### 🛠️ Key Technical Implementations
*   **`median.dim` (Medium):** Implemented using a high-efficiency **Radix Sort + Vectorized Gather** pipeline. This matches PyTorch's complexity and handles original index tracking.
*   **Numerical Stability:** Pointwise operators (like `cosh` and `logaddexp`) use explicit `float32` promotion to prevent overflows in `float16`/`bfloat16` precision.
*   **Hardware Utilization:** Our kernels are optimized for **HBM Bandwidth Utilization**, targetting >80% peak performance on memory-bound workloads.

---

### ⚖️ Evaluation Criteria
*   **Functional Correctness (30%):** Zero-error index matching and float32 tolerance validation.
*   **Performance (20%):** Achieving ≥0.9x speedup vs. PyTorch native implementation.
*   **Open-Source Adaptability (10%):** Apache 2.0 Licensing, PEP 8 styling, and FlagGems PR compatibility.

---
**License:** Apache 2.0

---
