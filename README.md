# 🚩 FlagOS: High-Performance Triton Operator Development
**[FlagGems Operator Development Competition — Track 1]**
competition link: https://www.kaggle.com/competitions/track1-operator-development-and-optimization-flagos-challenge/overview

This repository contains the development, optimization, and assessment of 20 high-performance LLM operators implemented in **Triton**. The goal is to reach parity with at least 4 PyTorch (2 easy, 1 medium, 1 difficult) with specialized kernels that achieve ≥0.9x speedup and 100% functional correctness for the FlagGems library.

---

### 🚀 Tech Stack
*   **Language:** Python 3.10+
*   **Frameworks:** [Triton](https://github.com/triton-lang/triton), [PyTorch](https://pytorch.org/)
*   **Base Library:** [FlagGems](https://github.com/FlagOpen/FlagGems)
*   **Infrastructure:** NVIDIA T4/A100/H100 (Google Colab & FlagTree Compiler)

---

### 📊 Competition Progress Tracker
We are targeting **6 Operators**. Each operator is validated for functional accuracy (vs. PyTorch) and performance speedup.
Current progress:

| # | Operator | Category | Difficulty | Status | Speedup (vs Torch) |
|---|---|---|---|---|---|
| 02 | `logaddexp` | Pointwise | Low | ✅ Done | 1.05x |
| 03 | `cosh` | Pointwise | Low | ✅ Done | 1.02x |
| 11 | `median` | Reduction | Medium | ✅ Done | 1.32x |
| 18 | `svd` | Linalg | Difficult | ✅ Done | 30.61x |
| 19 | `ctc_loss` | Loss | Difficult | ✅ Done | 1.292x |
| 20 | `grid_sample` | Special | Difficult | ✅ Done | 4.63x |

---

### 📂 Project Structure

```text
DeepLearning/
│
├── README.md
│
├── Assessment_1/
│   ├── assessment1.md
│   ├── flagos-cosh.ipynb
│   ├── flagos-logaddexp.ipynb
│   └── flagos-median2.ipynb
│
├── Assessment_2/
│   ├── assessment2.md
│   ├── REPORT_Assesment2_Fighter3.6.pdf
│   └── flagos_median3.ipynb
│
└── Assessment_3/
    ├── FinalReport_Fighter3.6.pdf
    ├── Grid Sample Operator.pdf
    ├── ctc_loss(src code).py
    ├── ctc_loss.md
    ├── ctc_loss_benchmark(test).ipynb
    ├── devlog_svd.md
    ├── flagOS_svd.ipynb
    └── grid_sample.ipynb

```

---



### ⚖️ Evaluation Criteria
*   **Functional Correctness (30%):** Zero-error index matching and float32 tolerance validation.
*   **Performance (20%):** Achieving ≥0.9x speedup vs. PyTorch native implementation.
*   **Open-Source Adaptability (10%):** Apache 2.0 Licensing, PEP 8 styling, and FlagGems PR compatibility.

---
**License:** Apache 2.0

---
