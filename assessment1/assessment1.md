# Assessment 1: Problem Understanding & Baseline
## FlagGems Operator Development Competition — Track 1

### 1. Written Description of the Competition Task
The **FlagGems Operator Development Competition (Track 1)** focuses on the creation and optimization of high-performance GPU operators using the **Triton** programming language. The primary objective is to contribute to the **FlagGems** library—a collection of high-performance Triton-based operators designed to be seamless replacements for PyTorch's native operations.

**Key Objectives:**
- **Performance Parity & Excellence:** Develop kernels that achieve at least **0.9x speedup** relative to PyTorch's highly optimized native implementations.
- **Functional Correctness:** Ensure **100% accuracy** across various tensor shapes, strides, and data types (Float32, Float16, BFloat16).
- **Library Integration:** Implement at least **4 operators** (2 Easy, 1 Medium, and 1 Difficult) that comply with the FlagGems API and architectural standards.
- **Hardware Efficiency:** Maximize GPU memory bandwidth and compute utilization through vectorized memory access and efficient tiling strategies.

---

### 2. Dataset Overview and Evaluation Metric Explanation

#### **Dataset Overview**
Unlike traditional image or text datasets, the "dataset" for this competition consists of **synthetic multidimensional tensors** generated dynamically to stress-test the kernels.
- **Input Generation:** Tensors are typically generated using `torch.randn` or `torch.randint` to cover a wide range of values.
- **Data Types:** Support for `float32`, `float16`, and `bfloat16` is mandatory to ensure precision requirements for modern LLM training.
- **Shapes and Strides:** Kernels are tested on a spectrum of shapes—from small 1D vectors to large 4D tensors with non-contiguous memory layouts (strides)—to validate the robustness of the tiling logic.

#### **Evaluation Metrics**
1.  **Functional Correctness (30%):**
    - Measured using `torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2)`.
    - Verification across multiple seeds and input ranges to detect edge-case failures (e.g., NaN, Inf).
2.  **Performance / Speedup (20%):**
    - Measured in **Latency (ms)** using `triton.testing.do_bench`.
    - **Speedup Ratio = Latency(PyTorch) / Latency(Triton)`.
    - Goal: Speedup ≥ 0.9x.
3.  **Open-Source Adaptability & Quality (10%):**
    - Code readability, documentation, and compliance with the Apache 2.0 license.

---

### 3. Exploratory Data Analysis (EDA) / Baseline Notebook
In modern operator development, "EDA" involves profiling the performance characteristics of the baseline PyTorch operation across different workloads.

**Baseline Notebooks:** The current baseline models and profiling environments are maintained in the following `.ipynb` files within the repository:
- `flagos_median2.ipynb`: Baseline and development for the `median` operator.
- `flagos-cosh.ipynb`: Baseline and development for the `cosh` operator.
- `flagos-logaddexp.ipynb`: Baseline and development for the `logaddexp` operator.

These notebooks demonstrate the end-to-end workflow:
  - Defining the Triton JIT kernels.
  - Implementing the wrapper functions.
  - Benchmarking against native PyTorch across varying tensor sizes.
  - Visualizing the latency comparisons using `triton.testing.perf_report`.

---

### 4. Working Baseline Deep Learning Model (Functional Operator Kernels)
Due to the nature of Track 1 (Operator Development and Optimization), our "model" is not a traditional neural network architecture, but rather the specialized, high-performance GPU kernels themselves. These kernels act as the fundamental computational primitives that deep learning models rely on.

Our baseline operator models are fully functional, JIT-compiled Triton kernels built for the FlagGems framework. Rather than a raw code dump, their functionality is characterized by:
- **Direct Executability:** They are actively registered and executable as drop-in PyTorch replacements.
- **Dynamic Infrastructure:** They leverage specialized dynamic pointwise decorators to abstract complex boilerplate operations like automated N-dimensional tiling, generic broadcasting, and non-contiguous memory handling.
- **Numerical Robustness:** The underlying math (e.g., in operations like `cosh` or `logaddexp`) forces `float32` upcasting to prevent the common overflow and underflow problems encountered during mixed-precision LLM training.

*Note: The complete functionality and execution environment for these baseline models correspond to the `.ipynb` notebooks linked in Section 3.*

---

### 5. Documented Baseline Performance Score
The following table summarizes the documented performance of our initial baseline kernels compared to PyTorch:

| Operator | Status | Functional Match | Speedup (vs. PyTorch) |
| :--- | :--- | :--- | :--- |
| `logaddexp` | ✅ Completed | 100% | **1.05x** |
| `cosh` | ✅ Completed | 100% | **1.02x** |
| `median` | 🚧 In Progress | 40% | **~0.3x** |

