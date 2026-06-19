# Assessment 2: Median Operator (V2/V3)

## Iterative Development Log

### Experiment 1: Baseline (Full Sort)

* **Algorithm:** Bitonic Sort ($N \le 1024$), Radix Sort + Gather ($N > 1024$)
* **Hyperparameters:** `BLOCK_N = next_power_of_2(N)`, 8-bit passes
* **Scores vs PyTorch:** Shape (1024, 64) is ~0.09x speedup, Shape (1024, 512) is ~0.50x speedup
* **Insights:** Sorting is $O(N \log^2 N)$ or $O(N \log N)$. PyTorch uses $O(N)$ selection.

### Experiment 2: QuickSelect Prototype

* **Algorithm:** QuickSelect in Triton
* **Insights:** Highly divergent, uncoalesced memory access, incompatible with GPU SIMT execution.

### Experiment 3: Radix Select in Registers

* **Algorithm:** Radix Select (MSB to LSB counting)
* **Hyperparameters:** 1 bit/pass, 32 passes (Float32), `BLOCK_N = next_power_of_2(N)`, max $N \le 4096$
* **Scores vs PyTorch:** ~1.1x to 1.5x speedup
* **Insights:** 32 constant-time passes drastically reduce instruction count. Leverages GPU `popc` instructions and pure arithmetic without memory round-trips.

### Experiment 4: Edge Cases & Duplicates (V3)

* **Algorithm:** Radix Select (Bit-Level Canonicalization)
* **Problem:** Returned original indices for duplicate median values mismatched PyTorch.
* **Insights:** PyTorch selection is unstable. Bitwise math aggressively sorts `-0.0` (`0x7FFFFFFF`) before `0.0` (`0x80000000`).
* **Solution:** Forced `-0.0` to `0.0`. Updated tests for unstable duplicate indices. Submitted PR.

**Key Insight:** Trading complex branching logic for simpler, constant-time GPU arithmetic yields better performance and correctness.