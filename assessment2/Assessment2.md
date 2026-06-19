# What have we done for assessment 2:

1. **Experiment 1 (Baseline)**: Used Bitonic Sort and Radix Sort but achieved poor performance (~0.09x–0.50x speedup vs PyTorch) due to the O(N log N) or O(N log²N) complexity when PyTorch uses O(N) selection.

2. **Experiment 2 (QuickSelect)**: Attempted to implement QuickSelect but found it too divergent and incompatible with GPU SIMT execution models in Triton.

3. **Experiment 3 (Radix Select)**: Switched to a Radix Select algorithm that counts bits from MSB to LSB in exactly 32 constant-time passes. This dramatically improved performance to **~1.1x–1.5x speedup** for medium sizes (N ≤ 4096) by leveraging GPU counting instructions and eliminating intermediate memory access.

4. **Experiment 4 (Edge Cases & Duplicates)**: Addressed correctness issues with duplicates, floating-point special values (`-0.0` vs `0.0`), and NaN handling by canonicalizing these values before the bitwise selection algorithm.

5. **Submit Pull Request for The Competition**: We have submitted our coding to be reviewed by the admin.


The key insight: trading complex branching logic for simpler, constant-time arithmetic operations on GPUs yields better performance and correctness.
