# CTC Loss — A Fused Triton Kernel for FlagGems
### FlagOS Track 1 | Kaggle Competition Submission

**Result: 0.697× / 1.221× / 1.292× end-to-end speedup over PyTorch** (small / medium / large),
up from a 0.24× starting point. The peak speed was higher (~2.0x), but the final numbers reflect
necessary correctness overheads (cross-warp barriers and autograd.Function wrapping).

---

## Technical Depth (/30)

### Why CTC Loss is a hard operator

**CTC (Connectionist Temporal Classification) loss** is not an elementwise or reduction op —
it is a full dynamic-programming algorithm (Graves 2006) parallelized in Triton:

- **Alpha recursion** over time: a sequential dependency chain of length T per batch element
- **Log-domain arithmetic** with -∞-safe logaddexp (empty lattice states must not produce NaN)
- **Extended-label lattice**: blanks interleaved between targets, with the repeated-label
  skip rule (`l'[s] ≠ l'[s−2]`)
- **Variable lengths** in both time (input_lengths) and labels (target_lengths), handled
  per batch element with lane freezing
- **ATen API compatibility**: our `log_alpha` output must match PyTorch's exact layout,
  because the backward pass consumes it

### Final architecture: one kernel launch end-to-end

```
F.ctc_loss ──► device-key override of the aten::ctc_loss composite
                 └── ctc_loss_fwd_kernel  (grid = (N,), one program per batch element)
                       ├── extended-label mapping + skip rule  (registers, computed once)
                       ├── alpha recursion over T              (rows staged via L1/L2)
                       ├── per-sample nll readout
                       └── fused mean/sum reduction            (tl.atomic_add into scalar)
```

| Decision | Rationale |
|----------|-----------|
| **One program per batch element** (1D mapping) | The alpha recursion is sequential in T; the parallel dimension is the lattice width (2S+1 lanes per program). 2D tiling across batch×labels was tried and produced worse code: register pressure and compile bloat. |
| **Fused forward + reduction** via `tl.atomic_add` | The single most impactful change. mean/sum accumulate directly into a scalar during the forward kernel — the second kernel launch disappears entirely. |
| **Override the `aten::ctc_loss` composite**, not just `_ctc_loss` | Under `use_gems()`, the composite's own `clamp_min`/`div`/`mean` dispatch into FlagGems' Triton kernels — overhead we can only remove by owning the composite. Autograd stays correct: the CompositeImplicitAutograd decomposition still serves `requires_grad` inputs. |
| **`@libentry()` cached launcher** | FlagGems' standard decorator; skips Python re-binding/checking of kernel args on every call. |
| **Offsets as in-kernel pointer math** | For the common 2D target layout, the target base is just `n * stride` computed in-kernel — zero host-side tensor ops. (The rare 1D concatenated layout uses a per-program length scan; correct, not perf-critical.) |
| **Native backward capture** | Backward delegates to PyTorch's `aten::_ctc_loss_backward` CUDA kernel, snapshotted at import via `torch.library.get_kernel` (before our registration shadows it). Our forward keeps `log_alpha` bit-compatible with what that kernel expects. |

### Numerical stability

```
m      = max(a0, a1, a2)
safe_m = 0 if m == -∞ else m            # guard: exp(-∞ − -∞) would be NaN
out    = -∞ if m == -∞ else m + log(exp(a0−safe_m) + exp(a1−safe_m) + exp(a2−safe_m))
```

Three-way stable logaddexp at every lattice step; impossible alignments (T < 2S−1) yield
inf loss exactly like PyTorch, and `zero_infinity` is honored.

---

## Experimental Rigor (/25)

### The diagnostic insight that drove everything

At every stage, FlagGems' deficit vs PyTorch was a **near-constant absolute gap
(~0.65–0.85 ms) across all three shapes** — while GPU work scales with shape, our gap
didn't. That signature means *fixed per-call CPU overhead*, not kernel throughput. Each
round attacked a layer of that overhead and re-measured.

| Round | Hypothesis | Change | End-to-end speedup (small) |
|-------|-----------|--------|---------------------------|
| **0** (baseline) | — | ATen passthrough via captured native kernel | 0.24× |
| **1** | Passthrough can never win: identical GPU work + wrapper syncs (two `.tolist()` per call) | Custom Triton forward kernel, zero host syncs | 0.45× |
| **2** | Composite's reduction ops + our own wrapper's `arange`/`mul` dispatch into FlagGems kernels | Composite override + separate reduction kernel | ~no change |
| **3a** | Raw Triton launches re-bind args in Python each call | `@libentry` cached launchers, in-kernel offsets | ~no change |
| **3b** | Even *two* cached launches are too many | **Reduction fused into forward via `tl.atomic_add`; `num_warps` tuned** | **0.98–1.21× (Pre-correctness)** |
| **4-6** | Cross-warp race on large shapes and autograd disconnect | Add `torch.autograd.Function` and conditional `tl.debug_barrier()` | **0.697× (Final)** |

The flat results of Rounds 2–3a were themselves informative: they bounded how much each
overhead layer contributed and pointed to launch *count* as the remaining lever.

### What didn't work (ablations)

| Attempt | Outcome |
|---------|---------|
| 2D tiling across batch + lattice | Compiler bloat, register pressure, worse latency — reverted to 1D |
| Over-allocating `log_alpha` by T to avoid a `.item()` sync | BLOCK_S balloons (64 → 1024 for large T) → ~16× more ALU work per timestep — rejected |
| Host-side `torch.cumsum`/`torch.arange` for offsets | Under `use_gems()` these dispatch into FlagGems' *own* Triton kernels — adding the very overhead being removed |

### Measurement discipline

- **End-to-end wall-clock** (the competition metric): Colab notebook, `time.time()` around
  the call with `torch.cuda.synchronize()`, after warmup
- **Kernel-only time** (`torch.cuda.Event`) to separate GPU math from CPU overhead —
  this is how we proved the wrapper, not the kernel, was the bottleneck
- **Correctness**: `pytest tests/test_ctc_loss.py` vs PyTorch reference (5 shape configs ×
  3 reductions, plus dedicated backward, `zero_infinity`, zero-target-length, repeated-label
  skip-rule, 1D-target, and composite-backward tests)
- A stale-code incident (benchmarking an old branch) taught us to pin measurements to a
  commit hash. **Two measurement points matter here:** the *pre-correctness peak* (Round 3b,
  commit `54ca1697`) and the *final correctness-constrained* build (head of the `pr/ctc-loss`
  branch, code frozen at `7798c34f`). The final numbers below are from that final build —
  **not** from `54ca1697`, which predates the `autograd.Function` + barrier changes that
  lowered the small-shape result.

---

## Results & Analysis (/20)

### End-to-end speedup vs PyTorch (competition shapes, final `pr/ctc-loss` build)

These are the **final, correctness-constrained** numbers — race-free and autograd-correct
(measured on the final build, code frozen at commit `7798c34f`):

```
Shape     T     N     C     Final speedup
─────────────────────────────────────────────
small     50    8     28    0.697×
medium    150   16    64    1.221×
large     300   32    128   1.292×
```

> Note: We achieved up to 2.01× speedup before we discovered a cross-warp race condition and an autograd graph disconnect. The final numbers (0.697x / 1.221x / 1.292x) are the fully correct, race-free, and backward-compatible versions.

### Why the curve looks like this

- **Small shapes hover below 1× (0.697×):** FlagGems routes ATen calls through Python by design;
  a structural dispatch floor (op override → Python wrapper → cached launcher) remains. Even after eliminating the double-launch, the `torch.autograd.Function.apply` boundary adds unavoidable Python overhead that consumes a massive portion of the ~0.27ms total budget.
- **Medium/large reach ~1.3×:** with the overhead floor amortized, the kernel's efficiency
  dominates. This would be higher (~2.0×), but the multi-warp synchronization (`tl.debug_barrier`) required to avoid a race condition on `log_alpha` rows heavily penalizes the hot loop.
- Kernel-only timings (CUDA events, FlagGems' generic benchmark shapes, T=64) showed our
  kernel up to ~8× faster than ATen's — which is precisely why end-to-end, the only metric
  that matters, had to be fixed on the CPU side.

### Honest limitations

1. **Backward is not custom** — it reuses PyTorch's native CUDA kernel via dispatch-time
   capture. Correct and zero-maintenance, but no training-side speedup.
2. **Performance Trade-offs for Correctness** — To make the op genuinely correct (handling FP32 gradients correctly across long sequences and passing all tests) we had to wrap the fused kernel in a `torch.autograd.Function` and add `tl.debug_barrier()` cross-warp synchronization in the hot inner loop. This slashed our peak speedup (large shape fell from 2.0x to 1.29x). Correctness gated speed.
3. **The 1D concatenated-target layout** still incurs a `.max().item()` sync and a
   per-program length scan — correct but unoptimized (the standard 2D layout is the fast path).
4. **`tl.atomic_add` makes mean/sum reduction order non-deterministic** — within fp32
   tolerance, verified by tests, but worth stating.

---

## Documentation (/15)

- **`CHANGELOG.md`** — every round logged: what changed, why, what was measured, what
  went wrong (including the regression scare that turned out to be baseline drift)
- **`plans/`** — the full audit trail: each round's plan archived with its outcome banner
- **`plan.md`** — the live plan; hypotheses, file-level instructions, verification criteria
- **Inline comments** — non-obvious decisions explained at the point of code
- **`CLAUDE.md`** — repo conventions, including the dispatcher-shadowing gotcha
  (an op override must snapshot the native kernel *at import time* or it recurses)

```
src/flag_gems/ops/ctc_loss.py    forward kernel + fused reduction + backward capture (~320 lines)
tests/test_ctc_loss.py           correctness: 3 shapes × 3 reductions, backward, zero_infinity
benchmark/test_ctc_loss.py       FlagGems benchmark integration
CHANGELOG.md                     full optimization history
plans/                           archived round plans
```

---

## Presentation (/10)

### Three takeaways

1. **Diagnose from the shape of the gap.** A constant absolute deficit across problem sizes
   means fixed per-call overhead — no amount of GPU optimization will move it. This one
   observation correctly predicted that kernel launches, not kernel math, were the bottleneck.

2. **Launch count beats kernel speed at small scale.** Fusing two launches into one
   (forward + reduction via `tl.atomic_add`) did more for end-to-end latency than every
   GPU-side optimization combined.

3. **Inside an op-override library, your own wrapper is the trap.** Any `torch.*` call in
   the wrapper dispatches back into the library's overridden kernels. The fix is to push
   work into the kernel (pointer math, fused reduction) — not to call more PyTorch.

### The journey (small shape, end-to-end vs PyTorch)

```
Round 0  ATen passthrough      ████▊                 0.24×
Round 1  Triton forward        █████████             0.45×
Round 2  composite override    █████████▍            0.47×
Round 3a libentry + offsets    █████████             ~0.45×
Round 3b single fused launch   ████████████████████  0.98–1.21×   (pre-correctness peak)
Rounds 4-6 Race fix + autograd ████████████          0.697×       (small shape final)
         medium / large        ████████████████████████  1.22–1.29×   (final)
```
