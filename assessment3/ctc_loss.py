import logging
import torch
import triton
import triton.language as tl
from flag_gems.runtime import device as runtime_device
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_DISPATCH_KEY = runtime_device.dispatch_key

def _capture_native(qualname):
    try:
        kernel = torch.library.get_kernel(qualname, _DISPATCH_KEY)
        keyset = torch._C.DispatchKeySet(getattr(torch._C.DispatchKey, _DISPATCH_KEY))
        return kernel, keyset
    except Exception:
        return None, None

_NATIVE_BWD_INTLIST, _BWD_INTLIST_KEYSET = _capture_native("aten::_ctc_loss_backward")
_NATIVE_BWD_TENSOR, _BWD_TENSOR_KEYSET = _capture_native("aten::_ctc_loss_backward.Tensor")

@libentry()
@triton.jit
def ctc_loss_fwd_kernel(
    lp_ptr, targets_ptr, input_lengths_ptr, target_lengths_ptr,
    log_alpha_ptr, nll_ptr, out_ptr,
    lp_stride_t, lp_stride_n, lp_stride_c,
    tgt_stride_n, tgt_stride_s,
    la_stride_n, la_stride_t, la_stride_s,
    T, N, C, MAX_S, BLANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    REDUCTION: tl.constexpr,
    ZERO_INF: tl.constexpr,
    SYNC: tl.constexpr,
):
    n = tl.program_id(0)

    # Load sequence lengths for this batch element
    Tn = tl.load(input_lengths_ptr + n)
    Sn = tl.load(target_lengths_ptr + n)
    L = 2 * Sn + 1

    s = tl.arange(0, BLOCK_S)
    valid_s = s < L

    # Compute extended labels
    is_odd = (s % 2) == 1
    tgt_idx = (s - 1) // 2

    if HAS_OFFSETS:
        tgt_base = 0
        for i in range(n):
            tgt_base += tl.load(target_lengths_ptr + i).to(tl.int32)
    else:
        tgt_base = n * tgt_stride_n
    
    # Safe load target_idx to avoid negative offset
    safe_target_idx = tgt_base + tl.maximum(tgt_idx, 0) * tgt_stride_s
    labels = tl.where(is_odd,
                      tl.load(targets_ptr + safe_target_idx, mask=is_odd & valid_s, other=BLANK),
                      BLANK)

    safe_target_idx_m1 = tgt_base + tl.maximum(tgt_idx - 1, 0) * tgt_stride_s
    labels_m2 = tl.where(is_odd & (tgt_idx >= 1),
                         tl.load(targets_ptr + safe_target_idx_m1, mask=is_odd & (tgt_idx >= 1) & valid_s, other=BLANK),
                         BLANK)

    skip_ok = is_odd & (tgt_idx >= 1) & (labels != labels_m2)

    # Initialize t = 0
    lp_t0_n = lp_ptr + 0 * lp_stride_t + n * lp_stride_n
    lp_0_blank = tl.load(lp_t0_n + BLANK * lp_stride_c)
    
    alpha = tl.where(s == 0, lp_0_blank, float("-inf"))
    if Sn > 0:
        l_1 = tl.load(targets_ptr + tgt_base + 0 * tgt_stride_s)
        lp_0_l1 = tl.load(lp_t0_n + l_1 * lp_stride_c)
        alpha = tl.where(s == 1, lp_0_l1, alpha)
        
    alpha = tl.where(valid_s, alpha, float("-inf"))
    
    # Store to log_alpha[n, 0, :]
    t0_row_ptr = log_alpha_ptr + n * la_stride_n + 0 * la_stride_t + s * la_stride_s
    tl.store(t0_row_ptr, alpha, mask=s < (2 * MAX_S + 1))
    if SYNC:
        tl.debug_barrier()

    # Loop over t
    for t in range(1, T):
        prev_row_ptr = log_alpha_ptr + n * la_stride_n + (t - 1) * la_stride_t
        a0 = tl.load(prev_row_ptr + s * la_stride_s, mask=valid_s, other=float("-inf"))
        
        a1_ptr = prev_row_ptr + tl.maximum(s - 1, 0) * la_stride_s
        a1 = tl.load(a1_ptr, mask=(s >= 1) & valid_s, other=float("-inf"))
        
        a2_ptr = prev_row_ptr + tl.maximum(s - 2, 0) * la_stride_s
        a2 = tl.load(a2_ptr, mask=(s >= 2) & valid_s, other=float("-inf"))
        
        a2_valid = tl.where(skip_ok, a2, float("-inf"))
        
        m1 = tl.maximum(a0, a1)
        m = tl.maximum(m1, a2_valid)
        
        safe_m = tl.where(m == float("-inf"), 0.0, m)
        sum_exp = tl.exp(a0 - safe_m) + tl.exp(a1 - safe_m) + tl.exp(a2_valid - safe_m)
        a_prev = tl.where(m == float("-inf"), float("-inf"), m + tl.log(sum_exp))
        
        lp_t_n = lp_ptr + t * lp_stride_t + n * lp_stride_n
        lp_val = tl.load(lp_t_n + labels * lp_stride_c, mask=valid_s, other=float("-inf"))
        
        alpha_new = a_prev + lp_val
        
        # Freezing: if t >= Tn, we keep the previous row's values
        alpha_new = tl.where(t < Tn, alpha_new, a0)
        alpha_new = tl.where(valid_s, alpha_new, float("-inf"))
        
        curr_row_ptr = log_alpha_ptr + n * la_stride_n + t * la_stride_t + s * la_stride_s
        tl.store(curr_row_ptr, alpha_new, mask=s < (2 * MAX_S + 1))
        if SYNC:
            tl.debug_barrier()

    # Readout nll
    final_t = tl.maximum(Tn - 1, 0)
    final_row_ptr = log_alpha_ptr + n * la_stride_n + final_t * la_stride_t
    a_L1 = tl.load(final_row_ptr + (L - 1) * la_stride_s)
    a_L2_raw = tl.load(final_row_ptr + tl.maximum(L - 2, 0) * la_stride_s)
    a_L2 = tl.where(L >= 2, a_L2_raw, float("-inf"))
    
    m_final = tl.maximum(a_L1, a_L2)
    safe_m_final = tl.where(m_final == float("-inf"), 0.0, m_final)
    nll_val = tl.where(m_final == float("-inf"), float("-inf"), m_final + tl.log(tl.exp(a_L1 - safe_m_final) + tl.exp(a_L2 - safe_m_final)))
    
    nll_val = tl.where(Sn == 0, a_L1, nll_val)
    nll_val = -nll_val
    
    tl.store(nll_ptr + n, nll_val)

    # --- Inline reduction (avoids a second kernel launch) ---
    if REDUCTION > 0:
        reduced_val = nll_val
        if ZERO_INF:
            is_inf = (reduced_val == float("inf")) | (reduced_val == float("-inf"))
            reduced_val = tl.where(is_inf, 0.0, reduced_val)
        if REDUCTION == 1:  # mean
            t_len = tl.load(target_lengths_ptr + n)
            t_len_clamped = tl.maximum(t_len, 1)
            contrib = reduced_val / t_len_clamped.to(reduced_val.dtype) / N
            tl.atomic_add(out_ptr, contrib)
        elif REDUCTION == 2:  # sum
            tl.atomic_add(out_ptr, reduced_val)


@libentry()
@triton.jit
def ctc_loss_reduction_kernel(
    nll_ptr, target_lengths_ptr, out_ptr, 
    N, REDUCTION: tl.constexpr, ZERO_INF: tl.constexpr, BLOCK_N: tl.constexpr
):
    n = tl.arange(0, BLOCK_N)
    mask = n < N
    
    nll = tl.load(nll_ptr + n, mask=mask, other=0.0)
    if ZERO_INF:
        nll = tl.where(nll == float("inf"), 0.0, nll)
        nll = tl.where(nll == float("-inf"), 0.0, nll)
    
    if REDUCTION == 0: # none
        tl.store(out_ptr + n, nll, mask=mask)
    elif REDUCTION == 1: # mean
        t_len = tl.load(target_lengths_ptr + n, mask=mask, other=1)
        t_len_clamped = tl.maximum(t_len, 1)
        val = nll / t_len_clamped
        sum_val = tl.sum(val, axis=0)
        if tl.program_id(0) == 0:
            tl.store(out_ptr, sum_val / N)
    elif REDUCTION == 2: # sum
        sum_val = tl.sum(nll, axis=0)
        if tl.program_id(0) == 0:
            tl.store(out_ptr, sum_val)

def _ctc_loss(log_probs, targets, input_lengths, target_lengths_in, blank=0, zero_infinity=False,
              _reduction=0, _out=None):
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths, dtype=torch.int64, device=log_probs.device)
    if not isinstance(target_lengths_in, torch.Tensor):
        target_lengths = torch.tensor(target_lengths_in, dtype=torch.int64, device=log_probs.device)
    else:
        target_lengths = target_lengths_in

    # .to(device, non_blocking=True) is a no-op if already on device
    input_lengths = input_lengths.to(log_probs.device, non_blocking=True)
    target_lengths = target_lengths.to(log_probs.device, non_blocking=True)

    T = log_probs.size(0)
    N = log_probs.size(1)
    C = log_probs.size(2)

    if targets.dim() == 1:
        if isinstance(target_lengths_in, torch.Tensor):
            max_S = int(target_lengths_in.max().item()) if N > 0 else 0
        else:
            max_S = int(max(target_lengths_in)) if N > 0 else 0
        tgt_stride_n = 0
        tgt_stride_s = targets.stride(0)
        HAS_OFFSETS = True
    else:
        max_S = targets.size(1)
        tgt_stride_n = targets.stride(0)
        tgt_stride_s = targets.stride(1)
        HAS_OFFSETS = False

    log_alpha = torch.empty((N, T, 2 * max_S + 1), dtype=log_probs.dtype, device=log_probs.device)
    nll = torch.empty(N, dtype=log_probs.dtype, device=log_probs.device)

    # Use nll as dummy out_ptr when no reduction requested
    out_ptr = _out if _out is not None else nll

    if N > 0 and T > 0:
        BLOCK_S = triton.next_power_of_2(2 * max_S + 1)
        BLOCK_S = max(16, BLOCK_S)
        num_warps = min(8, max(1, BLOCK_S // 32))

        ctc_loss_fwd_kernel[(N,)](
            log_probs, targets, input_lengths, target_lengths,
            log_alpha, nll, out_ptr,
            log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
            tgt_stride_n, tgt_stride_s,
            log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
            T, N, C, max_S, blank,
            BLOCK_S=BLOCK_S,
            HAS_OFFSETS=HAS_OFFSETS,
            REDUCTION=_reduction,
            ZERO_INF=zero_infinity if _reduction > 0 else False,
            SYNC=(num_warps > 1),
            num_warps=num_warps,
        )

    return nll, log_alpha


def _ctc_loss_inference(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity):
    N = log_probs.size(1)

    if reduction == 0:  # none
        nll, log_alpha = _ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank, zero_infinity,
        )
        if N == 0:
            out = torch.zeros(0, dtype=log_probs.dtype, device=log_probs.device)
        elif zero_infinity:
            out = torch.where(torch.isinf(nll), torch.zeros_like(nll), nll)
        else:
            out = nll
    else:
        if N == 0:
            out = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            nll = out
            log_alpha = out
        else:
            out = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
            nll, log_alpha = _ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank, zero_infinity, _reduction=reduction, _out=out,
            )

    return out, nll, log_alpha


class _CTCLossFunction(torch.autograd.Function):
    """Custom autograd function that wraps the fused Triton forward kernel
    and delegates backward to the captured native _ctc_loss_backward kernel.
    This gives us single-kernel-launch performance in the forward while
    maintaining correct gradient propagation for training."""

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths,
                blank, reduction, zero_infinity):
        N = log_probs.size(1)

        out, nll, log_alpha = _ctc_loss_inference(
            log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity
        )

        # Save tensors for backward (only non-integer tensors can be saved)
        if not isinstance(input_lengths, torch.Tensor):
            input_lengths = torch.tensor(input_lengths, dtype=torch.int64, device=log_probs.device)
        if not isinstance(target_lengths, torch.Tensor):
            target_lengths = torch.tensor(target_lengths, dtype=torch.int64, device=log_probs.device)
        input_lengths = input_lengths.to(log_probs.device, non_blocking=True)
        target_lengths = target_lengths.to(log_probs.device, non_blocking=True)

        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, nll, log_alpha)
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        ctx.reduction = reduction
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, nll, log_alpha = ctx.saved_tensors
        N = ctx.N
        reduction = ctx.reduction

        # Compute grad_nll: the gradient of the reduced loss w.r.t. nll
        if reduction == 0:  # none
            grad_nll = grad_output
        elif reduction == 1:  # mean
            tl_clamped = target_lengths.clamp(min=1).to(grad_output.dtype)
            grad_nll = grad_output / (N * tl_clamped)
        elif reduction == 2:  # sum
            grad_nll = grad_output.expand(N).contiguous()

        if ctx.zero_infinity:
            grad_nll = torch.where(torch.isinf(nll), torch.zeros_like(grad_nll), grad_nll)

        # Call the native backward kernel
        grad_input = _ctc_loss_backward(
            grad_nll, log_probs, targets,
            input_lengths, target_lengths,
            nll, log_alpha,
            ctx.blank, ctx.zero_infinity,
        )

        # Return grads for: log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity
        return grad_input, None, None, None, None, None, None


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction=1, zero_infinity=False):
    if torch.is_grad_enabled() and log_probs.requires_grad:
        return _CTCLossFunction.apply(
            log_probs, targets, input_lengths, target_lengths,
            blank, reduction, zero_infinity,
        )
    out, _, _ = _ctc_loss_inference(
        log_probs, targets, input_lengths, target_lengths,
        blank, reduction, zero_infinity,
    )
    return out


def _ctc_loss_backward(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank=0,
    zero_infinity=False,
):
    if isinstance(input_lengths, torch.Tensor):
        if _NATIVE_BWD_TENSOR is not None:
            return _NATIVE_BWD_TENSOR.call_boxed(
                _BWD_TENSOR_KEYSET,
                grad,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
                blank,
                zero_infinity,
            )
        input_lengths_list = input_lengths.tolist()
        target_lengths_list = target_lengths.tolist()
    else:
        if _NATIVE_BWD_INTLIST is not None:
            return _NATIVE_BWD_INTLIST.call_boxed(
                _BWD_INTLIST_KEYSET,
                grad,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                neg_log_likelihood,
                log_alpha,
                blank,
                zero_infinity,
            )
        input_lengths_list = list(input_lengths)
        target_lengths_list = list(target_lengths)

    device = log_probs.device
    grad_input = torch.ops.aten._ctc_loss_backward.default(
        grad.cpu(),
        log_probs.cpu(),
        targets.cpu(),
        input_lengths_list,
        target_lengths_list,
        neg_log_likelihood.cpu(),
        log_alpha.cpu(),
        blank,
        zero_infinity,
    )
    return grad_input.to(device)
