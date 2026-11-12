"""
Hand-written (non-ML) Triton kernel for the FlashInfer MoE definition
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.

Baseline for the Claude-tier experiment. No model wrote this; its speed comes only from the
tile settings below, which baseline_search.py tunes (random search / Triton autotune) at the
same budget of timed candidates as each Claude run.

Design (standard grouped-GEMM MoE, no tricks):
  1. Routing in PyTorch, identical to the reference (DeepSeek-V3 no-aux, top-8 of 4 groups).
  2. (token, local expert) pairs sorted by expert.
  3. For each local expert with tokens:
       GEMM1 + SwiGLU (one Triton kernel): FP8 tokens x FP8 W13, block scales applied per
       128-wide K block, fp32 accumulation, epilogue C = X1 * silu(X2).
       GEMM2 + scatter (one Triton kernel): C x FP8 W2, block scales, times the routing
       weight, added into the fp32 output rows of those tokens.
  4. Output cast to bfloat16.
Weights stay in FP8 in memory; they are dequantized per tile inside the kernels.
"""
import torch
import triton
import triton.language as tl

# ---- tunable settings (baseline_search.py rewrites these three lines) ----
CFG1 = {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 4}
CFG2 = {'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 4, 'PREC': 'ieee'}
AUTOTUNE = None

H = 7168
I = 2048
BLOCK = 128


@triton.jit
def _gemm1_swiglu(A, AS, TOK, W, WS, C, Tk, T, TK_BUCKET,
                  H: tl.constexpr, I: tl.constexpr, NKB: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Tk
    tok = tl.load(TOK + offs_m, mask=mask_m, other=0)          # rows past Tk read token 0 (never stored)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # column in [0, I)
    offs_k = tl.arange(0, BLOCK_K)
    nb1 = (pid_n * BLOCK_N) // 128                              # scale row for X1 columns
    nb2 = nb1 + I // 128                                        # scale row for X2 columns
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        kb = k0 // 128
        a = tl.load(A + tok[:, None] * H + (k0 + offs_k)[None, :]).to(tl.bfloat16)
        a_s = tl.load(AS + kb * T + tok)
        w1 = tl.load(W + offs_n[:, None] * H + (k0 + offs_k)[None, :]).to(tl.bfloat16)
        w2 = tl.load(W + (offs_n + I)[:, None] * H + (k0 + offs_k)[None, :]).to(tl.bfloat16)
        s1 = tl.load(WS + nb1 * NKB + kb)
        s2 = tl.load(WS + nb2 * NKB + kb)
        acc1 += tl.dot(a, tl.trans(w1)) * (a_s[:, None] * s1)
        acc2 += tl.dot(a, tl.trans(w2)) * (a_s[:, None] * s2)
    c = acc1 * (acc2 / (1.0 + tl.exp(-acc2)))
    tl.store(C + offs_m[:, None] * I + offs_n[None, :], c, mask=mask_m[:, None])


@triton.jit
def _gemm2_scatter(C, W, WS, TOK, WTOK, OUT, Tk, TK_BUCKET,
                   H: tl.constexpr, I: tl.constexpr, NKB2: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   PREC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Tk
    tok = tl.load(TOK + offs_m, mask=mask_m, other=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # column in [0, H)
    offs_k = tl.arange(0, BLOCK_K)
    nb = (pid_n * BLOCK_N) // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        kb = k0 // 128
        a = tl.load(C + offs_m[:, None] * I + (k0 + offs_k)[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(W + offs_n[:, None] * I + (k0 + offs_k)[None, :]).to(tl.float32)
        s = tl.load(WS + nb * NKB2 + kb)
        acc += tl.dot(a, tl.trans(w), input_precision=PREC) * s
    wt = tl.load(WTOK + offs_m, mask=mask_m, other=0.0)
    acc = acc * wt[:, None]
    ptr = OUT + tok[:, None] * H + offs_n[None, :]
    prev = tl.load(ptr, mask=mask_m[:, None], other=0.0)
    tl.store(ptr, prev + acc, mask=mask_m[:, None])     # experts run one after another: no races


def _configs(lst, extra=None):
    out = []
    for c in lst:
        meta = {k: c[k] for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K")}
        if extra:
            meta.update(extra)
        out.append(triton.Config(meta, num_warps=c["num_warps"], num_stages=c["num_stages"]))
    return out


if AUTOTUNE:
    _g1 = triton.autotune(configs=_configs(AUTOTUNE), key=["TK_BUCKET"])(_gemm1_swiglu)
    _g2 = triton.autotune(configs=_configs(AUTOTUNE, {"PREC": CFG2["PREC"]}), key=["TK_BUCKET"],
                          restore_value=["OUT"])(_gemm2_scatter)
else:
    _g1, _g2 = _gemm1_swiglu, _gemm2_scatter


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    device = hidden_states.device
    TOP_K, N_GROUP, TOPK_GROUP = 8, 8, 4

    # ---- 1) routing: same as the reference ----
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias
    group_size = E_global // N_GROUP
    top2_vals, _ = torch.topk(s_with_bias.view(T, N_GROUP, group_size), k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor

    # ---- 2) (token, local expert) pairs sorted by expert ----
    local_start = int(local_expert_offset)
    local = topk_idx - local_start
    valid = (local >= 0) & (local < E_local)
    flat_t = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K)[valid]
    flat_e = local[valid]
    order = torch.argsort(flat_e, stable=True)
    flat_t = flat_t[order]
    flat_e = flat_e[order]
    w_pair = weights[flat_t, flat_e + local_start].contiguous()
    tok_pair = flat_t.to(torch.int32).contiguous()
    counts = torch.bincount(flat_e, minlength=E_local).tolist()

    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    if not tok_pair.numel():
        return out.to(torch.bfloat16)

    hs = hidden_states.contiguous()
    hs_scale = hidden_states_scale.to(torch.float32).contiguous()      # [H/128, T]
    s13 = gemm1_weights_scale.to(torch.float32).contiguous()           # [E, 2I/128, H/128]
    s2 = gemm2_weights_scale.to(torch.float32).contiguous()            # [E, H/128, I/128]
    C = torch.empty((max(counts), I), dtype=torch.float32, device=device)

    # ---- 3) per-expert GEMM1+SwiGLU, GEMM2+scatter ----
    start = 0
    for le, n in enumerate(counts):
        if n == 0:
            continue
        tok = tok_pair[start:start + n]
        wt = w_pair[start:start + n]
        bucket = triton.next_power_of_2(n)
        if AUTOTUNE:
            _g1[lambda m: (triton.cdiv(n, m["BLOCK_M"]), I // m["BLOCK_N"])](
                hs, hs_scale, tok, gemm1_weights[le], s13[le], C, n, T, bucket,
                H=H, I=I, NKB=H // BLOCK)
            _g2[lambda m: (triton.cdiv(n, m["BLOCK_M"]), H // m["BLOCK_N"])](
                C, gemm2_weights[le], s2[le], tok, wt, out, n, bucket,
                H=H, I=I, NKB2=I // BLOCK)
        else:
            c1, c2 = CFG1, CFG2
            _g1[(triton.cdiv(n, c1["BLOCK_M"]), I // c1["BLOCK_N"])](
                hs, hs_scale, tok, gemm1_weights[le], s13[le], C, n, T, bucket,
                H=H, I=I, NKB=H // BLOCK,
                BLOCK_M=c1["BLOCK_M"], BLOCK_N=c1["BLOCK_N"], BLOCK_K=c1["BLOCK_K"],
                num_warps=c1["num_warps"], num_stages=c1["num_stages"])
            _g2[(triton.cdiv(n, c2["BLOCK_M"]), H // c2["BLOCK_N"])](
                C, gemm2_weights[le], s2[le], tok, wt, out, n, bucket,
                H=H, I=I, NKB2=I // BLOCK,
                BLOCK_M=c2["BLOCK_M"], BLOCK_N=c2["BLOCK_N"], BLOCK_K=c2["BLOCK_K"], PREC=c2["PREC"],
                num_warps=c2["num_warps"], num_stages=c2["num_stages"])
        start += n

    # ---- 4) output ----
    return out.to(torch.bfloat16)
