import torch
import triton
import triton.language as tl


@triton.jit
def _gemm1_kernel(A, ASC, TOK, W, WS, C, Tk, T,
                  H: tl.constexpr, I: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr,
                  FP8: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < Tk
    tok = tl.load(TOK + offs_m, mask=mask_m, other=0).to(tl.int64)
    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    nb = n0 // 128
    for k in range(H // 128):
        a = tl.load(A + tok[:, None] * H + k * 128 + offs_k[None, :],
                    mask=mask_m[:, None], other=0.0)
        asc = tl.load(ASC + k * T + tok, mask=mask_m, other=0.0)
        w1 = tl.load(W + offs_n[None, :].to(tl.int64) * H + k * 128 + offs_k[:, None])
        w2 = tl.load(W + (offs_n[None, :] + I).to(tl.int64) * H + k * 128 + offs_k[:, None])
        if not FP8:
            a = a.to(tl.bfloat16)
            w1 = w1.to(tl.bfloat16)
            w2 = w2.to(tl.bfloat16)
        s1 = tl.load(WS + nb * 56 + k)
        s2 = tl.load(WS + (16 + nb) * 56 + k)
        d1 = tl.dot(a, w1)
        d2 = tl.dot(a, w2)
        acc1 += d1 * (asc[:, None] * s1)
        acc2 += d2 * (asc[:, None] * s2)
    res = acc2 / (1.0 + tl.exp(-acc2)) * acc1
    tl.store(C + offs_m[:, None] * I + offs_n[None, :], res,
             mask=mask_m[:, None])


@triton.jit
def _gemm2_kernel(C, TOK, WT, W, WS, OUT, Tk,
                  H: tl.constexpr, I: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < Tk
    tok = tl.load(TOK + offs_m, mask=mask_m, other=0).to(tl.int64)
    wt = tl.load(WT + offs_m, mask=mask_m, other=0.0)
    n0 = pid_n * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    nb = n0 // 128
    for k in range(I // 128):
        c = tl.load(C + offs_m[:, None] * I + k * 128 + offs_k[None, :],
                    mask=mask_m[:, None], other=0.0)
        w = tl.load(W + offs_n[None, :].to(tl.int64) * I + k * 128 + offs_k[:, None]).to(tl.float32)
        s = tl.load(WS + nb * 16 + k)
        d = tl.dot(c, w, input_precision="tf32x3")
        acc += d * s
    optr = OUT + tok[:, None] * H + offs_n[None, :]
    o = tl.load(optr, mask=mask_m[:, None], other=0.0)
    tl.store(optr, o + acc * wt[:, None], mask=mask_m[:, None])


_USE_FP8 = [False]
_BIG_OK = [False]
_BIG2_OK = [False]


def _launch_gemm1(hs, asc, token_idx, w, ws, C, Tk, T, H, I, BM, BN):
    if BM == 64 and Tk > 128 and _BIG_OK[0]:
        try:
            bm = 128
            grid = (triton.cdiv(Tk, bm), I // BN)
            _gemm1_kernel[grid](hs, asc, token_idx, w, ws, C, Tk, T,
                                H=H, I=I, BM=bm, BN=BN, FP8=_USE_FP8[0],
                                num_warps=8, num_stages=2)
            return
        except Exception:
            _BIG_OK[0] = False
    grid1 = (triton.cdiv(Tk, BM), I // BN)
    if _USE_FP8[0]:
        try:
            _gemm1_kernel[grid1](hs, asc, token_idx, w, ws, C, Tk, T,
                                 H=H, I=I, BM=BM, BN=BN, FP8=True, num_warps=4)
            return
        except Exception:
            _USE_FP8[0] = False
    _gemm1_kernel[grid1](hs, asc, token_idx, w, ws, C, Tk, T,
                         H=H, I=I, BM=BM, BN=BN, FP8=False, num_warps=4)


def _launch_gemm2(C, token_idx, wt, w, ws, out, Tk, H, I, BM, BN):
    if BM == 64 and Tk > 128 and _BIG2_OK[0]:
        try:
            grid = (triton.cdiv(Tk, 128), H // BN)
            _gemm2_kernel[grid](C, token_idx, wt, w, ws, out, Tk,
                                H=H, I=I, BM=128, BN=BN,
                                num_warps=8, num_stages=2)
            return
        except Exception:
            _BIG2_OK[0] = False
    grid2 = (triton.cdiv(Tk, BM), H // BN)
    _gemm2_kernel[grid2](C, token_idx, wt, w, ws, out, Tk,
                         H=H, I=I, BM=BM, BN=BN, num_warps=4)


@torch.no_grad()
def kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias
    grouped = s_with_bias.view(T, 8, 32)
    top2, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, 8, 32).reshape(T, E_global)
    neg_inf = torch.finfo(torch.float32).min
    pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(pruned, k=8, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor

    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    hs = hidden_states.contiguous()
    asc = hidden_states_scale.to(torch.float32).contiguous()
    ws1 = gemm1_weights_scale.to(torch.float32).contiguous()
    ws2 = gemm2_weights_scale.to(torch.float32).contiguous()
    local_start = int(local_expert_offset)

    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_tok = (order // 8).to(torch.int32).contiguous()
    sorted_e = flat[order]
    sorted_w = weights[(order // 8), sorted_e].contiguous()
    counts = torch.bincount(flat, minlength=E_global).cpu().tolist()
    starts = [0] * (E_global + 1)
    for i in range(E_global):
        starts[i + 1] = starts[i] + counts[i]

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        Tk = counts[ge]
        if Tk == 0:
            continue
        a0 = starts[ge]
        token_idx = sorted_tok[a0:a0 + Tk]
        wt = sorted_w[a0:a0 + Tk]
        BM = 32
        BN = 64
        C = torch.empty((Tk, I), dtype=torch.float32, device=device)
        _launch_gemm1(hs, asc, token_idx, gemm1_weights[le], ws1[le],
                      C, Tk, T, H, I, BM, BN)
        _launch_gemm2(C, token_idx, wt, gemm2_weights[le], ws2[le], out, Tk,
                      H, I, BM, BN)

    return out.to(torch.bfloat16)
