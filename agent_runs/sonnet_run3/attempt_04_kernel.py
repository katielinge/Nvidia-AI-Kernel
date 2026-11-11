import torch
import triton
import triton.language as tl


@triton.jit
def _gemm1_kernel(
    hs_ptr, hss_ptr, w1_ptr, w1s_ptr, c_ptr,
    tile_e_ptr, tile_row0_ptr, tile_end_ptr, tok_ptr,
    T,
    BM: tl.constexpr, BN: tl.constexpr,
):
    H: tl.constexpr = 7168
    I: tl.constexpr = 2048
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(tile_row0_ptr + pid_m)
    end = tl.load(tile_end_ptr + pid_m)
    if row0 < end:
        e = tl.load(tile_e_ptr + pid_m).to(tl.int64)
        offs_m = row0 + tl.arange(0, BM)
        mmask = offs_m < end
        tok = tl.load(tok_ptr + offs_m, mask=mmask, other=0).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, 128)
        n0 = pid_n * BN
        acc1 = tl.zeros((BM, BN), dtype=tl.float32)
        acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        w1_base = w1_ptr + e * (2 * I * H)
        for kb in range(0, H // 128):
            k = kb * 128 + offs_k
            a = tl.load(hs_ptr + tok[:, None] * H + k[None, :], mask=mmask[:, None], other=0.0)
            a_s = tl.load(hss_ptr + kb * T + tok, mask=mmask, other=0.0)
            wp1 = w1_base + offs_n[None, :].to(tl.int64) * H + k[:, None]
            wp2 = w1_base + (offs_n[None, :] + I).to(tl.int64) * H + k[:, None]
            wa = tl.load(wp1)
            wb = tl.load(wp2)
            s1 = tl.load(w1s_ptr + e * (32 * 56) + (n0 // 128) * 56 + kb)
            s2 = tl.load(w1s_ptr + e * (32 * 56) + (16 + n0 // 128) * 56 + kb)
            d1 = tl.dot(a, wa)
            d2 = tl.dot(a, wb)
            acc1 += d1 * (a_s[:, None] * s1)
            acc2 += d2 * (a_s[:, None] * s2)
        c = acc2 / (1.0 + tl.exp(-acc2)) * acc1
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :],
                 c, mask=mmask[:, None])


@triton.jit
def _gemm2_kernel(
    c_ptr, w2_ptr, w2s_ptr, out_ptr,
    tile_e_ptr, tile_row0_ptr, tile_end_ptr, tok_ptr, wt_ptr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    H: tl.constexpr = 7168
    I: tl.constexpr = 2048
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row0 = tl.load(tile_row0_ptr + pid_m)
    end = tl.load(tile_end_ptr + pid_m)
    if row0 < end:
        e = tl.load(tile_e_ptr + pid_m).to(tl.int64)
        offs_m = row0 + tl.arange(0, BM)
        mmask = offs_m < end
        tok = tl.load(tok_ptr + offs_m, mask=mmask, other=0).to(tl.int64)
        wt = tl.load(wt_ptr + offs_m, mask=mmask, other=0.0)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, 128)
        n0 = pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        w_base = w2_ptr + e * (H * I)
        for kb in range(0, I // 128):
            k = kb * 128 + offs_k
            cc = tl.load(c_ptr + offs_m[:, None].to(tl.int64) * I + k[None, :],
                         mask=mmask[:, None], other=0.0)
            w = tl.load(w_base + offs_n[None, :].to(tl.int64) * I + k[:, None]).to(tl.float32)
            s = tl.load(w2s_ptr + e * (56 * 16) + (n0 // 128) * 16 + kb)
            acc += tl.dot(cc, w, input_precision="tf32x3") * s
        res = acc * wt[:, None]
        tl.atomic_add(out_ptr + tok[:, None] * H + offs_n[None, :], res, mask=mmask[:, None])


def _build_tiles(counts, BM, T, device, E_local):
    tiles_per = (counts + BM - 1) // BM
    tile_cum = torch.cumsum(tiles_per, 0)
    tile_start = tile_cum - tiles_per
    pair_start = torch.cumsum(counts, 0) - counts
    max_tiles = (T * 8) // BM + E_local
    tids = torch.arange(max_tiles, device=device)
    e_raw = torch.searchsorted(tile_cum, tids, right=True)
    inval = e_raw >= E_local
    e_c = e_raw.clamp(max=E_local - 1)
    end = pair_start[e_c] + counts[e_c]
    row0 = pair_start[e_c] + (tids - tile_start[e_c]) * BM
    end = torch.where(inval, torch.zeros_like(end), end)
    row0 = torch.where(inval, torch.zeros_like(row0), row0)
    return (max_tiles, e_c.to(torch.int32).contiguous(),
            row0.to(torch.int32).contiguous(), end.to(torch.int32).contiguous())


@torch.no_grad()
def kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    H = 7168
    I = 2048
    E_local = 32
    E_global = 256
    T = routing_logits.shape[0]
    device = hidden_states.device

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias
    grouped = s_with_bias.view(T, 8, 32)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, 8, 32).reshape(T, E_global)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=8, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor
    wt_pairs = weights.gather(1, topk_idx)

    BM = 16 if T <= 256 else (32 if T <= 1024 else 64)

    le = topk_idx - int(local_expert_offset)
    valid = (le >= 0) & (le < E_local)
    key = torch.where(valid, le, torch.full_like(le, E_local)).flatten()
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=E_local + 1)[:E_local]
    sorted_tok = (order // 8).to(torch.int32).contiguous()
    sorted_w = wt_pairs.flatten()[order].to(torch.float32).contiguous()

    hss = hidden_states_scale.contiguous()
    w1s = gemm1_weights_scale.contiguous()
    w2s = gemm2_weights_scale.contiguous()

    Cbuf = torch.empty((T * 8, I), dtype=torch.float32, device=device)
    out = torch.zeros((T, H), dtype=torch.float32, device=device)

    if T > 1024:
        BM1, nw1, ns1 = 128, 8, 2
    else:
        BM1, nw1, ns1 = BM, 4, 3
    mt1, e1, r1, en1 = _build_tiles(counts, BM1, T, device, E_local)
    BN1 = 64
    _gemm1_kernel[(mt1, I // BN1)](
        hidden_states, hss, gemm1_weights, w1s, Cbuf,
        e1, r1, en1, sorted_tok, T,
        BM=BM1, BN=BN1, num_warps=nw1, num_stages=ns1,
    )
    if BM1 == BM:
        mt2, e2, r2, en2 = mt1, e1, r1, en1
    else:
        mt2, e2, r2, en2 = _build_tiles(counts, BM, T, device, E_local)
    if T > 1024:
        BN2, nw2, ns2 = 64, 8, 1
    else:
        BN2, nw2, ns2 = 32, 4, 2
    _gemm2_kernel[(mt2, H // BN2)](
        Cbuf, gemm2_weights, w2s, out,
        e2, r2, en2, sorted_tok, sorted_w,
        BM=BM, BN=BN2, num_warps=nw2, num_stages=ns2,
    )
    return out.to(torch.bfloat16)
