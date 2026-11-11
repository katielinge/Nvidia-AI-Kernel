import torch
import triton
import triton.language as tl

H_DIM = 7168
I_DIM = 2048
E_LOCAL = 32
TOP_K = 8
CBLK = 64  # column block of C for per-row fp16 scaling (== BN1 == BK2)


@triton.jit
def _gemm1_swiglu_kernel(
    hs_ptr, hs_scale_ptr, w13_ptr, s13_ptr,
    sorted_tok_ptr, tile_expert_ptr, c_ptr, cs_ptr, T,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(tile_expert_ptr + pid_m)
    if e >= 32:
        return
    rows = pid_m * BM + tl.arange(0, BM)
    tok = tl.load(sorted_tok_ptr + rows)
    valid = tok >= 0
    tok = tl.where(valid, tok, 0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = hs_ptr + tok.to(tl.int64)[:, None] * 7168 + offs_k[None, :]
    w_base = w13_ptr + e.to(tl.int64) * (4096 * 7168)
    b1_ptrs = w_base + offs_n.to(tl.int64)[None, :] * 7168 + offs_k[:, None]
    b2_ptrs = b1_ptrs + 2048 * 7168
    s_base = s13_ptr + e * (32 * 56)
    nb = (pid_n * BN) // 128

    acc1 = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, 7168, BK):
        a = tl.load(a_ptrs)
        b1 = tl.load(b1_ptrs)
        b2 = tl.load(b2_ptrs)
        kb = k0 // 128
        asc = tl.load(hs_scale_ptr + kb * T + tok)
        ws1 = tl.load(s_base + nb * 56 + kb)
        ws2 = tl.load(s_base + (16 + nb) * 56 + kb)
        acc1 += tl.dot(a, b1) * (asc[:, None] * ws1)
        acc2 += tl.dot(a, b2) * (asc[:, None] * ws2)
        a_ptrs += BK
        b1_ptrs += BK
        b2_ptrs += BK

    c = acc2 / (1.0 + tl.exp(-acc2)) * acc1
    amax = tl.max(tl.abs(c), axis=1)
    s = tl.maximum(amax, 1e-30) * (1.0 / 32768.0)
    c16 = (c / s[:, None]).to(tl.float16)
    c_ptrs = c_ptr + rows.to(tl.int64)[:, None] * 2048 + offs_n[None, :]
    tl.store(c_ptrs, c16)
    tl.store(cs_ptr + rows.to(tl.int64) * (2048 // BN) + pid_n, s)


@triton.jit
def _gemm2_kernel(
    c_ptr, cs_ptr, w2_ptr, s2_ptr, sorted_tok_ptr, sorted_w_ptr, tile_expert_ptr, out_ptr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(tile_expert_ptr + pid_m)
    if e >= 32:
        return
    rows = pid_m * BM + tl.arange(0, BM)
    tok = tl.load(sorted_tok_ptr + rows)
    valid = tok >= 0
    tok = tl.where(valid, tok, 0)
    wt = tl.load(sorted_w_ptr + rows)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    c_ptrs = c_ptr + rows.to(tl.int64)[:, None] * 2048 + offs_k[None, :]
    cs_ptrs = cs_ptr + rows.to(tl.int64) * (2048 // BK)
    b_ptrs = (w2_ptr + e.to(tl.int64) * (7168 * 2048)
              + offs_n.to(tl.int64)[None, :] * 2048 + offs_k[:, None])
    s_base = s2_ptr + e * (56 * 16) + ((pid_n * BN) // 128) * 16

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, 2048, BK):
        c = tl.load(c_ptrs)
        cs = tl.load(cs_ptrs)
        b = tl.load(b_ptrs).to(tl.float16)
        ws = tl.load(s_base + k0 // 128)
        acc += tl.dot(c, b) * (cs[:, None] * ws)
        c_ptrs += BK
        cs_ptrs += 1
        b_ptrs += BK

    acc = acc * wt[:, None]
    o_ptrs = out_ptr + tok.to(tl.int64)[:, None] * 7168 + offs_n[None, :]
    tl.atomic_add(o_ptrs, acc, mask=valid[:, None])


def _route(routing_logits, routing_bias, routed_scaling_factor):
    T = routing_logits.shape[0]
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_wb = s + bias
    grouped = s_wb.view(T, 8, 32)
    top2, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    gscore = top2.sum(dim=2)
    _, gidx = torch.topk(gscore, k=4, dim=1, largest=True, sorted=False)
    gmask = torch.zeros_like(gscore)
    gmask.scatter_(1, gidx, 1.0)
    smask = gmask.unsqueeze(2).expand(T, 8, 32).reshape(T, 256)
    pruned = s_wb.masked_fill(smask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    w = s.gather(1, topk_idx)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-20) * routed_scaling_factor
    return topk_idx, w


@torch.no_grad()
def kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    device = hidden_states.device
    T = routing_logits.shape[0]
    topk_idx, w = _route(routing_logits, routing_bias, float(routed_scaling_factor))

    if T <= 64:
        BM = 16
    elif T <= 512:
        BM = 32
    elif T <= 2048:
        BM = 64
    else:
        BM = 128

    N = T * TOP_K
    local = topk_idx - int(local_expert_offset)
    valid = (local >= 0) & (local < E_LOCAL)
    key = torch.where(valid, local, torch.full_like(local, E_LOCAL)).reshape(-1)
    sorted_key, order = torch.sort(key, stable=True)
    counts = torch.bincount(key, minlength=E_LOCAL + 1)[:E_LOCAL]
    padded = (counts + BM - 1) // BM * BM
    cum_pad = torch.cumsum(padded, 0)
    pad_start = cum_pad - padded
    cum_raw = torch.cumsum(counts, 0)
    raw_start = cum_raw - counts

    max_rows = N + E_LOCAL * BM
    sorted_tok = torch.full((max_rows + 1,), -1, dtype=torch.int32, device=device)
    sorted_w = torch.zeros((max_rows + 1,), dtype=torch.float32, device=device)
    vmask = sorted_key < E_LOCAL
    sk = sorted_key.clamp(max=E_LOCAL - 1)
    ar = torch.arange(N, device=device)
    dest = pad_start[sk] + ar - raw_start[sk]
    dest = torch.where(vmask, dest, torch.full_like(dest, max_rows))
    sorted_tok[dest] = (order // TOP_K).to(torch.int32)
    sorted_w[dest] = w.reshape(-1)[order]
    sorted_tok[max_rows] = -1
    sorted_w[max_rows] = 0.0

    if T > 256:
        total = int(cum_pad[-1].item())
        num_tiles = max(total // BM, 1)
    else:
        num_tiles = max_rows // BM
    tile_expert = torch.searchsorted(
        cum_pad, torch.arange(num_tiles, device=device, dtype=torch.int64) * BM, right=True
    ).to(torch.int32)

    C = torch.empty((num_tiles * BM, I_DIM), dtype=torch.float16, device=device)
    CS = torch.empty((num_tiles * BM, I_DIM // CBLK), dtype=torch.float32, device=device)
    out = torch.zeros((T, H_DIM), dtype=torch.float32, device=device)

    hs = hidden_states.contiguous()
    hs_scale = hidden_states_scale.to(torch.float32).contiguous()
    w13 = gemm1_weights.contiguous()
    s13 = gemm1_weights_scale.to(torch.float32).contiguous()
    w2 = gemm2_weights.contiguous()
    s2 = gemm2_weights_scale.to(torch.float32).contiguous()

    if BM == 128:
        k1 = dict(BN=CBLK, BK=128, num_warps=8, num_stages=2)
        k2 = dict(BN=128, BK=CBLK, num_warps=8, num_stages=3)
    else:
        k1 = dict(BN=CBLK, BK=128, num_warps=4, num_stages=3)
        k2 = dict(BN=128, BK=CBLK, num_warps=4, num_stages=3)

    grid1 = (num_tiles, I_DIM // k1["BN"])
    _gemm1_swiglu_kernel[grid1](
        hs, hs_scale, w13, s13,
        sorted_tok, tile_expert, C, CS, T,
        BM=BM, BN=k1["BN"], BK=k1["BK"],
        num_warps=k1["num_warps"], num_stages=k1["num_stages"],
    )
    grid2 = (num_tiles, H_DIM // k2["BN"])
    _gemm2_kernel[grid2](
        C, CS, w2, s2, sorted_tok, sorted_w, tile_expert, out,
        BM=BM, BN=k2["BN"], BK=k2["BK"],
        num_warps=k2["num_warps"], num_stages=k2["num_stages"],
    )
    return out.to(torch.bfloat16)
