import torch


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
    H = 7168
    I = 2048
    BLOCK = 128
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    device = hidden_states.device

    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    group_size = E_global // N_GROUP

    # 1) Activations: dequantize once, store as bf16 (exact fp8 values times per-block scale).
    # hidden_states_scale is [H/128, T]; permute to [T, H/128].
    A_scale = hidden_states_scale.to(torch.float32).permute(1, 0)            # [T, 56]
    A = (
        hidden_states.to(torch.float32).view(T, H // BLOCK, BLOCK)
        * A_scale.unsqueeze(-1)
    ).view(T, H).to(torch.bfloat16)

    # 2) No-aux routing (same math as reference, float32)
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # 3) Local expert compute, one expert at a time (weights stay FP8 until per-expert dequant)
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel = (topk_idx == ge).any(dim=1)
        if not bool(sel.any()):
            continue
        token_idx = torch.nonzero(sel, as_tuple=False).squeeze(1)

        # Dequantize GEMM1 weights for this expert: [2I, H] with 128x128 block scales
        s13 = gemm1_weights_scale[le].to(torch.float32)                      # [32, 56]
        W13 = (
            gemm1_weights[le].to(torch.float32).view(2 * I // BLOCK, BLOCK, H // BLOCK, BLOCK)
            * s13.view(2 * I // BLOCK, 1, H // BLOCK, 1)
        ).view(2 * I, H).to(torch.bfloat16)

        # Dequantize GEMM2 weights: [H, I] with scales [56, 16]
        s2 = gemm2_weights_scale[le].to(torch.float32)                       # [56, 16]
        W2 = (
            gemm2_weights[le].to(torch.float32).view(H // BLOCK, BLOCK, I // BLOCK, BLOCK)
            * s2.view(H // BLOCK, 1, I // BLOCK, 1)
        ).view(H, I).to(torch.bfloat16)

        A_e = A.index_select(0, token_idx)                                   # [Tk, H] bf16

        G1 = (A_e @ W13.t()).to(torch.float32)                               # [Tk, 2I]
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1                               # SwiGLU, fp32
        C = C.to(torch.bfloat16)

        O = (C @ W2.t()).to(torch.float32)                                   # [Tk, H]

        w_tok = weights.index_select(0, token_idx)[:, ge]                    # [Tk] fp32
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

        del W13, W2, A_e, G1, X1, X2, C, O

    return output.to(torch.bfloat16)
