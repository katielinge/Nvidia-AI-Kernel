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

    # 1) Activations: dequantize once to bf16 (halves memory traffic, enables tensor-core GEMMs).
    A_scale = hidden_states_scale.to(torch.float32).permute(1, 0)            # [T, 56]
    A = (
        hidden_states.to(torch.float32).view(T, H // BLOCK, BLOCK)
        * A_scale.unsqueeze(-1)
    ).view(T, H).to(torch.bfloat16)                                          # bf16 [T, H]

    # 2) No-aux routing (float32, unchanged)
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

    # 3) Local expert compute, one expert at a time. Weights are dequantized per expert in bf16.
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

        # GEMM1 weights: [2I, H], 128x128 block scales [32, 56]
        s13 = gemm1_weights_scale[le].to(torch.bfloat16)
        W13 = (
            gemm1_weights[le].to(torch.bfloat16).view(2 * I // BLOCK, BLOCK, H // BLOCK, BLOCK)
            * s13.view(2 * I // BLOCK, 1, H // BLOCK, 1)
        ).view(2 * I, H)

        # GEMM2 weights: [H, I], scales [56, 16]
        s2 = gemm2_weights_scale[le].to(torch.bfloat16)
        W2 = (
            gemm2_weights[le].to(torch.bfloat16).view(H // BLOCK, BLOCK, I // BLOCK, BLOCK)
            * s2.view(H // BLOCK, 1, I // BLOCK, 1)
        ).view(H, I)

        A_e = A.index_select(0, token_idx)                                   # [Tk, H] bf16

        G1 = A_e @ W13.t()                                                   # [Tk, 2I] bf16 (fp32 accumulate)
        X1 = G1[:, :I].float()
        X2 = G1[:, I:].float()
        C = ((X2 * torch.sigmoid(X2)) * X1).to(torch.bfloat16)               # SwiGLU

        O = C @ W2.t()                                                       # [Tk, H] bf16

        w_tok = weights.index_select(0, token_idx)[:, ge]                    # [Tk] fp32
        output.index_add_(0, token_idx, O.float() * w_tok.unsqueeze(1))

        del W13, W2, A_e, G1, X1, X2, C, O

    return output.to(torch.bfloat16)
