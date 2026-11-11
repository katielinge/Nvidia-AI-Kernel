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
    """
    Memory-efficient version of the FlashInfer-Bench reference.

    Same math as the original, step for step. The only change: the original
    dequantized ALL 32 experts' weights to float32 up front (~17 GB peak),
    which does not fit on an 8 GB GPU. This version dequantizes one expert's
    weights at a time, inside the expert loop, and frees them before the
    next expert (~0.5 GB peak for weights).

    - FP8 block-scale dequantization: float = fp8 * scale (128x128 blocks)
    - DeepSeek-V3 no-aux routing:
        s = sigmoid(logits); s_with_bias = s + bias
        8 groups; group score = sum of top-2; keep top 4 groups
        global top-8 experts within kept groups (using s_with_bias)
        combine weights from s (without bias), normalized, times routed_scaling_factor
    - Local experts [local_expert_offset, local_expert_offset + 32):
        GEMM1 -> SwiGLU (silu(X2) * X1) -> GEMM2, weighted per-token accumulation
    """

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168"
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization of the activations (small: [T, H])
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)            # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()            # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                   # [T, H/128, 128]
        .reshape(T, H)                                         # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                              # [T, H] float32
    del A_fp32, A_scale_expanded

    # (Weight dequantization moved into the expert loop below.)

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)                  # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)          # [E_global]

    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, E]
    s_with_bias = s + bias                                     # [T, E]

    group_size = E_global // N_GROUP                           # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32]

    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]

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
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, E]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask_per_token = (topk_idx == ge).any(dim=1)       # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)

        # Dequantize ONLY this expert's weights (same formula as the original)
        # W13_e: [2I, H], scale [2I/128, H/128]
        S13_e = gemm1_weights_scale[le].to(torch.float32)
        S13_e = torch.repeat_interleave(S13_e, BLOCK, dim=0)
        S13_e = torch.repeat_interleave(S13_e, BLOCK, dim=1)   # [2I, H]
        W13_e = gemm1_weights[le].to(torch.float32) * S13_e
        del S13_e

        # W2_e: [H, I], scale [H/128, I/128]
        S2_e = gemm2_weights_scale[le].to(torch.float32)
        S2_e = torch.repeat_interleave(S2_e, BLOCK, dim=0)
        S2_e = torch.repeat_interleave(S2_e, BLOCK, dim=1)     # [H, I]
        W2_e = gemm2_weights[le].to(torch.float32) * S2_e
        del S2_e

        A_e = A.index_select(0, token_idx)                     # [Tk, H]

        G1 = A_e.matmul(W13_e.t())                             # [Tk, 2I]
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1                                       # [Tk, I]

        O = C.matmul(W2_e.t())                                 # [Tk, H]

        w_tok = weights.index_select(0, token_idx)[:, ge]      # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

        del W13_e, W2_e, A_e, G1, X1, X2, silu_X2, C, O

    return output.to(torch.bfloat16)
