"""
Checks that the patched reference gives the SAME output as the original.

The original needs ~17 GB at 32 experts, so we can't run it on an 8 GB GPU.
Both functions handle each expert independently, so we compare them with
4 experts instead of 32 (same code, fewer loop iterations), on several token
counts and expert offsets. Then we run the patched version at the full
32 experts to check it fits in memory.

Run from the Nvidia-AI-Kernel folder:  python verify_patch.py
"""
import pathlib
import torch

HERE = pathlib.Path(__file__).parent
DEV = "cuda"
FP8 = torch.float8_e4m3fn
H, I, BLOCK, E_GLOBAL = 7168, 2048, 128, 256
ASSERT_LINE = 'assert E_local == 32, "num_local_experts must be 32"'


# ---- Original reference, copied verbatim (only the 32-expert assert removed) ----
@torch.no_grad()
def original_run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
                 gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                 local_expert_offset, routed_scaling_factor):
    H = 7168; I = 2048
    E_local = gemm1_weights.shape[0]
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]
    TOP_K = 8; N_GROUP = 8; TOPK_GROUP = 4
    device = hidden_states.device
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)
    A_scale_TH = A_scale.permute(1, 0).contiguous()
    A_scale_expanded = A_scale_TH.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H).contiguous()
    A = A_fp32 * A_scale_expanded
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)
    W13 = W13_fp32 * S13_expanded
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)
    W2 = W2_fp32 * S2_expanded
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias
    group_size = E_global // N_GROUP
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
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue
        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)
        W13_e = W13[le]; W2_e = W2[le]
        G1 = A_e.matmul(W13_e.t())
        X1 = G1[:, :I]; X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))
    return output.to(torch.bfloat16)


def load_patched(allow_any_expert_count):
    src = (HERE / "patched_reference.py").read_text(encoding="utf-8")
    if allow_any_expert_count:
        assert ASSERT_LINE in src
        src = src.replace(ASSERT_LINE, "pass")
    ns = {}
    exec(compile(src, "patched_reference.py", "exec"), ns)
    return ns["run"]


def make_inputs(T, E_local, offset, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def fp8(shape):
        return (torch.randn(shape, generator=g, device=DEV) * 2).to(FP8)

    def scale(shape):
        return torch.rand(shape, generator=g, device=DEV) * 0.02 + 0.001

    # Build the big weight tensors one expert at a time to keep memory low
    w1 = torch.empty((E_local, 2 * I, H), dtype=FP8, device=DEV)
    w2 = torch.empty((E_local, H, I), dtype=FP8, device=DEV)
    for e in range(E_local):
        w1[e] = fp8((2 * I, H))
        w2[e] = fp8((H, I))
    return dict(
        routing_logits=torch.randn((T, E_GLOBAL), generator=g, device=DEV),
        routing_bias=(torch.randn((E_GLOBAL,), generator=g, device=DEV) * 0.1).to(torch.bfloat16),
        hidden_states=fp8((T, H)),
        hidden_states_scale=scale((H // BLOCK, T)),
        gemm1_weights=w1,
        gemm1_weights_scale=scale((E_local, 2 * I // BLOCK, H // BLOCK)),
        gemm2_weights=w2,
        gemm2_weights_scale=scale((E_local, H // BLOCK, I // BLOCK)),
        local_expert_offset=offset,
        routed_scaling_factor=2.5,
    )


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    patched_small = load_patched(allow_any_expert_count=True)

    # Part 1: original vs patched, 4 experts
    all_ok = True
    cases = [(1, 0), (16, 0), (128, 0), (512, 64), (1024, 128), (2048, 252)]
    for T, offset in cases:
        inp = make_inputs(T, E_local=4, offset=offset, seed=T + offset)
        a = original_run(**inp)
        b = patched_small(**inp)
        torch.cuda.synchronize()
        identical = torch.equal(a, b)
        diff = (a.float() - b.float()).abs().max().item()
        nonzero = (a != 0).any(dim=1).sum().item()
        print(f"T={T:5d} offset={offset:3d}  tokens routed here={nonzero:5d}  "
              f"bit-identical={identical}  max diff={diff:.3g}")
        all_ok &= identical or diff == 0.0
        del inp, a, b
        torch.cuda.empty_cache()

    # Part 2: patched at full size (32 experts) must fit in memory
    patched_full = load_patched(allow_any_expert_count=False)
    torch.cuda.reset_peak_memory_stats()
    inp = make_inputs(1024, E_local=32, offset=0, seed=7)
    out = patched_full(**inp)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nFull size (32 experts, T=1024): ran OK, output {tuple(out.shape)} {out.dtype}, "
          f"peak GPU memory {peak:.2f} GB")

    print("\nRESULT:", "PASS - patched reference matches the original"
          if all_ok else "FAIL - outputs differ, send this output to Claude")


if __name__ == "__main__":
    main()
