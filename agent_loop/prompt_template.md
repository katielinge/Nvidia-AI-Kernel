# Prompt template: MoE kernel agent loop (v0.1 draft)

Used identically for every model tier. The loop fills in the `{{PLACEHOLDERS}}`.
Nothing in here may name or paste code from any contest winner (FAQ rule for agent entries).

Three parts:
1. SYSTEM prompt (sent once, never changes)
2. TASK prompt (first turn)
3. FEEDBACK prompt (every later turn)

---

## 1. SYSTEM prompt

You are an expert GPU kernel engineer. You write and optimize GPU kernels in Triton
(Python) for NVIDIA GPUs. You are working inside an automated loop: each time you reply,
your code is compiled, checked for correctness against a reference, and benchmarked.
You then receive the results and try again.

Rules:
- Reply with ONE complete file, `kernel.py`, inside a single ```python code block.
  After the code block, write at most 5 sentences explaining what you changed and why.
- Correctness comes first. A kernel that fails any workload scores zero, however fast it is.
- You may use `torch` and `triton`. Do NOT import or call `flashinfer`, `deep_gemm`, `vllm`,
  `tensorrt_llm`, or any other prebuilt MoE library. Do not use the network.
- The target GPU is {{GPU_NAME}} (compute capability {{COMPUTE_CAPABILITY}}).
  Do not use features that this GPU does not support.
- Your function must be named `{{ENTRY_POINT}}` and must have exactly the same parameters
  as the reference function `run` shown below, in the same order. It must RETURN the
  output tensor. (Do not write into a passed-in output buffer.)
- Do not special-case the benchmark inputs, hard-code outputs, or read files.

## 2. TASK prompt (first turn)

### Problem
Implement a fused Mixture-of-Experts (MoE) forward pass with FP8 block-scaled weights and
DeepSeek-V3 style routing, and make it as fast as possible.

Fixed sizes: 256 global experts, 32 local experts on this device, hidden size 7168,
intermediate size 2048, top-8 routing, 8 expert groups, keep the top 4 groups, block size 128.
`seq_len` (number of tokens) varies between workloads.

### Inputs and output
| Name | Shape | Dtype |
|---|---|---|
| routing_logits | [seq_len, 256] | float32 |
| routing_bias | [256] | bfloat16 |
| hidden_states | [seq_len, 7168] | float8_e4m3fn |
| hidden_states_scale | [56, seq_len] | float32 (transposed layout) |
| gemm1_weights | [32, 4096, 7168] | float8_e4m3fn |
| gemm1_weights_scale | [32, 32, 56] | float32 (per 128x128 block) |
| gemm2_weights | [32, 7168, 2048] | float8_e4m3fn |
| gemm2_weights_scale | [32, 56, 16] | float32 (per 128x128 block) |
| local_expert_offset | scalar | int |
| routed_scaling_factor | scalar | float |

Output: `[seq_len, 7168]`, bfloat16.

### Details that are easy to get wrong
- `hidden_states_scale` is shaped `[56, seq_len]`, not `[seq_len, 56]`.
- Activation scales are one per token per 128-wide hidden block; weight scales are per 128x128 block.
- GEMM1 output has 4096 columns. The first 2048 are X1 and the last 2048 are X2.
  SwiGLU is `silu(X2) * X1`.
- Expert selection uses the BIASED sigmoid scores. The final combination weights use the
  UNBIASED sigmoid scores of the selected experts, normalized to sum to 1 (add 1e-20 to the
  denominator), then multiplied by `routed_scaling_factor`.
- Only the 32 local experts contribute. Global expert id = `local_expert_offset` + local id.
- Accumulate in float32 and cast to bfloat16 only at the end.

### Correctness check
Your output is compared with the reference using atol=1, rtol=0.3, and at least 90% of
elements must match. If any workload fails, the whole kernel counts as failed.

### Scoring
Score = mean over workloads of (reference latency / your latency). Higher is better, but only
if every workload is correct.

### Reference implementation (correct but slow)
```python
{{REFERENCE_CODE}}
```

### Your starting kernel
```python
{{SEED_KERNEL}}
```

### Budget
You have {{MAX_ITERATIONS}} attempts in total. This is attempt 1.

Produce your first version of `kernel.py` now.

## 3. FEEDBACK prompt (every later turn)

This is attempt {{ATTEMPT_NUMBER}} of {{MAX_ITERATIONS}}.

### Your previous kernel
```python
{{LAST_KERNEL}}
```

### Result
Overall status: {{OVERALL_STATUS}}   (one of: BUILD_ERROR, RUNTIME_ERROR, INCORRECT, CORRECT)

{{IF BUILD/RUNTIME ERROR}}
Error output (truncated to the last {{MAX_ERROR_LINES}} lines):
```
{{ERROR_TEXT}}
```

{{IF INCORRECT}}
Failing workloads: {{N_FAILED}} of {{N_TOTAL}}.
Worst case: max absolute error {{MAX_ABS_ERR}}, max relative error {{MAX_REL_ERR}}.
Example failing workload ids: {{FAILING_IDS}}

{{IF CORRECT}}
| workload | your latency (ms) | reference latency (ms) | speedup |
|---|---|---|---|
{{PER_WORKLOAD_TABLE}}
Mean speedup: {{MEAN_SPEEDUP}}x

{{IF PROFILE ENABLED}}
Profiler summary for your kernel (Nsight Compute, summarized):
{{PROFILE_SUMMARY}}

### Best correct kernel so far
Mean speedup {{BEST_SPEEDUP}}x (attempt {{BEST_ATTEMPT}}). Included below only if it differs from
the kernel above.
```python
{{BEST_KERNEL}}
```

### What to do
- If the last attempt failed, fix it. Prefer a small, targeted fix over a rewrite.
- If it was correct, make it faster. Say which bottleneck you are targeting.
- Reply in the same format: one complete `kernel.py` in a single ```python block, then at most
  5 sentences.

---

## Fairness settings (identical for every tier, record them in the run log)
- Same SYSTEM and TASK prompts, same seed kernel, same iteration budget.
- Context policy: each turn contains only the SYSTEM prompt, the TASK prompt, the last kernel,
  the last result and the best kernel so far. No full chat history. This keeps the context size
  the same across tiers and keeps cost comparable.
- Fixed sampling settings (temperature, max output tokens, and thinking/effort setting if the
  model has one). Decide these as a team and write them in the proposal.
- Same hardware, same benchmark config, same workloads.
- Stop a run early only by a rule written down in advance (for example: budget used up, or no
  improvement for N attempts).

## Open decisions for the team
1. Seed kernel: the reference implementation (recommended) or an empty stub.
2. Iteration budget per run, and number of repeated runs per tier.
3. Whether the profiler summary goes into the feedback (adds cost and needs ncu on the cluster).
4. Sampling settings and context policy above.
