"""Prompt text and helpers for the MoE kernel agent loop.

prompt_template.md is the human-readable version for the team. THIS file is what the loop
actually sends. If you change one, change the other.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SYSTEM_PROMPT = """You are an expert GPU kernel engineer. You write and optimize GPU kernels in Triton
(Python) for NVIDIA GPUs. You are working inside an automated loop: each time you reply,
your code is compiled, checked for correctness against a reference, and benchmarked.
You then receive the results and try again.

Rules:
- Reply with ONE complete file, `kernel.py`, inside a single ```python code block.
  After the code block, write at most 5 sentences explaining what you changed and why.
- Correctness comes first. A kernel that fails any workload scores zero, however fast it is.
- You may use `torch` and `triton`. Do NOT import or call `flashinfer`, `deep_gemm`, `vllm`,
  `tensorrt_llm`, or any other prebuilt MoE library. Do not use the network.
- The target GPU is {gpu_name} (compute capability {compute_capability}) with
  {gpu_memory_gb} GB of memory. Do not use features that this GPU does not support.
- Memory is tight: the FP8 weights alone take about 1.4 GB. Never dequantize all experts'
  weights to float32/bfloat16 at once. Keep weights in FP8 and dequantize per expert or per
  tile inside the kernel.
- Your function must be named `{entry_point}` and must have exactly the same parameters
  as the reference function `run` shown below, in the same order. It must RETURN the
  output tensor. (Do not write into a passed-in output buffer.)
- Do not special-case the benchmark inputs, hard-code outputs, or read files."""

TASK_PROMPT = """### Problem
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
{reference_code}
```

### Your starting kernel
```python
{seed_kernel}
```

### Budget
You have {max_iterations} attempts in total. This is attempt 1.

Produce your first version of `kernel.py` now."""

FEEDBACK_HEADER = """This is attempt {attempt} of {max_iterations}.

### Your previous kernel
```python
{last_kernel}
```

### Result
Overall status: {status}   (one of: FORMAT_ERROR, BUILD_ERROR, RUNTIME_ERROR, INCORRECT, CORRECT)
"""

FEEDBACK_FOOTER = """
### What to do
- If the last attempt failed, fix it. Prefer a small, targeted fix over a rewrite.
- If it was correct, make it faster. Say which bottleneck you are targeting.
- Reply in the same format: one complete `kernel.py` in a single ```python block, then at most
  5 sentences."""


def truncate_lines(text: str, max_lines: int) -> str:
    lines = (text or "").strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "[...earlier output omitted...]\n" + "\n".join(lines[-max_lines:])


def build_system(cfg: dict) -> str:
    return SYSTEM_PROMPT.format(
        gpu_name=cfg["gpu_name"],
        compute_capability=cfg["compute_capability"],
        gpu_memory_gb=cfg.get("gpu_memory_gb", "unknown"),
        entry_point=cfg["entry_point"],
    )


def build_task(reference_code: str, seed_kernel: str, max_iterations: int) -> str:
    return TASK_PROMPT.format(
        reference_code=reference_code, seed_kernel=seed_kernel, max_iterations=max_iterations
    )


def build_feedback(
    attempt: int,
    max_iterations: int,
    last_kernel: str,
    result,  # bench.BenchResult
    best_kernel: str | None,
    best_speedup: float | None,
    best_attempt: int | None,
    max_error_lines: int,
    profile_summary: str | None = None,
) -> str:
    parts = [
        FEEDBACK_HEADER.format(
            attempt=attempt,
            max_iterations=max_iterations,
            last_kernel=last_kernel,
            status=result.status,
        )
    ]
    if result.status in ("FORMAT_ERROR", "BUILD_ERROR", "RUNTIME_ERROR"):
        parts.append(
            f"Error output (last {max_error_lines} lines):\n```\n"
            f"{truncate_lines(result.error_text, max_error_lines)}\n```"
        )
    elif result.status == "INCORRECT":
        parts.append(
            f"Failing workloads: {result.n_failed} of {result.n_total}.\n"
            f"Worst case: max absolute error {result.max_abs_error}, "
            f"max relative error {result.max_rel_error}.\n"
            f"Example failing workload ids: {', '.join(result.failing_ids[:5])}"
        )
        if result.error_text:
            parts.append(
                f"Extra output:\n```\n{truncate_lines(result.error_text, max_error_lines)}\n```"
            )
    elif result.status == "CORRECT":
        rows = ["| workload | your latency (ms) | reference latency (ms) | speedup |", "|---|---|---|---|"]
        for w in result.workloads:
            rows.append(
                f"| {w['id']} | {w['latency_ms']:.3f} | {w['reference_latency_ms']:.3f} | {w['speedup']:.2f}x |"
            )
        parts.append("\n".join(rows))
        parts.append(f"Mean speedup: {result.mean_speedup:.3f}x")
        if profile_summary:
            parts.append(f"Profiler summary (Nsight Compute, summarized):\n{profile_summary}")

    if best_kernel is not None and best_kernel.strip() != last_kernel.strip():
        parts.append(
            f"### Best correct kernel so far\nMean speedup {best_speedup:.3f}x (attempt {best_attempt}).\n"
            f"```python\n{best_kernel}\n```"
        )
    elif best_kernel is not None:
        parts.append(f"(The kernel above is your best correct kernel so far: {best_speedup:.3f}x.)")
    else:
        parts.append("No correct kernel yet.")
    parts.append(FEEDBACK_FOOTER)
    return "\n\n".join(parts)


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_kernel(reply_text: str) -> str | None:
    """Return the LAST fenced code block in the reply, or None."""
    blocks = _CODE_BLOCK.findall(reply_text or "")
    if not blocks:
        return None
    return blocks[-1].strip() + "\n"


def load_reference_code(definition_json: Path) -> str:
    """Read the reference implementation out of the definition JSON.

    I have not confirmed the exact field name, so this tries 'reference' first and then
    falls back to any string value that contains 'def run('.
    """
    data = json.loads(Path(definition_json).read_text(encoding="utf-8"))
    ref = data.get("reference")
    if isinstance(ref, str) and "def run(" in ref:
        return ref.strip() + "\n"
    for value in data.values():
        if isinstance(value, str) and "def run(" in value:
            return value.strip() + "\n"
    raise ValueError(
        f"No reference code found in {definition_json}. Open the file and check the field name."
    )


def make_seed_kernel(reference_code: str, entry_point: str) -> str:
    """Seed = the reference implementation with `run` renamed to the entry point."""
    return reference_code.replace("def run(", f"def {entry_point}(", 1)
