"""
Non-ML baselines for the Claude-tier experiment, run with the SAME benchmark and the SAME
budget of timed candidates as a Claude run (run.max_iterations in run_config.toml).

  handwritten  the hand-written Triton kernel (baseline_kernel.py) with one sensible default
               config, no tuning: 1 timed candidate
  random       random search over tile settings: `budget` timed candidates, best one kept
  autotune     Triton's @autotune choosing among `budget` configs at run time, per
               token-count bucket: 1 benchmarked kernel that times `budget` configs internally

Output goes in agent_runs/<method>_run<k>/ in the same format as a Claude run (log.jsonl,
attempt kernels, best_kernel.py, summary.json), so run_experiments.py handles the final
benchmark and the table the same way.

Usually started by run_experiments.py. By hand (from the Nvidia-AI-Kernel folder):
    python agent_loop/baseline_search.py --method handwritten --run-id 1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
from bench import FakeBench, RealBench  # noqa: E402

METHODS = ("handwritten", "random", "autotune")

# Search space. BLOCK_N and BLOCK_K must divide 128 (one scale per tile); BLOCK_M >= 16 for tl.dot.
SPACE = {
    "BLOCK_M": [16, 32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "BLOCK_K": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}
PRECISIONS = ["tf32x3", "tf32", "ieee"]   # GEMM2 dot precision (GEMM1 inputs are exact in bf16)
DEFAULT1 = dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3)
DEFAULT2 = dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3, PREC="tf32x3")

# A curated list in the style of common Triton GEMM autotune lists (used for `autotune`).
AUTOTUNE_POOL = [
    dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
    dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=3),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
    dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4),
    dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=4),
    dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, num_warps=2, num_stages=4),
    dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, num_warps=2, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_warps=4, num_stages=2),
    dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=2, num_stages=4),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4),
    dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, num_warps=4, num_stages=2),
]


def sample(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SPACE.items()}


def make_source(template: str, cfg1: dict, cfg2: dict, autotune: list | None) -> str:
    src = re.sub(r"^CFG1 = .*$", f"CFG1 = {cfg1!r}", template, count=1, flags=re.M)
    src = re.sub(r"^CFG2 = .*$", f"CFG2 = {cfg2!r}", src, count=1, flags=re.M)
    src = re.sub(r"^AUTOTUNE = .*$", f"AUTOTUNE = {autotune!r}", src, count=1, flags=re.M)
    return src


def candidates(method: str, budget: int, run_id: int):
    """Yield (description, cfg1, cfg2, autotune_list)."""
    if method == "handwritten":
        yield "default config", DEFAULT1, DEFAULT2, None
    elif method == "random":
        rng = random.Random(1000 + run_id)
        seen = set()
        while len(seen) < budget:
            c1, c2 = sample(rng), sample(rng)
            c2["PREC"] = rng.choice(PRECISIONS)
            key = json.dumps([c1, c2], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            yield f"random #{len(seen)}", c1, c2, None
    elif method == "autotune":
        yield f"autotune over {min(budget, len(AUTOTUNE_POOL))} configs", DEFAULT1, DEFAULT2, \
            AUTOTUNE_POOL[:budget]
    else:
        raise SystemExit(f"unknown method {method}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "run_config.toml"))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--budget", type=int, default=None, help="default: run.max_iterations")
    ap.add_argument("--fake", action="store_true", help="fake benchmark, for testing")
    a = ap.parse_args()

    with open(a.config, "rb") as f:
        cfg = tomllib.load(f)
    run = cfg["run"]
    budget = a.budget or run["max_iterations"]
    out_dir = Path(run["output_dir"]) / f"{a.method}_run{a.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    log_path.write_text("", encoding="utf-8")

    template = (HERE / "baseline_kernel.py").read_text(encoding="utf-8")
    bench = FakeBench() if a.fake else RealBench(
        Path(run["repo_root"]), run["dataset_path"], run["definition_name"],
        timeout_s=run.get("benchmark_timeout_s", 1800), bench_cfg=cfg.get("bench", {}))

    best_code = best_sp = best_attempt = first_correct = None
    total_bench = 0.0
    n = 0
    try:
        for desc, c1, c2, at in candidates(a.method, budget, a.run_id):
            n += 1
            code = make_source(template, c1, c2, at)
            (out_dir / f"attempt_{n:02d}_kernel.py").write_text(code, encoding="utf-8")
            t0 = time.time()
            res = bench.run(code)
            bs = time.time() - t0
            total_bench += bs
            if res.status == "CORRECT":
                first_correct = first_correct or n
                if best_sp is None or res.mean_speedup > best_sp:
                    best_code, best_sp, best_attempt = code, res.mean_speedup, n
            entry = {"attempt": n, "candidate": desc, "cfg1": c1, "cfg2": c2, "autotune_configs": len(at or []),
                     "status": res.status, "mean_speedup": res.mean_speedup,
                     "min_matched_ratio": res.min_matched_ratio, "max_abs_error": res.max_abs_error,
                     "raw_statuses": res.raw_statuses, "bench_seconds": round(bs, 2),
                     "best_speedup_so_far": best_sp,
                     "error_tail": (res.error_text or "")[-1500:] if res.status != "CORRECT" else ""}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[{a.method} run{a.run_id}] candidate {n}: {desc}: {res.status}"
                  + (f", speedup {res.mean_speedup:.2f}x" if res.mean_speedup else ""), flush=True)
    finally:
        # put the seed kernel back
        if not a.fake:
            ref = prompts.load_reference_code(Path(run["definition_json"]))
            seed = prompts.make_seed_kernel(ref, run["entry_point"])
            (Path(run["repo_root"]) / "solution" / "triton" / "kernel.py").write_text(seed, encoding="utf-8")

    summary = {
        "tier": a.method, "model": "none (non-ML baseline)", "run_id": str(a.run_id), "fake": a.fake,
        "attempts_made": n, "stop_reason": "budget", "budget": budget,
        "timed_candidates": len(AUTOTUNE_POOL[:budget]) if a.method == "autotune" else n,
        "iterations_to_first_correct": first_correct, "best_mean_speedup": best_sp, "best_attempt": best_attempt,
        "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0,
        "model_ids_used": [], "total_llm_seconds": 0.0, "total_bench_seconds": round(total_bench, 1),
        "bench_settings": cfg.get("bench", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_code:
        (out_dir / "best_kernel.py").write_text(best_code, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
