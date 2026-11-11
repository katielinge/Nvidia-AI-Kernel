"""Runs ONE benchmark in its own process and writes results to a JSON file.

Run from the starter-kit repo root (the folder that contains scripts/, config.toml and local_bench.py):
    python agent_loop/bench_worker.py --out results.json [--warmup 3 --iters 20 --trials 3 --ref-cache ref_latency.json]

It is a separate process so that a crashing kernel (illegal memory access, hang) cannot kill
the agent loop. It uses local_bench.py (in-process FlashInfer-Bench evaluator, works on Windows
and 8 GB GPUs) instead of scripts/run_local.py.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--matched-ratio", type=float, default=None)
    args = ap.parse_args()

    from flashinfer_bench import BenchmarkConfig, Solution
    from scripts.pack_solution import pack_solution
    from local_bench import run_benchmark

    kw = dict(warmup_runs=args.warmup, iterations=args.iters, num_trials=args.trials)
    if args.atol is not None:
        kw["atol"] = args.atol
    if args.rtol is not None:
        kw["rtol"] = args.rtol
    if args.matched_ratio is not None:
        kw["required_matched_ratio"] = args.matched_ratio
    cfg = BenchmarkConfig(**kw)

    solution_path = pack_solution()
    solution = Solution.model_validate_json(Path(solution_path).read_text())
    results = run_benchmark(solution, cfg, ref_cache=args.ref_cache,
                            stop_on_fail=args.stop_on_fail, verbose=False)

    clean = {}
    for definition, workloads in (results or {}).items():
        clean[definition] = {}
        for uuid, entry in workloads.items():
            clean[definition][uuid] = {k: v for k, v in entry.items() if k != "solution"}
    Path(args.out).write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
