"""Agent loop: ask a Claude model to write/optimize the MoE kernel, benchmark it, feed results back.

Examples (run from the starter-kit repo root):
    python agent_loop/run_agent.py --config agent_loop/run_config.toml --tier sonnet --run-id 1
    python agent_loop/run_agent.py --config agent_loop/run_config.toml --tier sonnet --run-id 1 --fake
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import prompts  # noqa: E402
from bench import BenchResult, FakeBench, RealBench  # noqa: E402
from llm import ClaudeClient, FakeClient  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier", required=True, help="a key under [tiers] in the config file")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fake", action="store_true", help="fake LLM + fake benchmark (no cost, no GPU)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = cfg["run"]
    tier = cfg["tiers"][args.tier]

    out_dir = Path(run["output_dir"]) / f"{args.tier}_run{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    log_path.write_text("", encoding="utf-8")

    reference = prompts.load_reference_code(Path(run["definition_json"]))
    seed = prompts.make_seed_kernel(reference, run["entry_point"])
    system = prompts.build_system(run)
    task = prompts.build_task(reference, seed, run["max_iterations"])

    if args.fake:
        llm = FakeClient(
            [
                "```python\nthis is not python (\n```\nbroken on purpose",
                "```python\n# FAKE_CRASH\ndef kernel():\n    pass\n```\ncrash",
                "```python\n# FAKE_WRONG\ndef kernel():\n    pass\n```\nwrong",
                "```python\n# v1\ndef kernel():\n    return 1\n```\nfirst good one",
                "```python\n# v2 faster\ndef kernel():\n    return 2\n```\nbetter",
                "no code block at all in this reply",
                "```python\n# v3\ndef kernel():\n    return 3\n```\nmaybe better",
            ]
        )
        bench = FakeBench()
    else:
        llm = ClaudeClient(
            model=tier["model"],
            max_tokens=run["max_output_tokens"],
            temperature=run.get("temperature"),
            price_in_per_mtok=tier.get("price_in_per_mtok"),
            price_out_per_mtok=tier.get("price_out_per_mtok"),
        )
        bench = RealBench(
            repo_root=Path(run["repo_root"]),
            dataset_path=run["dataset_path"],
            definition=run["definition_name"],
            timeout_s=run.get("benchmark_timeout_s", 1800),
            bench_cfg=cfg.get("bench", {}),
        )

    max_it = run["max_iterations"]
    cost_cap = run.get("cost_cap_usd")  # stop the run if total cost reaches this
    patience = run.get("no_improve_patience")  # stop after this many correct-but-no-better attempts

    total_in = total_out = 0
    total_cost = 0.0
    cost_known = True
    best_kernel = None
    best_speedup = None
    best_attempt = None
    first_correct = None
    last_kernel = seed
    last_result: BenchResult | None = None
    since_improve = 0
    stop_reason = "max_iterations"

    for attempt in range(1, max_it + 1):
        if attempt == 1:
            user_msg = task
        else:
            user_msg = prompts.build_feedback(
                attempt, max_it, last_kernel, last_result, best_kernel, best_speedup, best_attempt,
                run.get("max_error_lines", 40),
            )

        t0 = time.time()
        reply = llm.generate(system, user_msg)
        llm_s = time.time() - t0
        total_in += reply.input_tokens
        total_out += reply.output_tokens
        if reply.cost_usd is None:
            cost_known = False
        else:
            total_cost += reply.cost_usd

        code = prompts.extract_kernel(reply.text)
        (out_dir / f"attempt_{attempt:02d}_reply.txt").write_text(reply.text, encoding="utf-8")
        if code is None:
            result = BenchResult(status="FORMAT_ERROR", error_text="No ```python code block found in your reply.")
            code = last_kernel
            bench_s = 0.0
        else:
            (out_dir / f"attempt_{attempt:02d}_kernel.py").write_text(code, encoding="utf-8")
            t1 = time.time()
            result = bench.run(code)
            bench_s = time.time() - t1

        improved = False
        if result.status == "CORRECT":
            if first_correct is None:
                first_correct = attempt
            if best_speedup is None or result.mean_speedup > best_speedup:
                best_kernel, best_speedup, best_attempt = code, result.mean_speedup, attempt
                improved = True
            since_improve = 0 if improved else since_improve + 1

        entry = {
            "attempt": attempt,
            "status": result.status,
            "mean_speedup": result.mean_speedup,
            "n_failed": result.n_failed,
            "n_total": result.n_total,
            "raw_statuses": result.raw_statuses,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
            "cost_usd": reply.cost_usd,
            "cumulative_cost_usd": total_cost if cost_known else None,
            "llm_seconds": round(llm_s, 2),
            "bench_seconds": round(bench_s, 2),
            "best_speedup_so_far": best_speedup,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"[{args.tier} run{args.run_id}] attempt {attempt}/{max_it}: {result.status}"
            + (f", speedup {result.mean_speedup:.2f}x" if result.mean_speedup else "")
            + (f", cost so far ${total_cost:.2f}" if cost_known else "")
        )

        last_kernel, last_result = code, result

        if cost_cap is not None and cost_known and total_cost >= cost_cap:
            stop_reason = "cost_cap"
            break
        if patience is not None and since_improve >= patience:
            stop_reason = "no_improvement"
            break

    summary = {
        "tier": args.tier,
        "model": tier.get("model"),
        "run_id": args.run_id,
        "fake": args.fake,
        "attempts_made": attempt,
        "stop_reason": stop_reason,
        "iterations_to_first_correct": first_correct,
        "best_mean_speedup": best_speedup,
        "best_attempt": best_attempt,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_cost if cost_known else None,
        "settings": {k: run.get(k) for k in ("max_iterations", "max_output_tokens", "temperature", "gpu_name")},
        "bench_settings": cfg.get("bench", {}),
        "model": tier.get("model"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_kernel:
        (out_dir / "best_kernel.py").write_text(best_kernel, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
