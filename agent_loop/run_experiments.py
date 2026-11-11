"""
Runs the whole experiment on this laptop and builds the results table. One command.

From the Nvidia-AI-Kernel folder (venv active, FIB_DATASET_PATH set, `claude` logged in):

  python agent_loop/run_experiments.py --smoke          quick check: Haiku, 2 attempts, 1 run
  python agent_loop/run_experiments.py                  everything: 3 tiers x 3 runs, then the
                                                        final full benchmark, then the table
  python agent_loop/run_experiments.py --table-only     just rebuild the table from saved results

It is safe to stop (Ctrl+C) and start again: finished runs and finished final benchmarks are
skipped. Runs are interleaved (run 1: haiku, sonnet, opus; run 2: ...) so that time-of-day and
laptop temperature affect every tier equally.

Steps
  1. Agent runs: agent_loop/run_agent.py for each tier and run id.
  2. Final benchmark: each run's best_kernel.py gets the full benchmark
     (3 warmup, 100 iterations, 5 trials, all 19 workloads). These are the report's numbers.
  3. Table: results/main_table.md and results/runs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
from bench import RealBench  # noqa: E402

FULL_BENCH = {"warmup_runs": 3, "iterations": 100, "num_trials": 5, "stop_on_fail": False}


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_dir(cfg, tier, run_id):
    return Path(cfg["run"]["output_dir"]) / f"{tier}_run{run_id}"


def finished(d: Path) -> bool:
    s = d / "summary.json"
    if not s.exists():
        return False
    data = json.loads(s.read_text(encoding="utf-8"))
    return data.get("stop_reason") != "usage_limit"


def write_smoke_config(cfg_path: Path) -> Path:
    text = cfg_path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("max_iterations"):
            line = "max_iterations = 2"
        elif line.strip().startswith("output_dir"):
            line = 'output_dir = "agent_runs_smoke"'
        lines.append(line)
    out = cfg_path.with_name("run_config_smoke.toml")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def agent_runs(cfg_path, cfg, tiers, n_runs, fake=False):
    for run_id in range(1, n_runs + 1):
        for tier in tiers:
            d = run_dir(cfg, tier, run_id)
            if finished(d):
                print(f"== {tier} run{run_id}: already finished, skipping")
                continue
            print(f"\n== {tier} run{run_id}: starting", flush=True)
            rc = subprocess.call([sys.executable, str(HERE / "run_agent.py"), "--config", str(cfg_path),
                                  "--tier", tier, "--run-id", str(run_id)] + (["--fake"] if fake else []))
            if rc != 0:
                print(f"!! {tier} run{run_id} exited with code {rc}. Fix the problem and run this script again.")
                sys.exit(rc)
            summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            if summ.get("stop_reason") == "usage_limit":
                print("!! Stopped because of the subscription usage limit. Run this script again later; "
                      "it continues where it stopped.")
                sys.exit(2)


def final_benchmarks(cfg, tiers, n_runs):
    run = cfg["run"]
    full = dict(cfg.get("bench", {}))
    full.update(FULL_BENCH)
    bench = RealBench(Path(run["repo_root"]), run["dataset_path"], run["definition_name"],
                      timeout_s=6 * 3600, bench_cfg=full)
    try:
        for run_id in range(1, n_runs + 1):
            for tier in tiers:
                d = run_dir(cfg, tier, run_id)
                best = d / "best_kernel.py"
                out = d / "final_bench.json"
                if not best.exists() or out.exists():
                    continue
                print(f"\n== final benchmark: {tier} run{run_id} (full 5 x 100, all workloads)", flush=True)
                res = bench.run(best.read_text(encoding="utf-8"))
                out.write_text(json.dumps({
                    "status": res.status, "mean_speedup": res.mean_speedup,
                    "n_total": res.n_total, "n_failed": res.n_failed,
                    "max_abs_error": res.max_abs_error, "max_rel_error": res.max_rel_error,
                    "workloads": res.workloads, "error_text": res.error_text[-5000:],
                    "settings": full,
                }, indent=2), encoding="utf-8")
                print(f"   {res.status}" + (f", mean speedup {res.mean_speedup:.3f}x" if res.mean_speedup else ""))
    finally:
        # put the seed kernel back in solution/triton/kernel.py
        ref = prompts.load_reference_code(Path(run["definition_json"]))
        seed = prompts.make_seed_kernel(ref, run["entry_point"])
        (Path(run["repo_root"]) / "solution" / "triton" / "kernel.py").write_text(seed, encoding="utf-8")


def fmt(x, nd=2, suffix=""):
    return "-" if x is None else f"{x:.{nd}f}{suffix}"


def build_table(cfg, tiers, n_runs):
    rows = []
    for tier in tiers:
        for run_id in range(1, n_runs + 1):
            d = run_dir(cfg, tier, run_id)
            if not (d / "summary.json").exists():
                continue
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            f = json.loads((d / "final_bench.json").read_text(encoding="utf-8")) if (d / "final_bench.json").exists() else {}
            final_sp = f.get("mean_speedup") if f.get("status") == "CORRECT" else None
            cost = s.get("total_cost_usd")
            rows.append({
                "tier": tier, "run": run_id,
                "model_ids": " ".join(s.get("model_ids_used") or [s.get("model") or ""]),
                "attempts": s.get("attempts_made"),
                "first_correct_attempt": s.get("iterations_to_first_correct"),
                "best_attempt": s.get("best_attempt"),
                "loop_best_speedup": s.get("best_mean_speedup"),
                "final_status": f.get("status", "no correct kernel" if not (d / "best_kernel.py").exists() else "not run"),
                "final_speedup": final_sp,
                "input_tokens": s.get("total_input_tokens"),
                "output_tokens": s.get("total_output_tokens"),
                "api_equiv_cost_usd": cost,
                "llm_minutes": round((s.get("total_llm_seconds") or 0) / 60, 1),
                "bench_minutes": round((s.get("total_bench_seconds") or 0) / 60, 1),
                "cost_per_1x_gain": (cost / (final_sp - 1)) if (cost is not None and final_sp and final_sp > 1) else None,
                "stop_reason": s.get("stop_reason"),
            })

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    if rows:
        with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md = ["# Results: Claude tiers on the FlashInfer MoE kernel (RTX 5060 Laptop GPU)", "",
          "Speedup = patched PyTorch reference latency / kernel latency, mean over all 19 workloads, "
          "full benchmark (3 warmup, 100 iterations, 5 trials). A run's speedup counts only if its best "
          "kernel passes all 19 workloads. Identical code measured 1.00-1.03x, so differences under ~3% are noise. "
          "Cost is Claude Code's API-equivalent estimate; runs used a subscription and were not billed.", "",
          "## Per tier", "",
          "| Tier | Runs | Runs with a correct kernel | Final speedup, median (min-max) | First correct attempt, median | API-equiv. cost per run, median | Model time per run, median (min) |",
          "|---|---|---|---|---|---|---|"]
    for tier in tiers:
        tr = [r for r in rows if r["tier"] == tier]
        if not tr:
            continue
        sp = [r["final_speedup"] for r in tr if r["final_speedup"] is not None]
        fc = [r["first_correct_attempt"] for r in tr if r["first_correct_attempt"] is not None]
        cs = [r["api_equiv_cost_usd"] for r in tr if r["api_equiv_cost_usd"] is not None]
        lm = [r["llm_minutes"] for r in tr]
        sp_txt = f"{statistics.median(sp):.2f}x ({min(sp):.2f}-{max(sp):.2f})" if sp else "-"
        md.append(f"| {tier} | {len(tr)} | {len(sp)} | {sp_txt} | "
                  f"{fmt(statistics.median(fc), 0) if fc else '-'} | "
                  f"{('$' + fmt(statistics.median(cs))) if cs else '-'} | {fmt(statistics.median(lm), 1)} |")
    md += ["", "## Per run", "",
           "| Tier | Run | Model | Attempts | First correct | Loop best | Final speedup | Final status | In/out tokens | API-equiv. $ | $ per 1.0x gain | Model min | Bench min |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tier']} | {r['run']} | {r['model_ids']} | {r['attempts']} | {r['first_correct_attempt'] or '-'} | "
                  f"{fmt(r['loop_best_speedup'], 2, 'x')} | {fmt(r['final_speedup'], 3, 'x')} | {r['final_status']} | "
                  f"{r['input_tokens']}/{r['output_tokens']} | {fmt(r['api_equiv_cost_usd'])} | {fmt(r['cost_per_1x_gain'])} | "
                  f"{r['llm_minutes']} | {r['bench_minutes']} |")
    (out_dir / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {out_dir / 'main_table.md'} and {out_dir / 'runs.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "run_config.toml"))
    ap.add_argument("--tiers", default=None, help="comma list, default: all tiers in the config")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="Haiku only, 2 attempts, 1 run, separate folder")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--fake", action="store_true", help="test the pipeline with a fake model and benchmark")
    a = ap.parse_args()

    if not os.environ.get("FIB_DATASET_PATH"):
        os.environ["FIB_DATASET_PATH"] = str(Path("../mlsys26-contest").resolve())

    cfg_path = Path(a.config)
    if a.smoke:
        cfg_path = write_smoke_config(cfg_path)
        a.runs = 1
        a.tiers = a.tiers or "haiku"
    cfg = load(cfg_path)
    tiers = a.tiers.split(",") if a.tiers else list(cfg["tiers"].keys())

    if not a.table_only and not a.skip_agent:
        agent_runs(cfg_path, cfg, tiers, a.runs, fake=a.fake)
    if not a.table_only and not a.skip_final and not a.smoke and not a.fake:
        final_benchmarks(cfg, tiers, a.runs)
    if not a.smoke:
        build_table(cfg, tiers, a.runs)
    else:
        d = run_dir(cfg, tiers[0], 1)
        print("\nSmoke test summary:\n" + (d / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
