"""
In-process benchmark runner for Windows / small GPUs.

WHY THIS EXISTS
  The official FlashInfer-Bench runners (persistent and isolated) pass GPU
  tensors between processes with CUDA IPC, which native Windows does not
  support. They also keep every trial's inputs on the GPU at once
  (~1.5 GB per trial for this MoE), which overflows an 8 GB GPU.

WHAT IT DOES
  Calls the official evaluator code directly, in one process:
    - same input generation        (evaluator.build_baseline -> gen_inputs)
    - same correctness check       (evaluator.check_correctness; for this MoE
                                    that is LowBitEvaluator)
    - same timing                  (evaluator.eval_performance -> time_runnable)
    - same speedup                 (mean reference latency / mean solution latency)
  The only differences: no worker process, and trials run one at a time
  (inputs for trial k are freed before trial k+1 is generated).

USAGE (from the Nvidia-AI-Kernel folder, venv active, FIB_DATASET_PATH set)
  python local_bench.py                    full run: all 19 workloads, 3 warmup / 100 iters / 5 trials
  python local_bench.py --quick            1 warmup / 1 iter / 1 trial (fast smoke test)
  python local_bench.py --workloads 2      only the first 2 workloads
  python local_bench.py --ref-cache ref_latency.json
        Times the reference once and saves it; later runs reuse the saved
        reference latency instead of re-timing the slow reference every time.
        (Reference outputs for correctness are still recomputed every run.)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Correctness, EvaluationStatus, Performance

DEVICE = "cuda:0"


def _with_log(evaluation, captured: str):
    """Attach captured error text to an Evaluation's log."""
    if not captured:
        return evaluation
    log = (captured + "\n" + (evaluation.log or "")).strip()
    try:
        return evaluation.model_copy(update={"log": log})
    except Exception:
        try:
            evaluation.log = log
        except Exception:
            pass
        return evaluation


def _fail(status, msg: str):
    return make_eval(status=status, device=DEVICE, log_path=None, extra_msg=msg)


def evaluate_workload(definition, workload, sol_runnable, cfg, root, cached_ref_ms=None):
    """Run all trials for one workload, one trial at a time. Returns (Evaluation, ref_latencies)."""
    evaluator = resolve_evaluator(definition)
    cfg_one = replace(cfg, num_trials=1, profile_baseline=cached_ref_ms is None)

    sol_lat, ref_lat = [], []
    max_abs = max_rel = 0.0
    min_matched = 1.0

    for _ in range(cfg.num_trials):
        captured = io.StringIO()
        bl = None
        try:
            with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
                bl = evaluator.build_baseline(
                    definition=definition, workload=workload, cfg=cfg_one,
                    device=DEVICE, trace_set_root=root,
                )
                ref_ms = bl.mean_latency_ms if cached_ref_ms is None else cached_ref_ms

                correctness, failed = evaluator.check_correctness(
                    definition=definition, sol_runnable=sol_runnable,
                    inputs=bl.inputs, ref_outputs=bl.outputs,
                    cfg=cfg, log_path=None, device=DEVICE,
                )
                if failed is not None:
                    return _with_log(failed, captured.getvalue()), ref_lat

                perf, failed = evaluator.eval_performance(
                    definition=definition, sol_runnable=sol_runnable,
                    inputs=bl.inputs, ref_mean_latency_ms=ref_ms,
                    cfg=cfg, log_path=None, device=DEVICE,
                )
                if failed is not None:
                    return _with_log(failed, captured.getvalue()), ref_lat
        except Exception:
            msg = captured.getvalue() + "\n" + traceback.format_exc()
            return _fail(EvaluationStatus.RUNTIME_ERROR, msg), ref_lat
        finally:
            del bl
            torch.cuda.empty_cache()

        sol_lat.append(perf.latency_ms)
        ref_lat.append(ref_ms)
        max_abs = max(max_abs, correctness.max_absolute_error)
        max_rel = max(max_rel, correctness.max_relative_error)
        extra = getattr(correctness, "extra", None) or {}
        min_matched = min(min_matched, extra.get("matched_ratio", 1.0))

    sol_mean = sum(sol_lat) / len(sol_lat)
    ref_mean = sum(ref_lat) / len(ref_lat)
    ev = make_eval(
        status=EvaluationStatus.PASSED, device=DEVICE, log_path=None,
        correctness=Correctness(max_relative_error=max_rel, max_absolute_error=max_abs,
                                extra={"matched_ratio": min_matched}),
        performance=Performance(latency_ms=sol_mean, reference_latency_ms=ref_mean,
                                speedup_factor=ref_mean / sol_mean),
    )
    return ev, ref_lat


def run_benchmark(solution: Solution, config: BenchmarkConfig = None, *,
                  max_workloads: int | None = None, ref_cache: str | None = None,
                  stop_on_fail: bool = False, verbose: bool = True) -> dict:
    """Same return format as scripts/run_local.py run_benchmark()."""
    import os
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    root_path = os.environ.get("FIB_DATASET_PATH")
    if not root_path:
        raise EnvironmentError("FIB_DATASET_PATH is not set. Run: $env:FIB_DATASET_PATH=\"C:\\Users\\Katie\\mlsys26-contest\"")
    ts = TraceSet.from_path(root_path)
    if solution.definition not in ts.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
    definition = ts.definitions[solution.definition]
    workloads = ts.workloads.get(solution.definition, [])
    if max_workloads:
        workloads = workloads[:max_workloads]

    cache = {}
    cache_path = Path(ref_cache) if ref_cache else None
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    results = {definition.name: {}}

    # Build (compile) the solution once
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            sol_runnable = BuilderRegistry.get_instance().build(definition, solution)
    except Exception:
        msg = captured.getvalue() + "\n" + traceback.format_exc()
        for wl in workloads:
            results[definition.name][wl.workload.uuid] = {
                "status": EvaluationStatus.COMPILE_ERROR.value, "solution": solution.name, "log": msg}
        if verbose:
            print("COMPILE_ERROR:\n" + msg)
        return results

    for i, wl_trace in enumerate(workloads, 1):
        wl = wl_trace.workload
        cached = cache.get(wl.uuid)
        cached_ms = (sum(cached) / len(cached)) if cached else None
        t0 = time.time()
        ev, ref_lat = evaluate_workload(definition, wl, sol_runnable, config, ts.root, cached_ms)
        if cache_path is not None and cached is None and ev.status == EvaluationStatus.PASSED:
            cache[wl.uuid] = ref_lat

        entry = {"status": ev.status.value, "solution": solution.name}
        if ev.performance:
            entry["latency_ms"] = ev.performance.latency_ms
            entry["reference_latency_ms"] = ev.performance.reference_latency_ms
            entry["speedup_factor"] = ev.performance.speedup_factor
        if ev.correctness:
            entry["max_abs_error"] = ev.correctness.max_absolute_error
            entry["max_rel_error"] = ev.correctness.max_relative_error
            extra = getattr(ev.correctness, "extra", None) or {}
            if "matched_ratio" in extra:
                entry["matched_ratio"] = extra["matched_ratio"]
        if ev.status != EvaluationStatus.PASSED:
            entry["log"] = ev.log
        results[definition.name][wl.uuid] = entry

        if verbose:
            line = f"[{i}/{len(workloads)}] {wl.uuid[:8]} {ev.status.value}"
            if ev.performance:
                line += (f" | {ev.performance.latency_ms:.3f} ms (ref {ev.performance.reference_latency_ms:.3f} ms)"
                         f" | {ev.performance.speedup_factor:.2f}x")
            line += f" | {time.time() - t0:.0f}s"
            print(line, flush=True)
            if ev.status != EvaluationStatus.PASSED:
                print("    " + (ev.log or "").strip().replace("\n", "\n    ")[-3000:], flush=True)

        if stop_on_fail and ev.status != EvaluationStatus.PASSED:
            break

    if cache_path is not None:
        cache_path.write_text(json.dumps(cache, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="1 warmup, 1 iteration, 1 trial")
    p.add_argument("--workloads", type=int, default=None, help="only the first N workloads")
    p.add_argument("--ref-cache", default=None, help="JSON file to save/reuse reference latencies")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--atol", type=float, default=None, help="default: tool default (0.01)")
    p.add_argument("--rtol", type=float, default=None, help="default: tool default (0.01)")
    p.add_argument("--matched-ratio", type=float, default=None,
                   help="fraction of elements that must be within tolerance (tool default for this MoE: 0.95)")
    p.add_argument("--out", default=None, help="save results JSON here")
    a = p.parse_args()

    kw = dict(warmup_runs=1, iterations=1, num_trials=1) if a.quick else \
         dict(warmup_runs=3, iterations=100, num_trials=5)
    if a.atol is not None: kw["atol"] = a.atol
    if a.rtol is not None: kw["rtol"] = a.rtol
    if a.matched_ratio is not None: kw["required_matched_ratio"] = a.matched_ratio
    cfg = BenchmarkConfig(**kw)

    from scripts.pack_solution import pack_solution
    solution = Solution.model_validate_json(pack_solution().read_text())
    print(f"Solution: {solution.name}  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: warmup={cfg.warmup_runs} iters={cfg.iterations} trials={cfg.num_trials} "
          f"atol={cfg.atol} rtol={cfg.rtol} matched_ratio={cfg.required_matched_ratio or '0.95 (MoE default)'}")

    t0 = time.time()
    results = run_benchmark(solution, cfg, max_workloads=a.workloads,
                            ref_cache=a.ref_cache, stop_on_fail=a.stop_on_fail)
    rows = list(results.values())[0].values()
    passed = [r for r in rows if r["status"] == "PASSED"]
    print(f"\nPassed {len(passed)}/{len(rows)} workloads in {time.time() - t0:.0f}s")
    if passed and len(passed) == len(rows):
        mean_speedup = sum(r["speedup_factor"] for r in passed) / len(passed)
        print(f"Mean speedup: {mean_speedup:.3f}x")

    out = Path(a.out) if a.out else PROJECT_ROOT / "results" / f"{solution.name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print("Results saved to", out)


if __name__ == "__main__":
    main()
