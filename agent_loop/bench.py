"""Benchmark adapters: RealBench (runs the starter kit on a GPU) and FakeBench (for testing)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchResult:
    status: str  # FORMAT_ERROR | BUILD_ERROR | RUNTIME_ERROR | INCORRECT | CORRECT
    error_text: str = ""
    n_total: int = 0
    n_failed: int = 0
    failing_ids: list = field(default_factory=list)
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    mean_speedup: float | None = None
    workloads: list = field(default_factory=list)  # dicts: id, latency_ms, reference_latency_ms, speedup
    raw_statuses: dict = field(default_factory=dict)  # status string -> count, for debugging
    min_matched_ratio: float | None = None  # lowest fraction of elements within tolerance


def classify_status(raw: str) -> str:
    """Map the library's status string to one of our categories.

    ASSUMPTION: the passing status is 'PASSED', compile problems contain 'COMPILE', and runtime
    problems contain 'RUNTIME' or 'TIMEOUT'. The starter kit does not define these strings, so
    CHECK the raw_statuses counts in the first real run's log and fix this function if needed.
    """
    s = (raw or "").upper()
    if s == "PASSED":
        return "PASSED"
    if "COMPILE" in s or "BUILD" in s:
        return "BUILD_ERROR"
    if "RUNTIME" in s or "TIMEOUT" in s:
        return "RUNTIME_ERROR"
    return "INCORRECT"


def summarize(results: dict, definition: str | None = None) -> BenchResult:
    """Turn the dict returned by run_benchmark (definition -> workload id -> entry) into a BenchResult."""
    if not results:
        return BenchResult(status="RUNTIME_ERROR", error_text="Benchmark returned no results.")
    defn = definition if definition in results else next(iter(results))
    entries = results[defn]
    if not entries:
        return BenchResult(status="RUNTIME_ERROR", error_text="Benchmark returned no workloads.")

    res = BenchResult(status="CORRECT", n_total=len(entries))
    counts: dict[str, int] = {}
    kinds = set()
    for wid, e in entries.items():
        raw = str(e.get("status", ""))
        counts[raw] = counts.get(raw, 0) + 1
        kind = classify_status(raw)
        if kind == "PASSED":
            res.workloads.append(
                {
                    "id": wid[:8],
                    "latency_ms": float(e.get("latency_ms", float("nan"))),
                    "reference_latency_ms": float(e.get("reference_latency_ms", float("nan"))),
                    "speedup": float(e.get("speedup_factor", float("nan"))),
                }
            )
        else:
            kinds.add(kind)
            res.n_failed += 1
            res.failing_ids.append(wid[:8])
        mr = e.get("matched_ratio")
        if mr is not None:
            res.min_matched_ratio = mr if res.min_matched_ratio is None else min(res.min_matched_ratio, mr)
        for key, attr in (("max_abs_error", "max_abs_error"), ("max_rel_error", "max_rel_error")):
            v = e.get(key)
            if v is not None:
                cur = getattr(res, attr)
                setattr(res, attr, v if cur is None else max(cur, v))
    res.raw_statuses = counts

    # error text from the first failing workload (local_bench.py stores it under "log")
    first_log = next((str(e.get("log")) for e in entries.values()
                      if classify_status(str(e.get("status", ""))) != "PASSED" and e.get("log")), "")

    if res.n_failed:
        # any failing workload zeros the whole kernel; report the most severe kind
        for k in ("BUILD_ERROR", "RUNTIME_ERROR", "INCORRECT"):
            if k in kinds:
                res.status = k
                break
        res.error_text = f"Status counts: {counts}"
        if first_log:
            res.error_text += "\nError from first failing workload:\n" + first_log
    else:
        sp = [w["speedup"] for w in res.workloads]
        res.mean_speedup = sum(sp) / len(sp)
    return res


class RealBench:
    """Writes the kernel into the starter-kit repo and runs bench_worker.py as a subprocess."""

    def __init__(self, repo_root: Path, dataset_path: str, definition: str, timeout_s: int = 1800,
                 bench_cfg: dict | None = None):
        self.repo = Path(repo_root).resolve()
        self.dataset = str(Path(dataset_path).resolve())
        self.definition = definition
        self.timeout = timeout_s
        self.bench_cfg = bench_cfg or {}
        self.worker = Path(__file__).with_name("bench_worker.py")

    def run(self, kernel_code: str) -> BenchResult:
        try:
            compile(kernel_code, "kernel.py", "exec")
        except SyntaxError as e:
            return BenchResult(status="BUILD_ERROR", error_text=f"SyntaxError: {e}")

        target = self.repo / "solution" / "triton" / "kernel.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(kernel_code, encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.json"
            env = dict(os.environ, FIB_DATASET_PATH=self.dataset, PYTHONUTF8="1")
            try:
                proc = subprocess.run(
                    [sys.executable, str(self.worker), "--out", str(out)] + self._worker_args(),
                    cwd=self.repo,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return BenchResult(status="RUNTIME_ERROR", error_text=f"Timed out after {self.timeout}s.")
            logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0 or not out.exists():
                kind = "BUILD_ERROR" if ("Compil" in logs or "SyntaxError" in logs) else "RUNTIME_ERROR"
                return BenchResult(status=kind, error_text=logs)
            results = json.loads(out.read_text(encoding="utf-8"))
        res = summarize(results, self.definition)
        if res.status != "CORRECT" and not res.error_text.strip():
            res.error_text = logs
        elif res.status != "CORRECT":
            res.error_text += "\n" + logs
        return res


def _worker_args_from(cfg: dict) -> list:
    args = []
    for key, flag in (("warmup_runs", "--warmup"), ("iterations", "--iters"), ("num_trials", "--trials"),
                      ("ref_cache", "--ref-cache"), ("atol", "--atol"), ("rtol", "--rtol"),
                      ("matched_ratio", "--matched-ratio")):
        if cfg.get(key) is not None:
            args += [flag, str(cfg[key])]
    if cfg.get("stop_on_fail"):
        args.append("--stop-on-fail")
    return args


RealBench._worker_args = lambda self: _worker_args_from(self.bench_cfg)


class FakeBench:
    """Pretends to benchmark. Deterministic from the kernel text. For testing the loop only."""

    def run(self, kernel_code: str) -> BenchResult:
        try:
            compile(kernel_code, "kernel.py", "exec")
        except SyntaxError as e:
            return BenchResult(status="BUILD_ERROR", error_text=f"SyntaxError: {e}")
        if "FAKE_CRASH" in kernel_code:
            return BenchResult(status="RUNTIME_ERROR", error_text="CUDA error: illegal memory access\n  at line 12")
        if "FAKE_WRONG" in kernel_code:
            return BenchResult(
                status="INCORRECT", n_total=5, n_failed=2, failing_ids=["a1b2c3d4", "e5f6a7b8"],
                max_abs_error=3.5, max_rel_error=0.9,
            )
        h = int(hashlib.md5(kernel_code.encode()).hexdigest(), 16)
        base = 1.0 + (h % 400) / 100.0
        wl = [
            {"id": f"w{i:07d}", "latency_ms": 10 / (base + i * 0.1), "reference_latency_ms": 10.0,
             "speedup": base + i * 0.1}
            for i in range(5)
        ]
        return BenchResult(
            status="CORRECT", n_total=5, workloads=wl,
            mean_speedup=sum(w["speedup"] for w in wl) / 5,
        )
