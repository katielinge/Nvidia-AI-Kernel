"""
Knock-out ablation: WHY is Sonnet's kernel faster than the non-ML baselines?

Sonnet run 1's best kernel has the same overall structure as our hand-written baseline
(routing in PyTorch, one launch per expert, a GEMM1+SwiGLU kernel and a GEMM2+scatter kernel).
It differs in three places. This script switches each one back to the baseline's choice,
alone and together, and benchmarks every variant the same way:

  fp8     GEMM1 multiplies FP8 values on the FP8 tensor cores   -> baseline: convert to bf16 first
  gemm2   GEMM2 = two bf16 tensor-core dots (hi/lo split)        -> baseline: fp32 operands, tf32x3
  tiles   tile height chosen per expert (16/32/64, 128 if large) -> baseline: always 32

With all three switched off, the kernel matches the hand-written baseline's design and config.

From the Nvidia-AI-Kernel folder (venv active, FIB_DATASET_PATH set):
    python agent_loop/ablation_knockout.py
About 10 to 20 minutes. Safe to stop and restart: finished variants are skipped.

Writes results/ablation/<variant>.json and results/ablation.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

SOURCE = Path("agent_runs") / "sonnet_run1" / "best_kernel.py"
BENCH = ["--warmup", "3", "--iters", "20", "--trials", "3", "--ref-cache", "ref_latency.json"]

# (old text, new text) pairs; each must match exactly once
KNOCK = {
    "fp8": [("_USE_FP8 = [True]", "_USE_FP8 = [False]")],
    "gemm2": [
        ("""        c_hi = c.to(tl.bfloat16)
        c_lo = (c - c_hi.to(tl.float32)).to(tl.bfloat16)
        w = tl.load(W + offs_n[None, :].to(tl.int64) * I + k * 128 + offs_k[:, None]).to(tl.bfloat16)""",
         """        w = tl.load(W + offs_n[None, :].to(tl.int64) * I + k * 128 + offs_k[:, None]).to(tl.float32)"""),
        ("""        d = tl.dot(c_hi, w) + tl.dot(c_lo, w)""",
         """        d = tl.dot(c, w, input_precision="tf32x3")"""),
    ],
    "tiles": [
        ("""        BM = 16 if Tk <= 16 else (32 if Tk <= 32 else 64)""", """        BM = 32"""),
        ("_BIG_OK = [True]", "_BIG_OK = [False]"),
        ("_BIG2_OK = [True]", "_BIG2_OK = [False]"),
    ],
}
VARIANTS = [
    ("original", []),
    ("no_fp8", ["fp8"]),
    ("no_gemm2_split", ["gemm2"]),
    ("no_adaptive_tiles", ["tiles"]),
    ("no_fp8_no_gemm2", ["fp8", "gemm2"]),
    ("all_three_off", ["fp8", "gemm2", "tiles"]),
]
DESCRIBE = {
    "original": "Sonnet run 1 best kernel, unchanged",
    "no_fp8": "GEMM1 converts FP8 to bf16 before the dot",
    "no_gemm2_split": "GEMM2 uses fp32 operands with tf32x3 (baseline's choice)",
    "no_adaptive_tiles": "Tile height fixed at 32 (baseline's choice)",
    "no_fp8_no_gemm2": "Both GEMM changes off; adaptive tiles kept",
    "all_three_off": "All three off: the baseline's design and config",
}


def make_variant(src: str, knocks: list[str]) -> str:
    for k in knocks:
        for old, new in KNOCK[k]:
            n = src.count(old)
            if n != 1:
                raise SystemExit(f"Knock-out '{k}': expected 1 match, found {n}: {old[:60]!r}")
            src = src.replace(old, new)
    return src


def bench(code: str, repo: Path, dataset: str) -> dict:
    (repo / "solution" / "triton" / "kernel.py").write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        env = dict(os.environ, FIB_DATASET_PATH=dataset, PYTHONUTF8="1")
        proc = subprocess.run([sys.executable, str(HERE / "bench_worker.py"), "--out", str(out)] + BENCH,
                              cwd=repo, env=env, capture_output=True, text=True, timeout=2 * 3600)
        if proc.returncode != 0 or not out.exists():
            return {"_error": ((proc.stdout or "") + (proc.stderr or ""))[-4000:]}
        res = json.loads(out.read_text(encoding="utf-8"))
        return next(iter(res.values())) if res else {"_error": "no results"}


def main():
    with open(HERE / "run_config.toml", "rb") as f:
        run = tomllib.load(f)["run"]
    repo = Path(run["repo_root"]).resolve()
    dataset = os.environ.get("FIB_DATASET_PATH") or str(Path(run["dataset_path"]).resolve())
    src = SOURCE.read_text(encoding="utf-8")
    out_dir = Path("results") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = {name: make_variant(src, ks) for name, ks in VARIANTS}   # check all patches before running
    try:
        for name, _ in VARIANTS:
            dest = out_dir / f"{name}.json"
            if dest.exists():
                print(f"== {name}: already done, skipping")
                continue
            (out_dir / f"{name}_kernel.py").write_text(codes[name], encoding="utf-8")
            print(f"== {name}: {DESCRIBE[name]}", flush=True)
            t0 = time.time()
            res = bench(codes[name], repo, dataset)
            if "_error" in res:
                print("   benchmark failed:\n" + res["_error"])
                sys.exit(1)
            dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
            ok = [e for e in res.values() if e.get("status") == "PASSED"]
            print(f"   passed {len(ok)}/{len(res)}" + (
                f", mean speedup {sum(e['speedup_factor'] for e in ok) / len(ok):.2f}x" if len(ok) == len(res) else "")
                + f"  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ref = prompts.load_reference_code(Path(run["definition_json"]))
        (repo / "solution" / "triton" / "kernel.py").write_text(
            prompts.make_seed_kernel(ref, run["entry_point"]), encoding="utf-8")
    report(out_dir)


def report(out_dir: Path):
    rows = []
    for name, ks in VARIANTS:
        f = out_dir / f"{name}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        ok = [e for e in res.values() if e.get("status") == "PASSED"]
        sp = sum(e["speedup_factor"] for e in ok) / len(ok) if len(ok) == len(res) else None
        rows.append((name, ks, len(ok), len(res), sp))
    base = next((r[4] for r in rows if r[0] == "original"), None)
    L = ["# Knock-out ablation of Sonnet run 1's kernel", "",
         "Each variant switches one or more of Sonnet's design choices back to the hand-written baseline's "
         "choice. Same benchmark for every row: 19 contest workloads, 3 trials x 20 iterations, cached "
         "reference timings, same correctness rule.", "",
         "| Variant | What changed | Passed | Mean speedup | vs. original |", "|---|---|---|---|---|"]
    for name, ks, n_ok, n, sp in rows:
        rel = f"{sp / base:.2f}" if (sp and base) else "-"
        L.append(f"| {name} | {DESCRIBE[name]} | {n_ok}/{n} | {f'{sp:.2f}x' if sp else 'failed'} | {rel} |")
    (Path("results") / "ablation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(Path("results") / "ablation")
    else:
        main()
