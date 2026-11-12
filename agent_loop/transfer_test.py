"""
Transfer test: do the best kernels still work, and stay fast, on workloads they never saw?

Every kernel in this project (Claude's and the baselines') was developed and tuned against the
19 contest workloads. This script benchmarks each run's best_kernel.py on 26 NEW workloads:

  A. "shifted" (19): the same real routing traces, but served by a different shard of experts
     (local_expert_offset moved by 96), so different tokens reach the 32 local experts.
  B. "new sizes" (7): token counts that are not in the contest set (3, 24, 128, 256, 512, 2048,
     4096), random routing logits, a real routing bias.

Nothing is retuned or regenerated. Same correctness rule and timing code as the main benchmark
(per-attempt setting: 3 trials x 20 iterations). The reference is timed once and cached.

From the Nvidia-AI-Kernel folder (venv active, FIB_DATASET_PATH set):
    python agent_loop/transfer_test.py
Safe to stop and restart: finished kernels are skipped.

Writes  results/transfer/<run>.json        raw per-workload results for each kernel
        results/transfer.md                tables
        results/fig5_transfer.png          speedup on unseen workloads by token count
"""
from __future__ import annotations

import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

METHODS = ["haiku", "handwritten", "random", "autotune", "autotune_fp8", "sonnet", "opus"]
NEW_SIZES = [3, 24, 128, 256, 512, 2048, 4096]
NEW_OFFSETS = [0, 64, 160, 224, 32, 96, 128]
OFFSET_SHIFT = 96
BENCH = ["--warmup", "3", "--iters", "20", "--trials", "3"]


def build_dataset(cfg) -> tuple[Path, dict]:
    """Create transfer_dataset/ and return (root, {uuid8: info})."""
    run = cfg["run"]
    src_root = Path(run["dataset_path"]).resolve()
    defn = run["definition_name"]
    root = Path("transfer_dataset").resolve()
    (root / "definitions" / "moe").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "moe").mkdir(parents=True, exist_ok=True)
    (root / "solutions").mkdir(exist_ok=True)
    shutil.copy2(Path(run["definition_json"]), root / "definitions" / "moe" / f"{defn}.json")

    src_lines = [json.loads(l) for l in
                 (src_root / "workloads" / "moe" / f"{defn}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def absolutize(inputs):
        out = json.loads(json.dumps(inputs))
        for spec in out.values():
            if spec.get("type") == "safetensors":
                spec["path"] = str((src_root / spec["path"]).resolve())
        return out

    info, lines = {}, []
    # A. shifted expert shard, real routing traces
    for rec in src_lines:
        w = rec["workload"]
        inputs = absolutize(w["inputs"])
        old = int(inputs["local_expert_offset"]["value"])
        new = (old + OFFSET_SHIFT) % 256
        inputs["local_expert_offset"]["value"] = new
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"transfer-shift-{w['uuid']}"))
        lines.append({"definition": defn, "solution": None, "evaluation": None,
                      "workload": {"uuid": uid, "axes": w["axes"], "inputs": inputs}})
        info[uid[:8]] = {"set": "shifted", "tokens": w["axes"]["seq_len"], "offset": new,
                         "from": w["uuid"][:8]}
    # B. new token counts, random logits, a real routing bias
    bias_spec = absolutize(src_lines[0]["workload"]["inputs"])["routing_bias"]
    template = src_lines[0]["workload"]["inputs"]
    for T, off in zip(NEW_SIZES, NEW_OFFSETS):
        inputs = {k: {"type": "random"} for k in template}
        inputs["routing_bias"] = bias_spec
        inputs["local_expert_offset"] = {"type": "scalar", "value": off}
        inputs["routed_scaling_factor"] = template["routed_scaling_factor"]
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"transfer-size-{T}"))
        lines.append({"definition": defn, "solution": None, "evaluation": None,
                      "workload": {"uuid": uid, "axes": {"seq_len": T}, "inputs": inputs}})
        info[uid[:8]] = {"set": "new size", "tokens": T, "offset": off, "from": None}

    (root / "workloads" / "moe" / f"{defn}.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return root, info


def bench_kernel(code: str, repo: Path, root: Path, ref_cache: Path) -> dict:
    (repo / "solution" / "triton" / "kernel.py").write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        env = dict(os.environ, FIB_DATASET_PATH=str(root), PYTHONUTF8="1")
        proc = subprocess.run([sys.executable, str(HERE / "bench_worker.py"), "--out", str(out),
                               "--ref-cache", str(ref_cache)] + BENCH,
                              cwd=repo, env=env, capture_output=True, text=True, timeout=3 * 3600)
        if proc.returncode != 0 or not out.exists():
            return {"_error": ((proc.stdout or "") + (proc.stderr or ""))[-4000:]}
        res = json.loads(out.read_text(encoding="utf-8"))
        return next(iter(res.values())) if res else {"_error": "no results"}


def main():
    with open(HERE / "run_config.toml", "rb") as f:
        cfg = tomllib.load(f)
    run = cfg["run"]
    repo = Path(run["repo_root"]).resolve()
    runs_dir = Path(run["output_dir"])
    out_dir = Path("results") / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    root, info = build_dataset(cfg)
    (out_dir / "_workloads.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ref_cache = Path("transfer_ref_latency.json").resolve()

    todo = [d for m in METHODS for d in sorted(runs_dir.glob(f"{m}_run*")) if (d / "best_kernel.py").exists()
            and d.name.rsplit("_run", 1)[0] == m]
    try:
        for d in todo:
            dest = out_dir / f"{d.name}.json"
            if dest.exists():
                print(f"== {d.name}: already done, skipping")
                continue
            print(f"== {d.name}: benchmarking on {len(info)} unseen workloads", flush=True)
            t0 = time.time()
            res = bench_kernel((d / "best_kernel.py").read_text(encoding="utf-8"), repo, root, ref_cache)
            if "_error" in res:
                print("   worker failed:\n" + res["_error"])
                sys.exit(1)
            dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
            ok = [e for e in res.values() if e.get("status") == "PASSED"]
            msg = f"   passed {len(ok)}/{len(res)}"
            if ok:
                msg += f", mean speedup on passed {sum(e['speedup_factor'] for e in ok) / len(ok):.2f}x"
            print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ref = prompts.load_reference_code(Path(run["definition_json"]))
        (repo / "solution" / "triton" / "kernel.py").write_text(
            prompts.make_seed_kernel(ref, run["entry_point"]), encoding="utf-8")

    report(out_dir, info)


LABEL = {"haiku": "Haiku", "handwritten": "Hand-written (untuned)", "random": "Random search",
         "autotune": "Triton autotune", "sonnet": "Sonnet", "opus": "Opus",
         "autotune_fp8": "Autotune + FP8 dot"}
COLOR = {"haiku": "#2a78d6", "handwritten": "#eb6834", "random": "#1baf7a", "autotune": "#eda100",
         "sonnet": "#e87ba4", "opus": "#008300", "autotune_fp8": "#6250d6"}
MARKER = {"haiku": "o", "handwritten": "s", "random": "^", "autotune": "D", "sonnet": "v", "opus": "P",
          "autotune_fp8": "X"}


def report(out_dir: Path, info: dict):
    data = {}
    for f in sorted(out_dir.glob("*_run*.json")):
        m, k = f.stem.rsplit("_run", 1)
        data[(m, int(k))] = {u[:8]: e for u, e in json.loads(f.read_text(encoding="utf-8")).items()}
    methods = [m for m in METHODS if any(k[0] == m for k in data)]
    n = len(info)
    L = ["# Transfer test: unseen workloads", "",
         f"Each run's best kernel, unchanged, on {n} workloads it was never developed or tuned on: "
         f"{sum(1 for v in info.values() if v['set'] == 'shifted')} real routing traces served by a different "
         f"expert shard (offset +{OFFSET_SHIFT}), and {len(NEW_SIZES)} new token counts {NEW_SIZES} with random "
         "routing logits. Same correctness rule (atol 0.01, rtol 0.01, 95% of elements) and timing code; "
         "3 trials x 20 iterations.", "",
         "## Per run", "",
         "| Run | Passed | Mean speedup, shifted traces | Mean speedup, new sizes | Failures |", "|---|---|---|---|---|"]
    for (m, k), res in sorted(data.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
        ok = {u: e for u, e in res.items() if e.get("status") == "PASSED"}
        def mean(which):
            v = [e["speedup_factor"] for u, e in ok.items() if info[u]["set"] == which]
            return f"{sum(v) / len(v):.2f}x" if v else "-"
        fails = [f"{info[u]['tokens']} tok ({info[u]['set']}): {e.get('status')}"
                 + (f", matched {e['matched_ratio']:.3f}" if e.get("matched_ratio") is not None else "")
                 for u, e in res.items() if e.get("status") != "PASSED"]
        L.append(f"| {m} run{k} | {len(ok)}/{len(res)} | {mean('shifted')} | {mean('new size')} | "
                 f"{'; '.join(fails) if fails else 'none'} |")

    def med(m, u):
        v = [res[u]["speedup_factor"] for (mm, _), res in data.items()
             if mm == m and res.get(u, {}).get("status") == "PASSED"]
        return st.median(v) if v else None

    L += ["", "## New token counts: speedup by size (median of runs that passed)", "",
          "| Tokens | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    new = sorted((u for u in info if info[u]["set"] == "new size"), key=lambda u: info[u]["tokens"])
    for u in new:
        L.append(f"| {info[u]['tokens']} | " + " | ".join(
            (f"{med(m, u):.2f}x" if med(m, u) is not None else "fail") for m in methods) + " |")
    Path("results/transfer.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipped fig5. Run: pip install matplotlib)")
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    order = sorted(info, key=lambda u: (info[u]["tokens"], info[u]["set"]))
    for m in methods:
        pts = [(info[u]["tokens"], med(m, u)) for u in order if med(m, u) is not None]
        by_t = {}
        for t, s in pts:
            by_t.setdefault(t, []).append(s)
        xs = sorted(by_t)
        ax.plot(xs, [st.median(by_t[t]) for t in xs], color=COLOR[m], lw=1.6, marker=MARKER[m], ms=5,
                markeredgecolor="white", markeredgewidth=0.8, label=LABEL[m])
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Tokens in the workload (log scale)")
    ax.set_ylabel("Speedup on unseen workloads (median)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax.set_title("Transfer: speedup on workloads no kernel was developed on", loc="left", fontsize=11)
    fig.savefig("results/fig5_transfer.png")
    print("\nSaved results/transfer.md and results/fig5_transfer.png")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(Path("results") / "transfer", json.loads((Path("results") / "transfer" / "_workloads.json").read_text()))
    else:
        main()
