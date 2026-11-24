"""
Profiler breakdown: WHERE does each method's time go?

The knock-out ablation says which design changes matter. This says where the time is spent, from
the CUDA profiler, for the median run of each method (by final speedup):

  - how many GPU kernels one call launches
  - GPU time split into: routing/setup (PyTorch ops + small Triton kernels), GEMM1 (+SwiGLU),
    GEMM2 (+scatter), library GEMMs (cuBLAS, used by kernels that call torch matmul)
  - idle gap = wall-clock time per call minus GPU-busy time. This is CPU/launch overhead: Python
    and PyTorch dispatch between kernels while the GPU waits.

Inputs are random (same shapes and dtypes as the contest), at 7, 64, 1024 and 8192 tokens.
Numbers are per call, averaged over several calls after warmup (so autotuning is excluded).

From the Nvidia-AI-Kernel folder (venv active):
    python agent_loop/profile_kernels.py
Takes about 5-10 minutes.  --report-only rebuilds the table and figure from saved JSON.

Writes  results/profile/<run>.json     raw numbers, incl. the top kernels by GPU time
        results/profile.md             tables
        results/fig6_profile.png       stacked time breakdown per method
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("agent_runs")
OUT = Path("results") / "profile"
METHODS = ["haiku", "handwritten", "random", "autotune", "autotune_fp8", "sonnet", "opus"]
LABEL = {"haiku": "Haiku", "handwritten": "Hand-written", "random": "Random search",
         "autotune": "Autotune", "autotune_fp8": "Autotune + FP8", "sonnet": "Sonnet", "opus": "Opus"}
TOKENS = [7, 64, 1024, 8192]
H, I, E, EL = 7168, 2048, 256, 32
CATS = ["Routing / setup", "GEMM1", "GEMM2", "Library GEMM", "Idle (launch overhead)"]


def median_runs() -> dict:
    rows = list(csv.DictReader(open("results/runs.csv", encoding="utf-8")))
    pick = {}
    for m in METHODS:
        r = sorted((float(x["final_speedup"]), int(x["run"])) for x in rows if x["tier"] == m)
        if r:
            pick[m] = r[len(r) // 2]   # (speedup, run) of the median run
    return pick


def make_weights(dev):
    import torch
    g = torch.Generator(device=dev).manual_seed(0)
    w1 = torch.empty((EL, 2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
    w2 = torch.empty((EL, H, I), dtype=torch.float8_e4m3fn, device=dev)
    for e in range(EL):   # one expert at a time: a float32 copy of all weights would not fit in 8 GB
        w1[e] = torch.randn((2 * I, H), generator=g, device=dev).to(torch.float8_e4m3fn)
        w2[e] = torch.randn((H, I), generator=g, device=dev).to(torch.float8_e4m3fn)
    w1s = torch.rand((EL, 2 * I // 128, H // 128), generator=g, device=dev) * 0.01 + 0.001
    w2s = torch.rand((EL, H // 128, I // 128), generator=g, device=dev) * 0.01 + 0.001
    return w1, w1s, w2, w2s


def make_inputs(T, weights, dev):
    import torch
    g = torch.Generator(device=dev).manual_seed(T)
    w1, w1s, w2, w2s = weights
    return dict(
        routing_logits=torch.randn((T, E), generator=g, device=dev),
        routing_bias=(torch.randn((E,), generator=g, device=dev) * 0.01).to(torch.bfloat16),
        hidden_states=torch.randn((T, H), generator=g, device=dev).to(torch.float8_e4m3fn),
        hidden_states_scale=torch.rand((H // 128, T), generator=g, device=dev) * 0.01 + 0.001,
        gemm1_weights=w1, gemm1_weights_scale=w1s, gemm2_weights=w2, gemm2_weights_scale=w2s,
        local_expert_offset=64, routed_scaling_factor=2.5)


def load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location(f"k_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


def triton_names(path: Path) -> set:
    """Names of the @triton.jit functions in a kernel file (how they appear in the profiler)."""
    src = path.read_text(encoding="utf-8")
    return set(re.findall(r"@triton\.jit[^\n]*\n(?:@[^\n]*\n)*def\s+(\w+)", src))


def category(name: str, tnames: set) -> str:
    n = name.lower()
    base = name.split("(")[0].strip()
    if base in tnames or any(base.startswith(t) for t in tnames):
        if "gemm1" in n:       # every Triton GEMM in this project is named _gemm1_* / _gemm2_*
            return "GEMM1"
        if "gemm2" in n:
            return "GEMM2"
        return "Routing / setup"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "cublas", "gemv", "matmul")):
        return "Library GEMM"
    return "Routing / setup"


def device_us(evt) -> float:
    for attr in ("device_time_total", "cuda_time_total", "device_time", "cuda_time"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


def profile_one(fn, inputs, n_calls, tnames):
    import torch
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        fn(**inputs)
    torch.cuda.synchronize()
    # wall clock per call (what the benchmark sees)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn(**inputs)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000 / n_calls
    # GPU kernels per call
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_calls):
            fn(**inputs)
        torch.cuda.synchronize()
    per = {}
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = device_us(e)
        if us <= 0:
            continue
        d = per.setdefault(e.key, [0, 0.0])
        d[0] += e.count
        d[1] += us
    kernels = sorted(({"name": k[:120], "category": category(k, tnames), "launches": c / n_calls,
                       "gpu_ms": us / 1000 / n_calls} for k, (c, us) in per.items()),
                     key=lambda x: -x["gpu_ms"])
    busy = sum(k["gpu_ms"] for k in kernels)
    cats = {c: 0.0 for c in CATS}
    for k in kernels:
        cats[k["category"]] += k["gpu_ms"]
    cats["Idle (launch overhead)"] = max(wall_ms - busy, 0.0)
    return {"wall_ms": wall_ms, "gpu_busy_ms": busy, "launches": sum(k["launches"] for k in kernels),
            "breakdown_ms": cats, "top_kernels": kernels[:8]}


def main():
    import torch
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    picks = median_runs()
    extra = [a for a in sys.argv[1:] if not a.startswith("--")]   # e.g. opus_run2: profile extra runs too
    print("Median run per method:", {m: f"run{r} ({s:.2f}x)" for m, (s, r) in picks.items()})
    print("Generating weights...", flush=True)
    weights = make_weights(dev)
    inputs = {T: make_inputs(T, weights, dev) for T in TOKENS}
    rows = {(r["tier"], int(r["run"])): float(r["final_speedup"])
            for r in csv.DictReader(open("results/runs.csv", encoding="utf-8"))}
    jobs = [(m, spd, run) for m, (spd, run) in picks.items()]
    for a in extra:
        m, k = a.rsplit("_run", 1)
        jobs.append((m, rows[(m, int(k))], int(k)))
    for m, spd, run in jobs:
        name = f"{m}_run{run}"
        dest = OUT / f"{name}.json"
        if dest.exists():
            print(f"== {name}: already done, skipping")
            continue
        print(f"== {name}", flush=True)
        path = RUNS / name / "best_kernel.py"
        fn = load_kernel(path)
        tnames = triton_names(path)
        res = {"method": m, "run": run, "final_speedup": spd, "by_tokens": {}}
        for T in TOKENS:
            r = profile_one(fn, inputs[T], 10 if T <= 1024 else 3, tnames)
            res["by_tokens"][str(T)] = r
            b = r["breakdown_ms"]
            print(f"   {T:>5} tok: {r['wall_ms']:8.3f} ms/call, {r['launches']:5.0f} launches, "
                  + ", ".join(f"{c.split(' ')[0]} {v:.3f}" for c, v in b.items()), flush=True)
            torch.cuda.empty_cache()
        dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    report()


def report():
    data, extras = {}, []
    picks = median_runs()
    for f in OUT.glob("*_run*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if picks.get(d["method"], (0, None))[1] == d["run"]:
            data[d["method"]] = d
        else:
            extras.append(d)
    methods = [m for m in METHODS if m in data]
    L = ["# Profiler breakdown", "",
         "Median run of each method (by final speedup), random inputs with contest shapes, per call after "
         "warmup. *Idle* = wall-clock time minus GPU-busy time: Python/PyTorch dispatch and kernel-launch "
         "overhead while the GPU waits. Times in ms.", ""]
    for T in TOKENS:
        L += [f"## {T} tokens", "",
              "| Method | Wall ms | Kernel launches | " + " | ".join(CATS) + " |",
              "|---|---|---|" + "---|" * len(CATS)]
        for m in methods:
            r = data[m]["by_tokens"][str(T)]
            b = r["breakdown_ms"]
            L.append(f"| {LABEL[m]} (run{data[m]['run']}) | {r['wall_ms']:.3f} | {r['launches']:.0f} | "
                     + " | ".join(f"{b[c]:.3f} ({100 * b[c] / r['wall_ms']:.0f}%)" for c in CATS) + " |")
        for d in extras:
            r = d["by_tokens"][str(T)]
            b = r["breakdown_ms"]
            L.append(f"| *{LABEL[d['method']]} (run{d['run']}, not median)* | {r['wall_ms']:.3f} | {r['launches']:.0f} | "
                     + " | ".join(f"{b[c]:.3f} ({100 * b[c] / r['wall_ms']:.0f}%)" for c in CATS) + " |")
        L.append("")
    L += ["## Top kernels by GPU time (1024 tokens)", ""]
    for m in methods:
        L.append(f"**{LABEL[m]}**: " + "; ".join(
            f"`{k['name'][:60]}` {k['gpu_ms']:.3f} ms x{k['launches']:.0f} [{k['category']}]"
            for k in data[m]["by_tokens"]["1024"]["top_kernels"][:4]))
        L.append("")
    Path("results/profile.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped fig6)")
        return
    colors = {"Routing / setup": "#eda100", "GEMM1": "#2a78d6", "GEMM2": "#1baf7a",
              "Library GEMM": "#6250d6", "Idle (launch overhead)": "#b8b6ae"}
    show = [7, 1024, 8192]
    methods = [m for m in methods if m != "haiku"]   # Haiku is ~10x slower; its bar would hide the rest
    fig, axes = plt.subplots(1, len(show), figsize=(12, 4.2))
    for ax, T in zip(axes, show):
        y = range(len(methods))
        left = [0.0] * len(methods)
        for c in CATS:
            v = [data[m]["by_tokens"][str(T)]["breakdown_ms"][c] for m in methods]
            ax.barh(list(y), v, left=left, color=colors[c], edgecolor="white", linewidth=1, label=c)
            left = [a + b for a, b in zip(left, v)]
        ax.set_yticks(list(y))
        ax.set_yticklabels([LABEL[m] for m in methods] if T == show[0] else [])
        ax.invert_yaxis()
        ax.set_title(f"{T} tokens", loc="left", fontsize=10)
        ax.set_xlabel("ms per call")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#e4e3df")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.7, -0.18), ncol=5)
    fig.suptitle("Where each method's time goes (median run, per call; Haiku omitted, see profile.md)", x=0.01, ha="left", fontsize=11)
    fig.savefig("results/fig6_profile.png", dpi=200, bbox_inches="tight")
    print("\nSaved results/profile.md and results/fig6_profile.png")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
    else:
        main()
