"""
Builds the report figures and per-workload tables from saved results.

From the Nvidia-AI-Kernel folder (FIB_DATASET_PATH set, so workload token counts can be read):
    python agent_loop/analyze_results.py

Reads   agent_runs/<method>_run<k>/{final_bench.json, summary.json, log.jsonl}
Writes  results/fig1_speedup_by_method.png
        results/fig2_speedup_vs_tokens.png
        results/fig3_cost_vs_speedup.png
        results/fig4_progress_over_attempts.png
        results/analysis.md   (per-workload table, worst cases, kernel design features)
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent_runs")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
DEF = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Fixed method order and colors (color follows the method in every figure).
METHODS = ["haiku", "handwritten", "random", "autotune", "sonnet", "opus", "autotune_fp8"]
LABEL = {"haiku": "Haiku", "handwritten": "Hand-written (untuned)", "random": "Random search",
         "autotune": "Triton autotune", "sonnet": "Sonnet", "opus": "Opus",
         "autotune_fp8": "Autotune + FP8 dot (ablation)"}
COLOR = {"haiku": "#2a78d6", "handwritten": "#eb6834", "random": "#1baf7a", "autotune": "#eda100",
         "sonnet": "#e87ba4", "opus": "#008300", "autotune_fp8": "#6250d6"}
MARKER = {"haiku": "o", "handwritten": "s", "random": "^", "autotune": "D", "sonnet": "v", "opus": "P",
          "autotune_fp8": "X"}
CLAUDE = {"haiku", "sonnet", "opus"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load_runs():
    runs = {}
    for d in sorted(RUNS_DIR.glob("*_run*")):
        m, k = d.name.rsplit("_run", 1)
        if m not in METHODS or not (d / "final_bench.json").exists():
            continue
        fb = json.loads((d / "final_bench.json").read_text(encoding="utf-8"))
        sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        log = [json.loads(l) for l in (d / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        src = (d / "best_kernel.py").read_text(encoding="utf-8") if (d / "best_kernel.py").exists() else ""
        runs[(m, int(k))] = {"final": fb, "summary": sm, "log": log, "src": src,
                             "wl": {w["id"]: w for w in fb.get("workloads", [])}}
    return runs


def token_counts():
    root = os.environ.get("FIB_DATASET_PATH")
    cands = [Path(root)] if root else []
    cands += [Path("../mlsys26-contest"), Path.home() / "mlsys26-contest"]
    for r in cands:
        f = r / "workloads" / "moe" / f"{DEF}.jsonl"
        if f.exists():
            out = {}
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    w = json.loads(line)["workload"]
                    out[w["uuid"][:8]] = w["axes"]["seq_len"]
            return out
    raise SystemExit("Workload file not found: set FIB_DATASET_PATH to the mlsys26-contest folder.")


def present(runs):
    return [m for m in METHODS if any(k[0] == m for k in runs)]


def per_method(runs, m):
    return [runs[k] for k in sorted(runs) if k[0] == m]


def fig1(runs, methods):
    meds = {m: st.median(r["final"]["mean_speedup"] for r in per_method(runs, m)) for m in methods}
    order = sorted(methods, key=lambda m: meds[m])
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(order) + 1.2))
    for y, m in enumerate(order):
        vals = [r["final"]["mean_speedup"] for r in per_method(runs, m)]
        ax.plot([min(vals), max(vals)], [y, y], color=COLOR[m], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(vals, [y] * len(vals), s=36, color=COLOR[m], marker=MARKER[m],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([meds[m]], [y], s=140, marker="|", color=INK, zorder=4, linewidth=2)
        ax.text(max(vals) + 0.6, y, f"{meds[m]:.1f}×", va="center", color=INK, fontsize=9)
    if "autotune" in meds:
        ax.axvline(meds["autotune"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(meds["autotune"], len(order) - 0.45, "best non-ML baseline", color=INK2, fontsize=8, ha="center")
    ax.set_yticks(range(len(order)), [LABEL[m] + ("" if m in CLAUDE else "  (non-ML)") for m in order])
    ax.set_xlabel("Speedup over patched PyTorch reference (mean of 19 workloads)")
    ax.set_xlim(0, max(r["final"]["mean_speedup"] for r in runs.values()) * 1.12)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Final speedup by method: dots = 3 runs, tick = median", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "fig1_speedup_by_method.png")
    plt.close(fig)
    return meds


def fig2(runs, methods, T):
    ids = sorted(T, key=T.get)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for m in methods:
        ys = [st.median(r["wl"][i]["speedup"] for r in per_method(runs, m)) for i in ids]
        xs = [T[i] for i in ids]
        ax.plot(xs, ys, color=COLOR[m], lw=1.6, marker=MARKER[m], ms=5, markeredgecolor="white",
                markeredgewidth=0.8, label=LABEL[m])
    ax.set_xscale("log")
    ax.set_xlabel("Tokens in the workload (log scale)")
    ax.set_ylabel("Speedup (median of 3 runs)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.8, T[ids[-1]] * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.text(0.01, -0.36, "Workloads with similar token counts differ in how many tokens reach this GPU's 32 experts, "
            "which causes the zig-zags.", transform=ax.transAxes, fontsize=7.5, color=INK2)
    ax.set_title("Speedup by workload size: every method falls off at 10k+ tokens", loc="left",
                 fontsize=11, color=INK)
    fig.savefig(OUT / "fig2_speedup_vs_tokens.png")
    plt.close(fig)


def fig3(runs, meds):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for m in ["haiku", "sonnet", "opus"]:
        rs = per_method(runs, m)
        if not rs:
            continue
        ax.scatter([r["summary"]["total_cost_usd"] for r in rs], [r["final"]["mean_speedup"] for r in rs],
                   s=60, color=COLOR[m], marker=MARKER[m], edgecolor="white", linewidth=1.2, label=LABEL[m], zorder=3)
    if "autotune" in meds:
        ax.axhline(meds["autotune"], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(3.4, meds["autotune"] + 0.5, f"Triton autotune, $0: {meds['autotune']:.1f}×", color=INK2,
                fontsize=8, ha="right")
    ax.set_xlabel("API-equivalent cost per run (USD, not billed)")
    ax.set_ylabel("Final speedup")
    ax.set_xlim(left=-0.1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Cost vs. speedup: each point is one 15-attempt run", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "fig3_cost_vs_speedup.png")
    plt.close(fig)


def fig4(runs):
    tiers = [m for m in ["haiku", "sonnet", "opus", "random"] if any(k[0] == m for k in runs)]
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.0 * len(tiers), 3.4), sharey=True)
    for ax, m in zip(axes, tiers):
        for r in per_method(runs, m):
            best, xs, ys, ok_x, ok_y, bad_x = 0.0, [], [], [], [], []
            for e in r["log"]:
                sp = e.get("mean_speedup")
                if e.get("status") == "CORRECT" and sp:
                    best = max(best, sp)
                    ok_x.append(e["attempt"]); ok_y.append(sp)
                else:
                    bad_x.append(e["attempt"])
                xs.append(e["attempt"]); ys.append(best if best else None)
            ax.step([x for x, y in zip(xs, ys) if y], [y for y in ys if y], where="post",
                    color=COLOR[m], lw=2, alpha=0.9)
            ax.scatter(ok_x, ok_y, s=12, color=COLOR[m], edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter(bad_x, [0.3] * len(bad_x), s=14, marker="x", color=INK2, linewidth=1, zorder=3)
        ax.set_title(LABEL[m], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Attempt / candidate")
        ax.set_xticks([1, 5, 10, 15])
    axes[0].set_ylabel("Speedup (per-attempt benchmark)")
    fig.suptitle("Best speedup so far over the 15-attempt budget (3 runs each; × = failed attempt)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_progress_over_attempts.png")
    plt.close(fig)


def _gemm1_fp8(code: str) -> bool:
    """True if the first tl.dot (GEMM1) gets FP8 operands: no unconditional .to(bf16/fp16) on them.
    A conversion indented deeper than the dot (inside `if not FP8:` / `else:`) is a fallback path."""
    lines = code.splitlines()
    for n, line in enumerate(lines):
        m = re.search(r"tl\.dot\(\s*([A-Za-z_]\w*)\s*,\s*(?:tl\.trans\()?\s*([A-Za-z_]\w*)", line)
        if not m:
            continue
        indent = len(line) - len(line.lstrip())
        for name in m.groups():
            for prev in lines[:n]:
                if re.match(rf"\s*{name}\s*=.*\.to\(tl\.(bfloat16|float16)\)", prev):
                    if len(prev) - len(prev.lstrip()) <= indent:
                        return False
        return True
    return False


def features(src: str) -> dict:
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    jit = code.count("@triton.jit")
    return {
        "triton_kernels": jit,
        "single_grouped_launch": ("tile_expert" in code) or ("searchsorted" in code) or ("tile_row0" in code),
        "fp8_dot_gemm1": jit > 0 and _gemm1_fp8(code),
        "routing_in_triton": bool(re.search(r"def _route\w*\(", code)) and "tl.program_id" in code,
    }


def write_md(runs, methods, T, meds):
    ids = sorted(T, key=T.get)
    L = ["# Per-workload analysis", "",
         "Speedup = patched PyTorch reference latency / kernel latency, full benchmark (5 trials x 100 iterations). "
         "Each cell is the median of 3 runs.", "",
         "| Tokens | Workload | Reference ms | " + " | ".join(LABEL[m] for m in methods) + " |",
         "|---|---|---|" + "---|" * len(methods)]
    any_run = next(iter(runs.values()))
    for i in ids:
        ref = any_run["wl"][i]["reference_latency_ms"]
        cells = [f"{st.median(r['wl'][i]['speedup'] for r in per_method(runs, m)):.2f}x" for m in methods]
        L.append(f"| {T[i]} | {i} | {ref:.1f} | " + " | ".join(cells) + " |")
    L += ["", "## Worst workload per run", "", "| Run | Mean speedup | Worst workload (tokens) | Worst speedup |",
          "|---|---|---|---|"]
    for (m, k), r in sorted(runs.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
        wid = min(r["wl"], key=lambda i: r["wl"][i]["speedup"])
        L.append(f"| {m} run{k} | {r['final']['mean_speedup']:.2f}x | {wid} ({T[wid]}) | {r['wl'][wid]['speedup']:.2f}x |")
    if "autotune" in methods:
        L += ["", "## Claude vs. best non-ML baseline (autotune), by workload size", "",
              "| Tokens | " + " | ".join(f"{LABEL[m]} / autotune" for m in methods if m in CLAUDE) + " |",
              "|---|" + "---|" * len([m for m in methods if m in CLAUDE])]
        for i in ids:
            a = st.median(r["wl"][i]["speedup"] for r in per_method(runs, "autotune"))
            L.append(f"| {T[i]} | " + " | ".join(
                f"{st.median(r['wl'][i]['speedup'] for r in per_method(runs, m)) / a:.2f}" for m in methods if m in CLAUDE) + " |")
    L += ["", "## Design features of each best kernel (automatic scan; spot-checked by hand)", "",
          "FP8 dot = GEMM1 multiplies FP8 values directly on the tensor cores. Sonnet run 1 and Opus run 3 try FP8 "
          "first and fall back to bf16 only if FP8 raises an error.", "",
          "| Run | Final speedup | Triton kernels | One grouped launch for all experts | FP8 tensor-core dot in GEMM1 | Routing in Triton |",
          "|---|---|---|---|---|---|"]
    for (m, k), r in sorted(runs.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
        f = features(r["src"])
        yn = lambda b: "yes" if b else "no"
        L.append(f"| {m} run{k} | {r['final']['mean_speedup']:.2f}x | {f['triton_kernels']} | {yn(f['single_grouped_launch'])} | "
                 f"{yn(f['fp8_dot_gemm1'])} | {yn(f['routing_in_triton'])} |")
    (OUT / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    T = token_counts()
    methods = present(runs)
    meds = fig1(runs, methods)
    fig2(runs, methods, T)
    fig3(runs, meds)
    fig4(runs)
    write_md(runs, methods, T, meds)
    print("Wrote", ", ".join(sorted(p.name for p in OUT.glob("fig*.png"))), "and analysis.md to", OUT)


if __name__ == "__main__":
    main()
