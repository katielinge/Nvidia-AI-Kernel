"""
Edge-case correctness test: do the best kernels survive inputs built to break them?

The contest workloads (and the transfer test) use realistic routing. Real kernels break on the
unusual cases: experts that receive no tokens, one expert that receives everything, sizes that are
not a multiple of the tile, and extreme FP8 scales. This script builds 11 such workloads and runs
every run's best_kernel.py on them, unchanged.

  Routing (bias = 0, local experts = offset .. offset+31):
    hot_8_experts      256 tok  every token picks the SAME 8 local experts; the other 24 get nothing
    skewed_75pct       1024 tok 75% of tokens pile onto 8 local experts, the rest route normally
    one_token_each      32 tok  token i picks local expert i (+7 non-local): every expert gets 1 token
    no_local_tokens     64 tok  no token picks a local expert -> the correct output is all zeros
  Sizes that are not powers of two (random routing, real bias):
    odd_17, odd_129, odd_1000, odd_4097
  FP8 activation scales (64 tok, random routing, real bias):
    scale_large (all 100), scale_small (all 0.01), scale_mixed (log-uniform 1e-3 .. 1e3 per block)

Exact score ties are left out on purpose: the reference uses torch.topk(sorted=False), so which
tied expert wins is undefined, and a mismatch there would not be a kernel bug.

Same correctness rule as everywhere else (atol 0.01, rtol 0.01, 95% of elements). Timing is light
(3 warmup x 10 iters x 1 trial): the point here is correctness; speedups are only indicative.

From the Nvidia-AI-Kernel folder (venv active):
    python agent_loop/edge_case_test.py
Safe to stop and restart: finished kernels are skipped.  --report-only rebuilds the tables.

Writes  edge_dataset/                      the 11 workloads (git-ignored)
        results/edge_cases/<run>.json      raw per-workload results
        results/edge_cases.md              tables
"""
from __future__ import annotations

import json
import shutil
import statistics as st
import sys
import time
import tomllib
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402
from transfer_test import METHODS, LABEL, bench_kernel  # noqa: E402

BENCH_NOTE = "3 warmup x 10 iters x 1 trial"
import transfer_test  # noqa: E402
transfer_test.BENCH = ["--warmup", "3", "--iters", "10", "--trials", "1"]

E, G, H_BLOCKS = 256, 32, 56          # global experts, experts per group, hidden blocks
SEED = 1234


def _logits(T, base=-6.0, noise=0.5, gen=None):
    import torch
    return base + noise * (torch.rand(T, E, generator=gen) - 0.5)


def build_cases():
    """Return a list of (name, tokens, offset, tensors{key: tensor}, notes)."""
    import torch
    g = torch.Generator().manual_seed(SEED)
    cases = []

    # hot: all tokens -> local experts 0..7 (offset 0 => experts 0..7, all in group 0)
    T = 256
    L = _logits(T, gen=g)
    L[:, 0:8] = 6.0 + (torch.rand(T, 8, generator=g) - 0.5)
    cases.append(("hot_8_experts", T, 0, {"routing_logits": L}, "8 of 32 local experts get every token"))

    # skewed: offset 64; 768 tokens -> experts 64..71, 256 tokens random
    T = 1024
    L = torch.randn(T, E, generator=g)
    L[:768] = _logits(768, gen=g)
    L[:768, 64:72] = 6.0 + (torch.rand(768, 8, generator=g) - 0.5)
    cases.append(("skewed_75pct", T, 64, {"routing_logits": L}, "75% of tokens on 8 local experts"))

    # one token per local expert: token i -> local expert i (group 0) + 7 experts in group 4
    T = 32
    L = _logits(T, gen=g)
    for i in range(T):
        L[i, i] = 6.0 + float(torch.rand(1, generator=g)) - 0.5
        L[i, 128:135] = 6.0 + (torch.rand(7, generator=g) - 0.5)
    cases.append(("one_token_each", T, 0, {"routing_logits": L}, "every local expert gets exactly 1 token"))

    # no local tokens: offset 0, everything routed to group 5 (experts 160..167)
    T = 64
    L = _logits(T, gen=g)
    L[:, 160:168] = 6.0 + (torch.rand(T, 8, generator=g) - 0.5)
    cases.append(("no_local_tokens", T, 0, {"routing_logits": L}, "correct output is all zeros"))

    for T, off in [(17, 32), (129, 96), (1000, 160), (4097, 224)]:
        cases.append((f"odd_{T}", T, off, {}, "random routing, real bias"))

    T = 64
    cases.append(("scale_large", T, 0, {"hidden_states_scale": torch.full((H_BLOCKS, T), 100.0)},
                  "activation scales all 100"))
    cases.append(("scale_small", T, 0, {"hidden_states_scale": torch.full((H_BLOCKS, T), 0.01)},
                  "activation scales all 0.01"))
    mixed = torch.pow(10.0, torch.rand(H_BLOCKS, T, generator=g) * 6 - 3)
    cases.append(("scale_mixed", T, 0, {"hidden_states_scale": mixed}, "scales log-uniform 1e-3..1e3"))
    return cases


def build_dataset(cfg) -> tuple[Path, dict]:
    import torch
    from safetensors.torch import save_file
    run = cfg["run"]
    src_root = Path(run["dataset_path"]).resolve()
    defn = run["definition_name"]
    root = Path("edge_dataset").resolve()
    (root / "definitions" / "moe").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "moe").mkdir(parents=True, exist_ok=True)
    (root / "blob").mkdir(exist_ok=True)
    (root / "solutions").mkdir(exist_ok=True)
    shutil.copy2(Path(run["definition_json"]), root / "definitions" / "moe" / f"{defn}.json")

    first = json.loads((src_root / "workloads" / "moe" / f"{defn}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    template = first["workload"]["inputs"]
    real_bias = dict(template["routing_bias"])
    real_bias["path"] = str((src_root / real_bias["path"]).resolve())

    info, lines = {}, []
    for name, T, off, tensors, note in build_cases():
        inputs = {k: {"type": "random"} for k in template}
        inputs["local_expert_offset"] = {"type": "scalar", "value": off}
        inputs["routed_scaling_factor"] = template["routed_scaling_factor"]
        routing_case = "routing_logits" in tensors
        if routing_case:   # controlled routing: zero bias so the logits alone decide
            tensors = dict(tensors, routing_bias=torch.zeros(E, dtype=torch.bfloat16))
        else:
            inputs["routing_bias"] = real_bias
        if tensors:
            path = root / "blob" / f"{name}.safetensors"
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
            for k in tensors:
                inputs[k] = {"type": "safetensors", "path": str(path), "tensor_key": k}
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"edge-{name}"))
        lines.append({"definition": defn, "solution": None, "evaluation": None,
                      "workload": {"uuid": uid, "axes": {"seq_len": T}, "inputs": inputs}})
        info[uid[:8]] = {"case": name, "tokens": T, "offset": off, "note": note}

    (root / "workloads" / "moe" / f"{defn}.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return root, info


def main():
    with open(HERE / "run_config.toml", "rb") as f:
        cfg = tomllib.load(f)
    run = cfg["run"]
    repo = Path(run["repo_root"]).resolve()
    runs_dir = Path(run["output_dir"])
    out_dir = Path("results") / "edge_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    root, info = build_dataset(cfg)
    (out_dir / "_workloads.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ref_cache = Path("edge_ref_latency.json").resolve()

    todo = [d for m in METHODS for d in sorted(runs_dir.glob(f"{m}_run*"))
            if (d / "best_kernel.py").exists() and d.name.rsplit("_run", 1)[0] == m]
    try:
        for d in todo:
            dest = out_dir / f"{d.name}.json"
            if dest.exists():
                print(f"== {d.name}: already done, skipping")
                continue
            print(f"== {d.name}: {len(info)} edge-case workloads", flush=True)
            t0 = time.time()
            res = bench_kernel((d / "best_kernel.py").read_text(encoding="utf-8"), repo, root, ref_cache)
            if "_error" in res:
                print("   worker failed:\n" + res["_error"])
                sys.exit(1)
            dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
            bad = [info[u[:8]]["case"] for u, e in res.items() if e.get("status") != "PASSED"]
            print(f"   passed {len(res) - len(bad)}/{len(res)}"
                  + (f"  FAILED: {', '.join(bad)}" if bad else "") + f"  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ref = prompts.load_reference_code(Path(run["definition_json"]))
        (repo / "solution" / "triton" / "kernel.py").write_text(
            prompts.make_seed_kernel(ref, run["entry_point"]), encoding="utf-8")
    report(out_dir, info)


def report(out_dir: Path, info: dict):
    data = {}
    for f in sorted(out_dir.glob("*_run*.json")):
        m, k = f.stem.rsplit("_run", 1)
        data[(m, int(k))] = {u[:8]: e for u, e in json.loads(f.read_text(encoding="utf-8")).items()}
    methods = [m for m in METHODS if any(k[0] == m for k in data)]
    order = list(info)  # build order

    def status(e):
        s = e.get("status", "?")
        if s == "PASSED":
            return "pass"
        mr = e.get("matched_ratio")
        return s.lower().replace("_", " ") + (f" ({mr:.2f})" if mr is not None else "")

    L = ["# Edge-case correctness test", "",
         f"Each run's best kernel, unchanged, on {len(info)} workloads built to stress it. Correctness rule "
         "as everywhere else (atol 0.01, rtol 0.01, 95% of elements); "
         f"timing {BENCH_NOTE}, so speedups are indicative only.", "",
         "| Case | Tokens | What it tests |", "|---|---|---|"]
    for u in order:
        L.append(f"| {info[u]['case']} | {info[u]['tokens']} | {info[u]['note']} |")

    L += ["", "## Runs that passed each case", "",
          "| Case | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for u in order:
        cells = []
        for m in methods:
            runs = [res.get(u, {}) for (mm, _), res in data.items() if mm == m]
            cells.append(f"{sum(e.get('status') == 'PASSED' for e in runs)}/{len(runs)}")
        L.append(f"| {info[u]['case']} | " + " | ".join(cells) + " |")

    L += ["", "## Median speedup on each case (runs that passed)", "",
          "| Case | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for u in order:
        cells = []
        for m in methods:
            v = [res[u]["speedup_factor"] for (mm, _), res in data.items()
                 if mm == m and res.get(u, {}).get("status") == "PASSED"]
            cells.append(f"{st.median(v):.2f}x" if v else "-")
        L.append(f"| {info[u]['case']} | " + " | ".join(cells) + " |")

    L += ["", "## Every failure", ""]
    fails = [(f"{m} run{k}", info[u]["case"], status(e)) for (m, k), res in sorted(
        data.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])) for u, e in res.items()
        if e.get("status") != "PASSED"]
    if fails:
        L += ["| Run | Case | Result (matched fraction) |", "|---|---|---|"]
        L += [f"| {r} | {c} | {s} |" for r, c, s in fails]
    else:
        L.append("None: every kernel passed every case.")
    Path("results/edge_cases.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nSaved results/edge_cases.md")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        out = Path("results") / "edge_cases"
        report(out, json.loads((out / "_workloads.json").read_text(encoding="utf-8")))
    else:
        main()
