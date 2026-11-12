# Which Claude tier should drive a kernel-writing agent?

**CIS 4930/6930 ML for Computer Systems, Fall 2026 — final project**
Team: Michael Parker, Nandhu Shankar, Katherine Lingenfelter, Ian Farmer

We put three Claude model tiers (Haiku 5.5, Sonnet 5.5, Opus 5.5) in the same agent loop and
asked each to write a fast Triton kernel for the fused FP8 Mixture-of-Experts (MoE) operation from
the [MLSys 2026 FlashInfer AI Kernel Generation Contest](http://mlsys26.flashinfer.ai/). Every
tier gets the same prompts, the same seed kernel, and the same budget of 15 attempts. We compare
them with each other and with three non-ML baselines given the same budget of timed candidates.

This repository is a fork of the contest starter kit. The original starter-kit README is kept as
[CONTEST_README.md](CONTEST_README.md).

## Main result

Speedup over the PyTorch reference, mean of the 19 contest MoE workloads, on one RTX 5060 Laptop
GPU (8 GB). Median of 3 runs, with the lowest and highest run in brackets.

| Method | Timed candidates per run | Final speedup | API-equivalent cost per run |
|---|---|---|---|
| Haiku 5.5 | 15 | 1.44x (1.43–1.45) | $0.03 |
| Hand-written Triton kernel, untuned (non-ML) | 1 | 8.02x (8.02–8.05) | $0 |
| Random search over tile settings (non-ML) | 15 | 11.18x (9.88–11.33) | $0 |
| Triton autotune (non-ML) | 15 | 11.51x (11.48–11.84) | $0 |
| Autotune + FP8 tensor-core dot (non-ML, ablation) | 15 | 12.19x (12.01–12.35) | $0 |
| Sonnet 5.5 | 15 | 17.54x (14.73–18.00) | $0.98 |
| Opus 5.5 | 15 | 20.17x (18.46–28.53) | $3.07 |

Against the best non-ML baseline (Triton autotune), Sonnet is 1.52x faster and Opus 1.75x faster.
Haiku is slower than every baseline: none of its 45 attempts used Triton.

**Why.** FP8 tensor-core multiplies, used by every Sonnet and Opus kernel, add only 6% when given to
the baseline. A knock-out ablation of Sonnet run 1's kernel ([results/ablation.md](results/ablation.md))
switches its three design choices back to the baseline's: without them it falls from 16.8x to 8.8x.
The largest single factor (−21% when removed) is its rewrite of the second matrix multiply as two
bf16 tensor-core multiplies of a high/low split, which no tile search can reach.

**Transfer.** All 21 best kernels passed 26 workloads they were never developed on; the lead over
autotune shrank to 1.34x (Sonnet) and 1.65x (Opus) ([results/transfer.md](results/transfer.md)). The full table, with every run, is
[results/main_table.md](results/main_table.md). The per-workload breakdown and the design features
of each best kernel are in [results/analysis.md](results/analysis.md), and the transfer test on
unseen workloads is in [results/transfer.md](results/transfer.md).

## Reproduce the main results table

**One command, no GPU needed.** This rebuilds `results/main_table.md` and `results/runs.csv` from
the saved runs in `agent_runs/`:

```
python agent_loop/run_experiments.py --table-only
```

It needs only Python 3.11 or newer. To rebuild the figures as well (needs `pip install matplotlib`
and the dataset, see step 3 below):

```
python agent_loop/analyze_results.py
```

**Rerun everything from scratch** (about 10 to 12 hours on the same laptop; needs the full setup below):
delete or rename `agent_runs/`, then

```
python agent_loop/run_experiments.py      # all agent runs, baselines, final benchmarks, table
python agent_loop/transfer_test.py        # best kernels on 26 unseen workloads
python agent_loop/ablation_knockout.py    # knock-out ablation of Sonnet run 1's kernel
python agent_loop/analyze_results.py      # figures and per-workload analysis
```

Stopping and restarting is safe: finished runs are skipped. New Claude runs will not be
bit-identical to ours, because model sampling is not seeded.

## Setup for a full rerun

Tested on Windows 11, NVIDIA RTX 5060 Laptop GPU (8 GB, compute capability 12.0), driver 596.08,
Python 3.12.10, PyTorch 2.14.1+cu130, triton-windows 3.9.0, flashinfer-bench 0.1.2.

1. **Python environment** (PowerShell, in this folder)
   ```
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install flashinfer-bench
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   pip install triton-windows matplotlib
   $env:PYTHONUTF8="1"
   ```
2. **Dataset**, cloned next to this folder
   ```
   git lfs install
   git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest ..\mlsys26-contest
   $env:FIB_DATASET_PATH="<full path to mlsys26-contest>"
   ```
3. **Memory-patched reference.** The contest reference dequantizes all 32 experts' weights to
   float32 at once, which does not fit in 8 GB. `patched_reference.py` dequantizes one expert at
   a time. `verify_patch.py` checks it is bit-identical to the original.
   ```
   python verify_patch.py
   python apply_patch.py <full path to mlsys26-contest>\definitions\moe\moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
   ```
   `apply_patch.py --restore <same path>` undoes it.
4. **Reference timings**, measured once and reused by every method (`ref_latency.json` in this
   repo holds ours; delete it to re-measure on your GPU)
   ```
   python local_bench.py --ref-cache ref_latency.json
   ```
5. **Claude Code**, used to call the models with a Claude subscription instead of an API key
   ```
   irm https://claude.ai/install.ps1 | iex
   claude        # log in once, then /exit
   ```
6. **Smoke test** (Haiku, 2 attempts, about 5 minutes)
   ```
   python agent_loop/run_experiments.py --smoke
   ```

## How the experiment works

- **Agent loop** (`agent_loop/run_agent.py`): the model gets the task, the reference code, and a
  seed kernel. Each attempt, its kernel is benchmarked on all 19 workloads and the result goes
  back to it: the error text, or the error size, or the speedup. It sees its last attempt and its
  best kernel so far. 15 attempts per run, 3 runs per tier.
- **Correctness**: FlashInfer-Bench's own low-bit evaluator. An output element fails only if its
  absolute error is above 0.01 and its relative error is above 0.01, and at least 95% of elements
  must pass, on every workload. One failing workload fails the whole kernel.
- **Timing**: FlashInfer-Bench's own timing code. Per attempt: 3 trials x 20 iterations. For the
  reported numbers, each run's best kernel is re-measured with 5 trials x 100 iterations.
- **Baselines** (`agent_loop/baseline_kernel.py`, `agent_loop/baseline_search.py`): one
  conventional Triton kernel (routing in PyTorch, then per expert GEMM1+SwiGLU and GEMM2), run
  untuned, tuned by random search, and tuned by Triton's autotuner, each with 15 timed candidates.
  `autotune_fp8` is an ablation that adds the one design choice every Sonnet and Opus kernel made.
- **Model calls** (`agent_loop/llm.py`): `claude -p` with all tools disabled, in an empty folder,
  with our system prompt. Cost is Claude Code's estimate of the same tokens on the paid API;
  nothing was billed.

## How this differs from the official contest setup

| Contest | This project | Why |
|---|---|---|
| NVIDIA B200 | RTX 5060 Laptop GPU, 8 GB | The only GPU we had access to |
| Reference dequantizes all weights at once | One expert at a time (`patched_reference.py`), bit-identical output | 8 GB of GPU memory |
| FlashInfer-Bench worker processes | Same evaluator code called in one process (`local_bench.py`) | Its runners share GPU tensors between processes, which Windows does not support, and keep every trial's inputs in memory at once |
| Speedup over the FlashInfer baseline | Speedup over the patched PyTorch reference | The official baseline targets B200 |

Our speedups are therefore not comparable with the contest leaderboard.

## Repository layout

```
agent_loop/            the experiment
  run_experiments.py   one command: agent runs, baselines, final benchmarks, results table
  run_agent.py         one run of one Claude tier
  prompts.py           system, task and feedback prompts (prompt_template.md explains them)
  llm.py               model backends (Claude Code CLI; paid API; fake for tests)
  bench.py, bench_worker.py   benchmark one kernel in a separate process
  baseline_kernel.py, baseline_search.py   the non-ML baselines
  transfer_test.py     best kernels on unseen workloads
  ablation_knockout.py switches Sonnet's design choices back to the baseline's, one at a time
  analyze_results.py   figures and per-workload analysis
  run_config.toml      every setting, shared by all methods
local_bench.py         in-process FlashInfer-Bench runner for Windows and small GPUs
patched_reference.py, verify_patch.py, apply_patch.py   the memory-patched reference
ref_latency.json       reference timings on our GPU
agent_runs/<method>_run<k>/   every attempt's reply and kernel, log.jsonl, best_kernel.py,
                              summary.json, final_bench.json
results/               main_table.md, runs.csv, analysis.md, transfer.md, ablation.md, figures
report/                final report (Word) and presentation (PowerPoint)
```

## AI use disclosure

- The subject of the study is AI: Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 wrote every kernel in
  `agent_runs/haiku_*`, `sonnet_*` and `opus_*` inside our agent loop.
- The experiment code in this repository (agent loop, in-process benchmark runner, reference
  patch, baseline search, analysis scripts) was written with the help of Claude (Anthropic) in a
  chat assistant, directed, run and checked by the team.
- The "hand-written" baseline kernel was also drafted with Claude in a chat assistant, outside
  the agent loop: one design, written once, never shown benchmark feedback. It is non-ML in how it
  is tuned (random search and autotune), not in who typed it.
- AI assistance was also used in drafting the report and slides. All results come from runs on
  our own hardware, saved in `agent_runs/`.
