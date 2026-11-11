# Agent loop: Claude tiers writing the FlashInfer MoE kernel (laptop version)

Everything runs on one Windows laptop (RTX 5060, 8 GB). No API key: the model calls go through
the Claude Code CLI using your Claude subscription login.

## One-time setup (PowerShell, in the Nvidia-AI-Kernel folder)
1. Python venv with flashinfer-bench, torch (cu130) and `triton-windows` (already done).
2. Patched reference installed with `apply_patch.py` (already done), `ref_latency.json` saved (already done).
3. Install Claude Code and log in once:
       irm https://claude.ai/install.ps1 | iex
       claude            # log in with your Claude account in the browser, then type /exit
       claude -p --model haiku "say hi"     # should print a short reply

## Every session
       cd C:\Users\Katie\Nvidia-AI-Kernel
       .\.venv\Scripts\Activate.ps1
       $env:PYTHONUTF8="1"
       $env:FIB_DATASET_PATH="C:\Users\Katie\mlsys26-contest"

## Run
       python agent_loop/run_experiments.py --smoke      # Haiku, 2 attempts: checks everything works
       python agent_loop/run_experiments.py              # 3 tiers x 3 runs + final benchmark + table
       python agent_loop/run_experiments.py --table-only # rebuild results/main_table.md

Stopping and restarting is safe; finished runs are skipped. If your plan's usage limit is hit, the
loop waits and retries (settings in run_config.toml), and stops cleanly after 8 hours of waiting.

## Files
- run_config.toml   settings shared by every tier (iterations, per-attempt benchmark, tiers)
- run_experiments.py  whole pipeline: agent runs -> final benchmark of each best kernel -> table
- run_agent.py      one run of one tier
- llm.py            ClaudeCodeClient (subscription, default), ClaudeClient (paid API), FakeClient
- bench.py / bench_worker.py  benchmark each attempt in a separate process using ../local_bench.py
- prompts.py        system prompt, task prompt, feedback prompt

## Outputs
- agent_runs/<tier>_run<k>/   log.jsonl, every attempt's reply and kernel, best_kernel.py,
                              summary.json, final_bench.json
- results/main_table.md, results/runs.csv

## Notes for the report
- Cost is Claude Code's API-equivalent estimate (total_cost_usd); nothing was billed.
- The exact model IDs used are in each summary.json (model_ids_used).
- Claude Code uses default sampling; temperature is not set. Same for every tier.
- Tools are disabled and customizations are not loaded (--safe-mode), and the model runs in an
  empty folder, so it sees only our prompts.
