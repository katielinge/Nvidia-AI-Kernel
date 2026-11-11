# Agent loop (draft v0.1)

Put this `agent_loop/` folder inside your fork of the starter kit (next to `scripts/`, `solution/`, `config.toml`).

Install: `pip install anthropic` (in the same venv). Set the key: `$env:ANTHROPIC_API_KEY = "..."` (PowerShell).
Never commit the key.

Test with no GPU and no cost (fake model + fake benchmark), run from the repo root:
    python agent_loop/run_agent.py --config agent_loop/run_config.toml --tier haiku --run-id 1 --fake
(Edit `definition_json` in run_config.toml first, or it will not find the reference code.)

Real run (on GAIVI, after the setup steps work):
    python agent_loop/run_agent.py --config agent_loop/run_config.toml --tier sonnet --run-id 1

Output per run goes to agent_runs/<tier>_run<id>/: log.jsonl (one line per attempt), every kernel and reply,
best_kernel.py, summary.json.

Before the first real run, check these (marked as assumptions in the code):
 1. bench.classify_status: status strings from flashinfer-bench ("PASSED" etc.) are guessed.
 2. prompts.load_reference_code: field name of the reference code inside the definition JSON.
 3. config.toml needs `destination_passing_style = false` under [build].
 4. Fill in model IDs and prices in run_config.toml.

Windows notes (found while testing on a real laptop):
 - Set `$env:PYTHONUTF8 = "1"` (or run `setx PYTHONUTF8 1`) before running the starter kit scripts by hand. The loop sets it for you.
 - In config.toml, `entry_point` must be written `kernel.py::kernel`, not just `kernel`.
 - The reference code is under the "reference" field of the definition JSON (confirmed).
