# Results: Claude tiers on the FlashInfer MoE kernel (RTX 5060 Laptop GPU)

Speedup = patched PyTorch reference latency / kernel latency, mean over all 19 workloads, full benchmark (3 warmup, 100 iterations, 5 trials). A run's speedup counts only if its best kernel passes all 19 workloads. Identical code measured 1.00-1.03x, so differences under ~3% are noise. Cost is Claude Code's API-equivalent estimate; runs used a subscription and were not billed.

## Per tier

| Tier | Runs | Runs with a correct kernel | Final speedup, median (min-max) | First correct attempt, median | API-equiv. cost per run, median | Model time per run, median (min) |
|---|---|---|---|---|---|---|
| haiku | 3 | 3 | 1.44x (1.43-1.45) | 1 | $0.03 | 2.8 |
| sonnet | 3 | 3 | 17.54x (14.73-18.00) | 2 | $0.98 | 6.1 |
| opus | 3 | 3 | 20.17x (18.46-28.53) | 3 | $3.07 | 14.7 |

## Per run

| Tier | Run | Model | Attempts | First correct | Loop best | Final speedup | Final status | In/out tokens | API-equiv. $ | $ per 1.0x gain | Model min | Bench min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| haiku | 1 | claude-haiku-5-5 | 15 | 2 | 1.51x | 1.435x | CORRECT | 69867/45018 | 0.03 | 0.08 | 3.0 | 20.1 |
| haiku | 2 | claude-haiku-5-5 | 15 | 1 | 1.57x | 1.440x | CORRECT | 72658/43108 | 0.03 | 0.08 | 2.8 | 22.0 |
| haiku | 3 | claude-haiku-5-5 | 15 | 1 | 1.52x | 1.448x | CORRECT | 72352/40681 | 0.03 | 0.07 | 2.7 | 22.4 |
| sonnet | 1 | claude-sonnet-5-5 | 15 | 2 | 18.61x | 18.003x | CORRECT | 93112/63417 | 0.98 | 0.06 | 6.1 | 14.4 |
| sonnet | 2 | claude-sonnet-5-5 | 15 | 3 | 15.23x | 14.734x | CORRECT | 85737/58547 | 0.90 | 0.07 | 5.6 | 13.1 |
| sonnet | 3 | claude-sonnet-5-5 | 15 | 2 | 17.54x | 17.545x | CORRECT | 103561/70399 | 1.09 | 0.07 | 6.6 | 23.4 |
| opus | 1 | claude-opus-5-5 | 15 | 2 | 21.28x | 20.175x | CORRECT | 133200/91339 | 2.83 | 0.15 | 12.0 | 9.8 |
| opus | 2 | claude-opus-5-5 | 15 | 3 | 29.95x | 28.535x | CORRECT | 124287/107020 | 3.07 | 0.11 | 14.7 | 10.5 |
| opus | 3 | claude-opus-5-5 | 15 | 3 | 19.70x | 18.457x | CORRECT | 135058/112239 | 3.26 | 0.19 | 14.7 | 16.0 |
