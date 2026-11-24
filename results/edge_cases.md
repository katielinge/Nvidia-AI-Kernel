# Edge-case correctness test

Each run's best kernel, unchanged, on 11 workloads built to stress it. Correctness rule as everywhere else (atol 0.01, rtol 0.01, 95% of elements); timing 3 warmup x 10 iters x 1 trial, so speedups are indicative only.

| Case | Tokens | What it tests |
|---|---|---|
| hot_8_experts | 256 | 8 of 32 local experts get every token |
| skewed_75pct | 1024 | 75% of tokens on 8 local experts |
| one_token_each | 32 | every local expert gets exactly 1 token |
| no_local_tokens | 64 | correct output is all zeros |
| odd_17 | 17 | random routing, real bias |
| odd_129 | 129 | random routing, real bias |
| odd_1000 | 1000 | random routing, real bias |
| odd_4097 | 4097 | random routing, real bias |
| scale_large | 64 | activation scales all 100 |
| scale_small | 64 | activation scales all 0.01 |
| scale_mixed | 64 | scales log-uniform 1e-3..1e3 |

## Runs that passed each case

| Case | Haiku | Hand-written (untuned) | Random search | Triton autotune | Autotune + FP8 dot | Autotune + agents' ideas | Sonnet | Opus |
|---|---|---|---|---|---|---|---|---|
| hot_8_experts | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| skewed_75pct | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| one_token_each | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| no_local_tokens | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| odd_17 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| odd_129 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| odd_1000 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| odd_4097 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| scale_large | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| scale_small | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| scale_mixed | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

## Median speedup on each case (runs that passed)

| Case | Haiku | Hand-written (untuned) | Random search | Triton autotune | Autotune + FP8 dot | Autotune + agents' ideas | Sonnet | Opus |
|---|---|---|---|---|---|---|---|---|
| hot_8_experts | 1.07x | 2.87x | 3.91x | 2.97x | 3.29x | 6.10x | 4.36x | 6.80x |
| skewed_75pct | 1.13x | 2.78x | 4.55x | 3.56x | 4.04x | 7.13x | 4.91x | 9.34x |
| one_token_each | 1.37x | 8.47x | 11.73x | 13.28x | 14.28x | 18.18x | 18.77x | 21.75x |
| no_local_tokens | 1.15x | 5.05x | 3.72x | 3.56x | 3.39x | 2.91x | 4.06x | 2.68x |
| odd_17 | 1.50x | 7.57x | 10.96x | 10.39x | 11.23x | 11.50x | 13.51x | 17.11x |
| odd_129 | 1.35x | 8.55x | 12.87x | 13.84x | 15.45x | 19.62x | 20.95x | 23.10x |
| odd_1000 | 1.23x | 9.15x | 11.85x | 10.18x | 12.31x | 16.31x | 14.71x | 21.87x |
| odd_4097 | 1.23x | 3.96x | 5.65x | 4.50x | 5.24x | 8.24x | 6.13x | 11.60x |
| scale_large | 1.37x | 9.70x | 13.15x | 15.27x | 15.84x | 20.79x | 21.27x | 28.23x |
| scale_small | 1.08x | 7.96x | 10.68x | 12.09x | 11.63x | 16.99x | 18.06x | 17.31x |
| scale_mixed | 1.30x | 8.61x | 12.18x | 13.95x | 12.29x | 17.65x | 17.84x | 21.86x |

## Every failure

None: every kernel passed every case.
