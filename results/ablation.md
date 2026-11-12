# Knock-out ablation of Sonnet run 1's kernel

Each variant switches one or more of Sonnet's design choices back to the hand-written baseline's choice. Same benchmark for every row: 19 contest workloads, 3 trials x 20 iterations, cached reference timings, same correctness rule.

| Variant | What changed | Passed | Mean speedup | vs. original |
|---|---|---|---|---|
| original | Sonnet run 1 best kernel, unchanged | 19/19 | 16.82x | 1.00 |
| no_fp8 | GEMM1 converts FP8 to bf16 before the dot | 19/19 | 16.30x | 0.97 |
| no_gemm2_split | GEMM2 uses fp32 operands with tf32x3 (baseline's choice) | 19/19 | 13.26x | 0.79 |
| no_adaptive_tiles | Tile height fixed at 32 (baseline's choice) | 19/19 | 14.69x | 0.87 |
| no_fp8_no_gemm2 | Both GEMM changes off; adaptive tiles kept | 19/19 | 12.28x | 0.73 |
| all_three_off | All three off: the baseline's design and config | 19/19 | 8.81x | 0.52 |
