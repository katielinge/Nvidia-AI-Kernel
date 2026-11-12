# Per-workload analysis

Speedup = patched PyTorch reference latency / kernel latency, full benchmark (5 trials x 100 iterations). Each cell is the median of 3 runs.

| Tokens | Workload | Reference ms | Haiku | Hand-written (untuned) | Random search | Triton autotune | Sonnet | Opus | Autotune + FP8 dot (ablation) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | e05c6c03 | 19.1 | 1.49x | 6.14x | 7.19x | 7.39x | 8.82x | 10.57x | 7.53x |
| 7 | b8f4f012 | 38.5 | 1.57x | 7.22x | 9.03x | 9.94x | 13.10x | 16.15x | 10.89x |
| 14 | 8cba5890 | 61.7 | 1.48x | 8.54x | 11.44x | 12.44x | 17.39x | 19.08x | 13.21x |
| 15 | 2e69caee | 29.2 | 1.50x | 7.40x | 8.89x | 9.50x | 12.39x | 13.32x | 10.21x |
| 16 | a7c2bcfd | 67.6 | 1.47x | 8.78x | 11.93x | 12.94x | 19.48x | 20.56x | 13.16x |
| 32 | 6230e838 | 108.5 | 1.45x | 9.36x | 12.49x | 13.83x | 21.66x | 24.25x | 14.75x |
| 52 | f7d6ac7c | 71.2 | 1.20x | 7.56x | 10.41x | 11.04x | 17.18x | 18.91x | 11.65x |
| 53 | fc378037 | 114.2 | 1.40x | 9.03x | 12.80x | 13.63x | 21.42x | 24.11x | 14.17x |
| 54 | 76010cb4 | 118.6 | 1.51x | 9.76x | 13.76x | 14.76x | 22.50x | 25.51x | 15.26x |
| 55 | 81955b1e | 121.9 | 1.48x | 9.57x | 13.62x | 14.51x | 22.36x | 25.41x | 15.39x |
| 56 | 4822167c | 131.9 | 1.48x | 9.70x | 13.84x | 14.59x | 22.74x | 25.60x | 15.37x |
| 57 | 74d7ff04 | 126.5 | 1.47x | 9.56x | 13.87x | 14.70x | 22.78x | 25.19x | 15.24x |
| 58 | e626d3e6 | 130.9 | 1.46x | 9.43x | 13.81x | 14.51x | 23.08x | 25.47x | 14.90x |
| 59 | eedc63b2 | 95.7 | 1.45x | 9.22x | 12.83x | 13.54x | 21.00x | 22.74x | 14.46x |
| 62 | 5eadab1e | 91.8 | 1.44x | 9.14x | 12.65x | 13.37x | 21.22x | 22.81x | 14.31x |
| 80 | 8f1ff9f1 | 138.9 | 1.44x | 9.24x | 13.32x | 14.25x | 23.02x | 25.62x | 14.78x |
| 901 | 1a4c6ba1 | 195.8 | 1.39x | 7.75x | 10.80x | 9.06x | 12.81x | 23.27x | 10.58x |
| 11948 | 58a34f27 | 357.3 | 1.30x | 2.73x | 3.93x | 2.91x | 4.45x | 8.42x | 3.27x |
| 14107 | 5e8dc11c | 461.1 | 1.27x | 2.37x | 3.44x | 2.58x | 4.01x | 7.50x | 2.98x |

## Worst workload per run

| Run | Mean speedup | Worst workload (tokens) | Worst speedup |
|---|---|---|---|
| haiku run1 | 1.43x | f7d6ac7c (52) | 1.20x |
| haiku run2 | 1.44x | f7d6ac7c (52) | 1.20x |
| haiku run3 | 1.45x | f7d6ac7c (52) | 1.20x |
| handwritten run1 | 8.02x | 5e8dc11c (14107) | 2.37x |
| handwritten run2 | 8.02x | 5e8dc11c (14107) | 2.37x |
| handwritten run3 | 8.05x | 5e8dc11c (14107) | 2.37x |
| random run1 | 9.88x | 5e8dc11c (14107) | 3.00x |
| random run2 | 11.18x | 5e8dc11c (14107) | 4.11x |
| random run3 | 11.33x | 5e8dc11c (14107) | 3.44x |
| autotune run1 | 11.51x | 5e8dc11c (14107) | 2.58x |
| autotune run2 | 11.48x | 5e8dc11c (14107) | 2.58x |
| autotune run3 | 11.84x | 5e8dc11c (14107) | 2.55x |
| sonnet run1 | 18.00x | 5e8dc11c (14107) | 4.01x |
| sonnet run2 | 14.73x | 5e8dc11c (14107) | 4.59x |
| sonnet run3 | 17.54x | 5e8dc11c (14107) | 3.18x |
| opus run1 | 20.17x | 5e8dc11c (14107) | 6.97x |
| opus run2 | 28.53x | 5e8dc11c (14107) | 7.50x |
| opus run3 | 18.46x | 5e8dc11c (14107) | 7.78x |
| autotune_fp8 run1 | 12.19x | 5e8dc11c (14107) | 3.01x |
| autotune_fp8 run2 | 12.01x | 5e8dc11c (14107) | 2.96x |
| autotune_fp8 run3 | 12.35x | 5e8dc11c (14107) | 2.98x |

## Claude vs. best non-ML baseline (autotune), by workload size

| Tokens | Haiku / autotune | Sonnet / autotune | Opus / autotune |
|---|---|---|---|
| 1 | 0.20 | 1.19 | 1.43 |
| 7 | 0.16 | 1.32 | 1.62 |
| 14 | 0.12 | 1.40 | 1.53 |
| 15 | 0.16 | 1.30 | 1.40 |
| 16 | 0.11 | 1.51 | 1.59 |
| 32 | 0.10 | 1.57 | 1.75 |
| 52 | 0.11 | 1.56 | 1.71 |
| 53 | 0.10 | 1.57 | 1.77 |
| 54 | 0.10 | 1.52 | 1.73 |
| 55 | 0.10 | 1.54 | 1.75 |
| 56 | 0.10 | 1.56 | 1.75 |
| 57 | 0.10 | 1.55 | 1.71 |
| 58 | 0.10 | 1.59 | 1.76 |
| 59 | 0.11 | 1.55 | 1.68 |
| 62 | 0.11 | 1.59 | 1.71 |
| 80 | 0.10 | 1.62 | 1.80 |
| 901 | 0.15 | 1.41 | 2.57 |
| 11948 | 0.45 | 1.53 | 2.90 |
| 14107 | 0.49 | 1.56 | 2.91 |

## Design features of each best kernel (automatic scan; spot-checked by hand)

FP8 dot = GEMM1 multiplies FP8 values directly on the tensor cores. Sonnet run 1 and Opus run 3 try FP8 first and fall back to bf16 only if FP8 raises an error.

| Run | Final speedup | Triton kernels | One grouped launch for all experts | FP8 tensor-core dot in GEMM1 | Routing in Triton |
|---|---|---|---|---|---|
| haiku run1 | 1.43x | 0 | no | no | no |
| haiku run2 | 1.44x | 0 | no | no | no |
| haiku run3 | 1.45x | 0 | no | no | no |
| handwritten run1 | 8.02x | 2 | no | no | no |
| handwritten run2 | 8.02x | 2 | no | no | no |
| handwritten run3 | 8.05x | 2 | no | no | no |
| random run1 | 9.88x | 2 | no | no | no |
| random run2 | 11.18x | 2 | no | no | no |
| random run3 | 11.33x | 2 | no | no | no |
| autotune run1 | 11.51x | 2 | no | no | no |
| autotune run2 | 11.48x | 2 | no | no | no |
| autotune run3 | 11.84x | 2 | no | no | no |
| sonnet run1 | 18.00x | 2 | no | yes | no |
| sonnet run2 | 14.73x | 2 | no | yes | no |
| sonnet run3 | 17.54x | 2 | yes | yes | no |
| opus run1 | 20.17x | 2 | yes | yes | no |
| opus run2 | 28.53x | 4 | yes | yes | yes |
| opus run3 | 18.46x | 3 | yes | yes | no |
| autotune_fp8 run1 | 12.19x | 2 | no | yes | no |
| autotune_fp8 run2 | 12.01x | 2 | no | yes | no |
| autotune_fp8 run3 | 12.35x | 2 | no | yes | no |
