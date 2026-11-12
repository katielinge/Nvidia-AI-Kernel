# Transfer test: unseen workloads

Each run's best kernel, unchanged, on 26 workloads it was never developed or tuned on: 19 real routing traces served by a different expert shard (offset +96), and 7 new token counts [3, 24, 128, 256, 512, 2048, 4096] with random routing logits. Same correctness rule (atol 0.01, rtol 0.01, 95% of elements) and timing code; 3 trials x 20 iterations.

## Per run

| Run | Passed | Mean speedup, shifted traces | Mean speedup, new sizes | Failures |
|---|---|---|---|---|
| haiku run1 | 26/26 | 1.32x | 1.29x | none |
| haiku run2 | 26/26 | 1.32x | 1.26x | none |
| haiku run3 | 26/26 | 1.32x | 1.24x | none |
| handwritten run1 | 26/26 | 8.09x | 7.98x | none |
| handwritten run2 | 26/26 | 7.92x | 7.77x | none |
| handwritten run3 | 26/26 | 8.02x | 7.65x | none |
| random run1 | 26/26 | 9.96x | 10.13x | none |
| random run2 | 26/26 | 11.30x | 11.14x | none |
| random run3 | 26/26 | 11.13x | 11.09x | none |
| autotune run1 | 26/26 | 11.75x | 10.63x | none |
| autotune run2 | 26/26 | 11.55x | 10.32x | none |
| autotune run3 | 26/26 | 11.68x | 10.57x | none |
| autotune_fp8 run1 | 26/26 | 12.28x | 11.87x | none |
| autotune_fp8 run2 | 26/26 | 12.13x | 11.17x | none |
| autotune_fp8 run3 | 26/26 | 11.96x | 11.57x | none |
| sonnet run1 | 26/26 | 16.22x | 15.96x | none |
| sonnet run2 | 26/26 | 12.94x | 13.06x | none |
| sonnet run3 | 26/26 | 15.42x | 14.64x | none |
| opus run1 | 26/26 | 18.32x | 20.23x | none |
| opus run2 | 26/26 | 26.89x | 26.18x | none |
| opus run3 | 26/26 | 17.28x | 17.19x | none |

## New token counts: speedup by size (median of runs that passed)

| Tokens | Haiku | Hand-written (untuned) | Random search | Triton autotune | Autotune + FP8 dot | Sonnet | Opus |
|---|---|---|---|---|---|---|---|
| 3 | 0.71x | 5.76x | 5.46x | 4.93x | 5.18x | 7.34x | 6.42x |
| 24 | 1.60x | 10.90x | 13.72x | 14.36x | 15.87x | 22.71x | 20.82x |
| 128 | 1.34x | 8.92x | 14.17x | 14.19x | 15.51x | 21.41x | 26.00x |
| 256 | 1.34x | 9.58x | 15.29x | 15.62x | 16.62x | 22.50x | 26.99x |
| 512 | 1.29x | 10.14x | 14.85x | 13.43x | 15.15x | 17.80x | 29.44x |
| 2048 | 1.25x | 5.46x | 7.48x | 6.13x | 7.18x | 7.54x | 18.95x |
| 4096 | 1.20x | 4.00x | 5.70x | 4.39x | 5.29x | 6.13x | 11.81x |
