# Profiler breakdown

Median run of each method (by final speedup), random inputs with contest shapes, per call after warmup. *Idle* = wall-clock time minus GPU-busy time: Python/PyTorch dispatch and kernel-launch overhead while the GPU waits. Times in ms.

## 7 tokens

| Method | Wall ms | Kernel launches | Routing / setup | GEMM1 | GEMM2 | Library GEMM | Idle (launch overhead) |
|---|---|---|---|---|---|---|---|
| Haiku (run2) | 36.305 | 346 | 22.797 (63%) | 0.000 (0%) | 0.000 (0%) | 5.808 (16%) | 7.701 (21%) |
| Hand-written (run2) | 4.960 | 79 | 0.093 (2%) | 1.447 (29%) | 2.254 (45%) | 0.000 (0%) | 1.166 (24%) |
| Random search (run2) | 3.623 | 79 | 0.079 (2%) | 1.267 (35%) | 1.171 (32%) | 0.000 (0%) | 1.105 (30%) |
| Autotune (run1) | 3.377 | 79 | 0.081 (2%) | 0.978 (29%) | 1.188 (35%) | 0.000 (0%) | 1.130 (33%) |
| Autotune + FP8 (run1) | 3.302 | 79 | 0.084 (3%) | 0.850 (26%) | 1.224 (37%) | 0.000 (0%) | 1.144 (35%) |
| Sonnet (run3) | 2.791 | 76 | 0.093 (3%) | 1.054 (38%) | 0.531 (19%) | 0.000 (0%) | 1.113 (40%) |
| Opus (run1) | 2.613 | 74 | 0.089 (3%) | 1.056 (40%) | 0.541 (21%) | 0.000 (0%) | 0.928 (36%) |
| *Opus (run2, not median)* | 1.355 | 10 | 0.013 (1%) | 1.062 (78%) | 0.540 (40%) | 0.000 (0%) | 0.000 (0%) |
| *Sonnet (run1, not median)* | 2.445 | 62 | 0.069 (3%) | 1.116 (46%) | 0.627 (26%) | 0.000 (0%) | 0.633 (26%) |

## 64 tokens

| Method | Wall ms | Kernel launches | Routing / setup | GEMM1 | GEMM2 | Library GEMM | Idle (launch overhead) |
|---|---|---|---|---|---|---|---|
| Haiku (run2) | 87.735 | 688 | 59.469 (68%) | 0.000 (0%) | 0.000 (0%) | 14.007 (16%) | 14.259 (16%) |
| Hand-written (run2) | 12.653 | 115 | 0.109 (1%) | 4.157 (33%) | 6.815 (54%) | 0.000 (0%) | 1.572 (12%) |
| Random search (run2) | 8.312 | 115 | 0.093 (1%) | 3.621 (44%) | 3.391 (41%) | 0.000 (0%) | 1.207 (15%) |
| Autotune (run1) | 8.219 | 115 | 0.100 (1%) | 2.998 (36%) | 3.661 (45%) | 0.000 (0%) | 1.461 (18%) |
| Autotune + FP8 (run1) | 7.999 | 115 | 0.103 (1%) | 2.546 (32%) | 3.902 (49%) | 0.000 (0%) | 1.449 (18%) |
| Sonnet (run3) | 5.868 | 76 | 0.143 (2%) | 2.742 (47%) | 1.657 (28%) | 0.000 (0%) | 1.326 (23%) |
| Opus (run1) | 5.730 | 74 | 0.123 (2%) | 2.826 (49%) | 1.445 (25%) | 0.000 (0%) | 1.336 (23%) |
| *Opus (run2, not median)* | 4.584 | 10 | 0.021 (0%) | 2.987 (65%) | 1.503 (33%) | 0.000 (0%) | 0.074 (2%) |
| *Sonnet (run1, not median)* | 5.858 | 98 | 0.105 (2%) | 3.166 (54%) | 1.800 (31%) | 0.000 (0%) | 0.787 (13%) |

## 1024 tokens

| Method | Wall ms | Kernel launches | Routing / setup | GEMM1 | GEMM2 | Library GEMM | Idle (launch overhead) |
|---|---|---|---|---|---|---|---|
| Haiku (run2) | 135.319 | 821 | 80.458 (59%) | 0.000 (0%) | 0.000 (0%) | 48.323 (36%) | 6.538 (5%) |
| Hand-written (run2) | 20.281 | 125 | 0.417 (2%) | 6.593 (33%) | 12.083 (60%) | 0.000 (0%) | 1.189 (6%) |
| Random search (run2) | 13.215 | 125 | 0.425 (3%) | 6.292 (48%) | 5.428 (41%) | 0.000 (0%) | 1.069 (8%) |
| Autotune (run1) | 16.873 | 125 | 0.423 (3%) | 6.153 (36%) | 9.154 (54%) | 0.000 (0%) | 1.143 (7%) |
| Autotune + FP8 (run1) | 14.623 | 125 | 0.446 (3%) | 4.117 (28%) | 8.728 (60%) | 0.000 (0%) | 1.333 (9%) |
| Sonnet (run3) | 12.688 | 93 | 0.517 (4%) | 3.085 (24%) | 7.945 (63%) | 0.000 (0%) | 1.141 (9%) |
| Opus (run1) | 6.967 | 91 | 0.528 (8%) | 3.080 (44%) | 2.384 (34%) | 0.000 (0%) | 0.976 (14%) |
| *Opus (run2, not median)* | 6.723 | 17 | 0.196 (3%) | 3.283 (49%) | 2.968 (44%) | 0.000 (0%) | 0.276 (4%) |
| *Sonnet (run1, not median)* | 10.893 | 123 | 0.423 (4%) | 3.756 (34%) | 5.716 (52%) | 0.000 (0%) | 0.998 (9%) |

## 8192 tokens

| Method | Wall ms | Kernel launches | Routing / setup | GEMM1 | GEMM2 | Library GEMM | Idle (launch overhead) |
|---|---|---|---|---|---|---|---|
| Haiku (run2) | 243.052 | 860 | 90.559 (37%) | 0.000 (0%) | 0.000 (0%) | 136.916 (56%) | 15.578 (6%) |
| Hand-written (run2) | 94.059 | 142 | 3.231 (3%) | 27.121 (29%) | 63.529 (68%) | 0.000 (0%) | 0.178 (0%) |
| Random search (run2) | 55.845 | 142 | 3.695 (7%) | 29.964 (54%) | 20.514 (37%) | 0.000 (0%) | 1.672 (3%) |
| Autotune (run1) | 92.938 | 142 | 3.552 (4%) | 25.617 (28%) | 61.600 (66%) | 0.000 (0%) | 2.170 (2%) |
| Autotune + FP8 (run1) | 78.464 | 142 | 3.580 (5%) | 11.927 (15%) | 60.743 (77%) | 0.000 (0%) | 2.215 (3%) |
| Sonnet (run3) | 75.552 | 121 | 3.802 (5%) | 13.732 (18%) | 57.227 (76%) | 0.000 (0%) | 0.791 (1%) |
| Opus (run1) | 35.134 | 91 | 3.685 (10%) | 15.424 (44%) | 15.344 (44%) | 0.000 (0%) | 0.681 (2%) |
| *Opus (run2, not median)* | 30.783 | 17 | 2.265 (7%) | 14.436 (47%) | 13.708 (45%) | 0.000 (0%) | 0.373 (1%) |
| *Sonnet (run1, not median)* | 64.285 | 123 | 3.523 (5%) | 16.006 (25%) | 43.296 (67%) | 0.000 (0%) | 1.459 (2%) |

## Top kernels by GPU time (1024 tokens)

**Haiku**: `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kerne` 43.868 ms x129 [Routing / setup]; `_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_tn_a` 35.037 ms x31 [Library GEMM]; `_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_c` 35.024 ms x66 [Routing / setup]; `_ZN7cutlass7Kernel2I42cutlass_80_simt_sgemm_128x64_8x5_tn_al` 9.694 ms x26 [Library GEMM]

**Hand-written**: `_gemm2_scatter` 12.083 ms x32 [GEMM2]; `_gemm1_swiglu` 6.593 ms x32 [GEMM1]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.132 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.086 ms x1 [Routing / setup]

**Random search**: `_gemm1_swiglu` 6.292 ms x32 [GEMM1]; `_gemm2_scatter` 5.428 ms x32 [GEMM2]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.131 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.091 ms x1 [Routing / setup]

**Autotune**: `_gemm2_scatter` 9.154 ms x32 [GEMM2]; `_gemm1_swiglu` 6.153 ms x32 [GEMM1]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.137 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.086 ms x1 [Routing / setup]

**Autotune + FP8**: `_gemm2_scatter` 8.728 ms x32 [GEMM2]; `_gemm1_swiglu` 4.117 ms x32 [GEMM1]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.131 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.105 ms x1 [Routing / setup]

**Sonnet**: `_gemm2_kernel` 7.945 ms x1 [GEMM2]; `_gemm1_kernel` 3.085 ms x1 [GEMM1]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.152 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.087 ms x1 [Routing / setup]

**Opus**: `_gemm1_kernel` 3.080 ms x1 [GEMM1]; `_gemm2_kernel` 2.384 ms x1 [GEMM2]; `_ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS_4cuda6detai` 0.164 ms x3 [Routing / setup]; `_ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_25bflo` 0.087 ms x1 [Routing / setup]

