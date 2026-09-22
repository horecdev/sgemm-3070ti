# GEMM Benchmark results

Benchmark of custom CUDA GEMM kernels in fp32 against NVIDIA cuBLAS  

**Hardware:** RTX 3070 Ti
**Matrix Dimensions:** M = 2048, K = 8192, N = 1024  

## Results
| Kernel | Time (ms) | Speedup (vs Naive) | Max Diff vs cuBLAS |
| :--- | :--- | :--- | :--- |
| **Naive** | 24.97 | 1.00x | 0.0 |
| **Tiled 16x16** | 18.13 | 1.38x | 0.0 |
| **cuBLAS** | 1.93 | 12.9x | - |

## Takeaway
- Tiling works - using shared memory to reduce VRAM reads by each thread gave a speedup of 38%.
- I got absolutely smoked by cuBLAS. It is nearly 10x faster.
- The math works out and is entirely correct. There is no difference in output between my custom kernels and cuBLAS.