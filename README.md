# Naive vs Tiled vs cuBLAS GEMM kernels.

## Quick summary (it's a short project):
First I wrote a naive implementation of GEMM in CUDA which fires one thread per output element. This one thread reads K elements from both A and B matrix, calculates dot product, and saves the result.

Second I wrote a tiled GEMM kernel. First it splits the output `[M, N]` matrix into tiles of size `[TILE_ROWS, TILE_COLS]`. Then A and B are chopped into patches of size `[TILE_ROWS, K_DEPTH]` and `[K_DEPTH, TILE_COLS]`. Next every thread handles a patch `[THREAD_ROWS, THREAD_COLS]` of the result `[TILE_ROWS, TILE_COLS]` tile. They compute partial dot product stepping over the K_DEPTH dim. This optimizes math done per load and with small enough K_DEPTH we can launch many blocks (because shared memory isnt bloated).

I compared both to cuBLAS to check whether the results are correct, and what speedups cuBLAS / tiled offer. To state it nicely, I got absolutely obliterated by cuBLAS.

## Benchmark!!!

**Hardware:** RTX 3090  
**Matrix Dimensions:** M = 2048, K = 8192, N = 1024  

| Kernel | Time (ms) | % cuBLAS speed | Max Diff vs cuBLAS |
| :--- | :--- | :--- | :--- |
| **Naive** | 17.76 | 9.0% | 0.0 |
| **Tiled 128x128** | 2.56 | 62.4% | 0.0 |
| **cuBLAS** | 1.60 | 100.0% | - |

## Takeaway
- Register tiling and chopping up the K dimension works - it gave a speedup of over 7x against naive.
- I got gapped by cuBLAS - achieved roughly 62.4% of it's speed.
- The math works out and is entirely correct. There is no difference in output between my custom kernels and cuBLAS.

## Build (in powershell):
```powershell
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
.\build\gemm_bench.exe
```
