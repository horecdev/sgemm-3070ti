# Naive vs Tiled vs cuBLAS GEMM kernels.

## Quick summary (it's a short project):
First I wrote a naive implementation of GEMM in CUDA which fires one thread per output element. This one thread reads K elements from both A and B matrix, calculates dot product, and saves the result.

Second I wrote a tiled GEMM kernel, that doesn't force every single thread to read all elements needed to produce a result. It splits matrices into tile and uses shared memory to load up elements once per tile. All threads use this shared buffer to accumulate partial dot product into their local registers. After all tiles are processed and full product is accumulated, each thread saves the result.

Also I compared both to cuBLAS to check whether the results are correct, and what speedups cuBLAS / tiled offer. To state it nicely, I got absolutely obliterated by cuBLAS.

## Benchmark!!!

**Hardware:** RTX 3070 Ti  
**Matrix Dimensions:** M = 2048, K = 8192, N = 1024  

| Kernel | Time (ms) | Speedup (vs Naive) | Max Diff vs cuBLAS |
| :--- | :--- | :--- | :--- |
| **Naive** | 24.97 | 1.00x | 0.0 |
| **Tiled 16x16** | 18.13 | 1.38x | 0.0 |
| **Tiled 32x32** | 20.42 | 1.22x | 0.0 |
| **cuBLAS** | 1.93 | 12.9x | - |

## Takeaway
- Tiling works - using shared memory to reduce VRAM reads by each thread gave a speedup of 38%.
- I got absolutely smoked by cuBLAS. It is almost 10x faster.
- The math works out and is entirely correct. There is no difference in output between my custom kernels and cuBLAS.

## Build (in powershell):
```powershell
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
.\build\gemm_bench.exe
```
