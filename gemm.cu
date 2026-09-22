#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

__global__ void sgemm_naive(int M, int K, int N, float* p_A, float* p_B, float* p_C) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < K; ++i) {
            sum += p_A[row * K + i] * p_B[i * N + col];
        }

        p_C[row * N + col] = sum;
    }
}

#define TILE_SIZE 32 // must be known at compile time 
// launches one thread for every output elem in C just like naive

// logically explaining. To calculate a 16x16 patch of C, the block needs a horizontal stripe of A (16 rows tall, 1024 cols wide)
// and a vertical stripe of B (1024 rows tall, 16 cols wide)

// the big ass 16x1024 or 1024x16 is chopped into 16x16 tiles that are sliding across K (since to calculate elems you go over whole K)
__global__ void sgemm_tiled(int M, int K, int N, float* p_A, float* p_B, float* p_C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f; // one for each thread

    int total_phases = (K + TILE_SIZE - 1) / TILE_SIZE; // round it up
    for (int phase = 0; phase < total_phases; ++phase) {
        if (row < M && (phase * TILE_SIZE + threadIdx.x) < K) { // didnt go off bounds and viable to slide
            tile_A[threadIdx.y][threadIdx.x] = p_A[row * K + (phase * TILE_SIZE + threadIdx.x)];
        }
        else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f; // got off bounds (tile OR row)
            // you can set to 0.0f because later you multiply with respective piece from B (also went off bounds, cuz its both K) and 0 * 0 = 0 in sum accum. 
            // There is no difference.
        }

        if ((phase * TILE_SIZE + threadIdx.y) < K && col < N) { // same bounds (tile OR col)
            tile_B[threadIdx.y][threadIdx.x] = p_B[(phase * TILE_SIZE + threadIdx.y) * N + col];
        }
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // so its all filled up

        // now you REUSE numbers that OTHER THREADS loaded up to tiles.
        // load from A across K horizontal, from B across K vertical
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        // must sync bc some threads may finish faster and overwrite shared memory
        __syncthreads();
    }

    if (row < M && col < N) {
        p_C[row * N + col] = sum;
    }
}

// say this is the part of C you are calculating (left) C = AB
// then the tiles created are on the right
//       C                      A                       B
// [             ]       [               ]       [       [  1  ]  ]
// [             ]       [               ]       [       [     ]  ]
// [      [   ]  ]       [[ 1 ][ 2 ][ 3 ]]       [       [  2  ]  ]
// [      [   ]  ]       [[   ][   ][   ]]       [       [     ]  ]
// [             ]       [               ]       [       [  3  ]  ]
// [             ]       [               ]       [       [     ]  ]
// this is so brilliant, because one thread used to load K times from A and K times from B.
// now it loads once per tile for A and B. Thats K / TILE_SIZE reads from VRAM. TILE_SIZE TIMES LESS.


void print_upper_left_square(float* p_matrix, int M, int N, int side_length) {
    if (side_length >= M || side_length >= N) {
        throw std::runtime_error("side_length is too big");
    }
    std::cout << "Upper left " << side_length << "x" << side_length << " square:" << std::endl;
    for (int i = 0; i < side_length; ++i) {
        std::cout << "[";
        for (int j = 0; j < side_length; ++j) {
            std::cout << p_matrix[i * N + j];
            if (j != side_length - 1) {
                std::cout << ", ";
            }
        }
        std::cout <<"]" << std::endl;
    }
}

float max_difference(float* p_m1, float* p_m2, int num_elems) {
    float max_diff = 0;

    for (int i = 0; i < num_elems; ++i) {
        max_diff = std::max(max_diff, std::abs(p_m1[i] - p_m2[i]));
    }

    return max_diff;
}

int main() {
    int M = 2048;
    int N = 1024;
    int K = 8192;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* host_A = static_cast<float*>(malloc(bytes_A));
    float* host_B = static_cast<float*>(malloc(bytes_B));
    float* host_C_naive = static_cast<float*>(malloc(bytes_C));
    float* host_C_tiled = static_cast<float*>(malloc(bytes_C));
    float* host_C_cublas = static_cast<float*>(malloc(bytes_C));

    for (int i = 0; i < M * K; ++i) {
        host_A[i] = static_cast<float>(rand() % 7 - 3);
    }
    for (int i = 0; i < K * N; ++i) {
        host_B[i] = static_cast<float>(rand() % 7 - 3);
    }

    float* gpu_A;
    float* gpu_B;
    float* gpu_C;
    cudaMalloc(&gpu_A, bytes_A);
    cudaMalloc(&gpu_B, bytes_B);
    cudaMalloc(&gpu_C, bytes_C);

    cudaMemcpy(gpu_A, host_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, host_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    // x is horizontal, y is vertical. thats why M, N maps to Y, 
    // also we launch one block for one element in the result matrix, so row is M, col is N
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);

    // prep for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 3;
    int runs = 10;
    float milliseconds = 0;

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!
    // BENCHMARK THE NAIVE VERSION

    for (int i = 0; i < warmup; ++i) {
        sgemm_naive<<<num_blocks, threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaDeviceSynchronize();
    std::cout << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        sgemm_naive<<<num_blocks, threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Naive GEMM: " << (milliseconds / runs) << "ms" << std::endl;

    cudaMemcpy(host_C_naive, gpu_C, bytes_C, cudaMemcpyDeviceToHost);

    // !!!!!!!!!!!!!!!!!!!!!!
    // BENCHMARK THE TILED KERNEL

    for (int i = 0; i < warmup; ++i) {
        sgemm_tiled<<<num_blocks, threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaDeviceSynchronize();
    std::cout << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        sgemm_tiled<<<num_blocks, threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tiled GEMM: " << (milliseconds / runs) << "ms" << std::endl;

    cudaMemcpy(host_C_tiled, gpu_C, bytes_C, cudaMemcpyDeviceToHost);

    // !!!!!!!!!!!!!!!!
    // BENCHMARK CUBLAS

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    // cublas is col major bruh so reverse order (A @ B = B^T @ A^T)
    for (int i = 0; i < warmup; ++i) {
        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            gpu_B, N,
            gpu_A, K,
            &beta,
            gpu_C, N
        );
    }
    cudaDeviceSynchronize();
    std::cout << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            gpu_B, N,
            gpu_A, K,
            &beta,
            gpu_C, N
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "cuBLAS GEMM: " << (milliseconds / runs) << "ms" << std::endl;
    
    cudaMemcpy(host_C_cublas, gpu_C, bytes_C, cudaMemcpyDeviceToHost);

    std::cout << "Max differences: " << std::endl;
    std::cout << "Naive vs cuBLAS: " << max_difference(host_C_naive, host_C_cublas, M * N) << std::endl;
    std::cout << "Tiled vs cuBLAS: " << max_difference(host_C_tiled, host_C_cublas, M * N) << std::endl;

    // FREE MEMORY

    cudaFree(gpu_A); cudaFree(gpu_B); cudaFree(gpu_C);
    free(host_A); free(host_B); free(host_C_naive); free(host_C_tiled); free(host_C_cublas);
}