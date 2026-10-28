#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <driver_types.h>
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

#define TILE_ROWS 128
#define TILE_COLS 128
#define K_DEPTH 16
#define THREAD_ROWS 8
#define THREAD_COLS 8
#define VECTORIZED_LOAD 4

// launches one thread for [THREAD_ROW, THREAD_COL] patch of output
// (TILE_SIZE * TILE_SIZE) / (THREAD_ROWS * THREAD_COLS) threads per patch
#define THREADS_PER_TILE (TILE_ROWS * TILE_COLS) / (THREAD_ROWS * THREAD_COLS)

// the big ass 16x1024 or 1024x16 is chopped into 16x16 tiles that are sliding across K (since to calculate elems you go over whole K)
__global__ void sgemm_tiled(int M, int K, int N, float* p_A, float* p_B, float* p_C) {
    __shared__ __align__(16) float tile_A[K_DEPTH][TILE_ROWS]; // tile_A is stored transposed.
    __shared__ __align__(16) float tile_B[K_DEPTH][TILE_COLS];

    float C_reg[THREAD_ROWS][THREAD_COLS] = {0.0f};

    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;

    int num_loads_A = TILE_ROWS * K_DEPTH / VECTORIZED_LOAD;
    int loads_per_row_A = K_DEPTH / VECTORIZED_LOAD;

    int num_loads_B = K_DEPTH * TILE_COLS / VECTORIZED_LOAD;
    int loads_per_row_B = TILE_COLS / VECTORIZED_LOAD;

    int total_phases = (K + K_DEPTH - 1) / K_DEPTH; // round it up
    for (int phase = 0; phase < total_phases; ++phase) {

        // LOAD A
        for (int load_idx = flat_tid; load_idx < num_loads_A; load_idx += THREADS_PER_TILE) {
            int start_elem_idx = load_idx * VECTORIZED_LOAD;
            int local_row = start_elem_idx / K_DEPTH;
            int local_col = start_elem_idx % K_DEPTH;

            int global_row_A = blockIdx.y * TILE_ROWS + local_row;
            int global_col_A = phase * K_DEPTH + local_col;

            if (global_row_A < M && global_col_A < K) {
                // cannot vectorize this load - its different rows, no stride of 1.
                float4 chunk_A = reinterpret_cast<float4*>(&p_A[global_row_A * K + global_col_A])[0];
                tile_A[local_col + 0][local_row] = chunk_A.x;
                tile_A[local_col + 1][local_row] = chunk_A.y;
                tile_A[local_col + 2][local_row] = chunk_A.z;
                tile_A[local_col + 3][local_row] = chunk_A.w;
            }
            else {
                tile_A[local_col + 0][local_row] = 0.0f;
                tile_A[local_col + 1][local_row] = 0.0f;
                tile_A[local_col + 2][local_row] = 0.0f;
                tile_A[local_col + 3][local_row] = 0.0f;
            }
        }


        // LOAD B
        for (int load_idx = flat_tid; load_idx < num_loads_B; load_idx += THREADS_PER_TILE) {
            int start_elem_idx = load_idx * VECTORIZED_LOAD;
            int local_row = start_elem_idx / TILE_COLS;
            int local_col = start_elem_idx % TILE_COLS;

            int global_row_B = phase * K_DEPTH + local_row;
            int global_col_B = blockIdx.x * TILE_COLS + local_col;

            if (global_row_B < K && global_col_B < N) {
                float4 chunk_B = reinterpret_cast<float4*>(&p_B[global_row_B * N + global_col_B])[0];
                reinterpret_cast<float4*>(&tile_B[local_row][local_col])[0] = chunk_B;
                // its faster than unloading chunks one coord by one, because doing (smth) = chunk.x introduces 4-way bank conflict. 
                // then .y, .z, .w too. float4 loads/saves are executed 8 lanes at a time
                // in reality it didn't really save time. U could just load tile_B[local_row][local_col + 0] = chunk.x, so on.
            }
            else {
                reinterpret_cast<float4*>(&tile_B[local_row][local_col])[0] = make_float4(0, 0, 0, 0);
            }
        }
        __syncthreads(); // so its all filled up

        float A_reg[THREAD_ROWS];
        float B_reg[THREAD_COLS];
        
        for (int k = 0; k < K_DEPTH; ++k) {
            for (int load_idx_A = 0; load_idx_A < THREAD_ROWS / VECTORIZED_LOAD; ++load_idx_A) {
                // tile_A IS TRANSPOSED SO IN REALITY WE ARE ACCESSING tile_A[smth][k]
                float4 chunk_A = reinterpret_cast<float4*>(&tile_A[k][threadIdx.y * THREAD_ROWS + load_idx_A * VECTORIZED_LOAD])[0];
                A_reg[load_idx_A * VECTORIZED_LOAD + 0] = chunk_A.x;
                A_reg[load_idx_A * VECTORIZED_LOAD + 1] = chunk_A.y;
                A_reg[load_idx_A * VECTORIZED_LOAD + 2] = chunk_A.z;
                A_reg[load_idx_A * VECTORIZED_LOAD + 3] = chunk_A.w;
            }
            for (int load_idx_B = 0; load_idx_B < THREAD_COLS / VECTORIZED_LOAD; ++load_idx_B) {
                float4 chunk_B = reinterpret_cast<float4*>(&tile_B[k][threadIdx.x * THREAD_COLS + load_idx_B * VECTORIZED_LOAD])[0];
                B_reg[load_idx_B * VECTORIZED_LOAD + 0] = chunk_B.x;
                B_reg[load_idx_B * VECTORIZED_LOAD + 1] = chunk_B.y;
                B_reg[load_idx_B * VECTORIZED_LOAD + 2] = chunk_B.z;
                B_reg[load_idx_B * VECTORIZED_LOAD + 3] = chunk_B.w;
            }
            for (int i = 0; i < THREAD_ROWS; ++i) {
                for (int j = 0; j < THREAD_COLS; ++j) {
                    C_reg[i][j] += A_reg[i] * B_reg[j];
                }
            }
        }

        // must sync bc some threads may finish faster and overwrite shared memory
        __syncthreads();
    }
    
    int start_global_row_C = blockIdx.y * TILE_ROWS + threadIdx.y * THREAD_ROWS;
    int start_global_col_C = blockIdx.x * TILE_COLS + threadIdx.x * THREAD_COLS;

    for (int i = 0; i < THREAD_ROWS; i++) {
        for (int j = 0; j < THREAD_COLS; j++) {
            int global_row = start_global_row_C + i;
            int global_col = start_global_col_C + j;

            if (global_row < M && global_col < N) {
                p_C[global_row * N + global_col] = C_reg[i][j];
            }
        }
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

    dim3 naive_num_blocks((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    dim3 tiled_num_blocks((N + TILE_COLS - 1) / TILE_COLS, (M + TILE_ROWS - 1) / TILE_ROWS);

    dim3 naive_threads_per_block(32, 32);
    dim3 tiled_threads_per_block(TILE_COLS / THREAD_COLS, TILE_ROWS / THREAD_ROWS);

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
        sgemm_naive<<<naive_num_blocks, naive_threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaDeviceSynchronize();
    std::cout << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        sgemm_naive<<<naive_num_blocks, naive_threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Naive GEMM: " << (milliseconds / runs) << "ms" << std::endl;

    cudaMemcpy(host_C_naive, gpu_C, bytes_C, cudaMemcpyDeviceToHost);

    // !!!!!!!!!!!!!!!!!!!!!!
    // BENCHMARK THE TILED KERNEL

    for (int i = 0; i < warmup; ++i) {
        sgemm_tiled<<<tiled_num_blocks, tiled_threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
    }
    cudaDeviceSynchronize();
    std::cout << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        sgemm_tiled<<<tiled_num_blocks, tiled_threads_per_block>>>(M, K, N, gpu_A, gpu_B, gpu_C);
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