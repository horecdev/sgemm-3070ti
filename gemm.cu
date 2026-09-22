#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>

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

void print_upper_left_square(float* p_matrix, int M, int N, int side_length) {
    if (side_length >= M || side_length >= N) {
        throw std::runtime_error("side_length is too big");
    }
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
int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* host_A = static_cast<float*>(malloc(bytes_A));
    float* host_B = static_cast<float*>(malloc(bytes_B));
    float* host_C = static_cast<float*>(malloc(bytes_C));

    for (int i = 0; i < M * K; ++i) {
        host_A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        host_B[i] = 1.0f;
    }

    float* gpu_A;
    float* gpu_B;
    float* gpu_C;
    cudaMalloc(&gpu_A, bytes_A);
    cudaMalloc(&gpu_B, bytes_B);
    cudaMalloc(&gpu_C, bytes_C);

    cudaMemcpy(gpu_A, host_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, host_B, bytes_B, cudaMemcpyHostToDevice);

    // prep for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 3;
    int runs = 10;
    float milliseconds = 0;

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!
    // BENCHMARK THE NAIVE VERSION

    dim3 threads_per_block(16, 16);
    // x is horizontal, y is vertical. thats why M, N maps to Y, 
    // also we launch one block for one element in the result matrix, so row is M, col is N
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);

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

    cudaMemcpy(host_C, gpu_C, bytes_C, cudaMemcpyDeviceToHost);
    print_upper_left_square(host_C, M, N, 5);

    // !!!!!!!!!!!!!!!!!!!!!!
    // BENCHMARK THE TILED KERNEL

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
    
    cudaMemcpy(host_C, gpu_C, bytes_C, cudaMemcpyDeviceToHost);
    print_upper_left_square(host_C, M, N, 5);

    std::cout << "Finished" << std::endl;

    // FREE MEMORY

    cudaFree(gpu_A); cudaFree(gpu_B); cudaFree(gpu_C);
    free(host_A); free(host_B); free(host_C);
}