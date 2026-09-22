cmake_minimum_required(VERSION 3.18)

project(gemm3090 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86) # so it find 3070 Ti

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench gemm.cu)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)