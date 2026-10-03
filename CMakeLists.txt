cmake_minimum_required(VERSION 3.22)
project(cuda_operator_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4090 Laptop GPU: Ada, compute capability 8.9.
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")

add_subdirectory(csrc)
